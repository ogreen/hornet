#include "Hornet.hpp"

#include <Host/FileUtil.hpp>            //xlib::extract_filepath_noextension
#include <Device/Util/CudaUtil.cuh>          //xlib::deviceInfo
#include <algorithm>                    //std:.generate
#include <chrono>                       //std::chrono
#include <random>                       //std::mt19937_64
#include <cuda_profiler_api.h>
//nvprof --profile-from-start off --log-file log.txt --print-gpu-trace


#include "cudf.h"


#include "HornetAlg.cuh"
#include "Operator++.cuh"
#include "Queue/TwoLevelQueue.cuh"
#include "StandardAPI.hpp"
#include "HostDeviceVar.cuh"
#include "LoadBalancing/VertexBased.cuh"
#include "LoadBalancing/ScanBased.cuh"
#include "LoadBalancing/BinarySearch.cuh"

#include <cub/cub.cuh>

#include <cub/util_allocator.cuh>

#include <moderngpu/kernel_segsort.hxx>
using namespace mgpu;


using namespace hornets_nest;
using namespace timer;
using namespace std::string_literals;

using HornetGPU = hornets_nest::gpu::Hornet<EMPTY, EMPTY>;

void exec(int argc, char* argv[]);

/**
 * @brief Example tester for Hornet
 */
int main(int argc, char* argv[]) {
    exec(argc, argv);
    cudaDeviceReset();
}



struct getPosition {
    eoff_t *d_positions;
    eoff_t *d_edgeCounter;


	OPERATOR(Vertex& vertex_src) {
    	auto degree = vertex_src.degree();
        auto    src = vertex_src.id();

        auto pos = atomicAdd(d_edgeCounter,degree);
        d_positions[src] = pos;
    }
};

struct getPositionSorted {
    eoff_t *d_sizes;

	OPERATOR(Vertex& vertex_src) {
    	auto degree = vertex_src.degree();
        auto    src = vertex_src.id();

        d_sizes[src] = degree;
    }
};


template<typename T>
struct convertToCOO {
    eoff_t *d_positions;
    T* srcArray;
    T* destArray;

	OPERATOR(Vertex& vertex_src, Edge& edge){
        auto    src 	= vertex_src.id();
        auto    dest 	= edge.dst_id();
        auto 	pos 	= atomicAdd(d_positions+src,1);
        srcArray[pos] 	= src;
        destArray[pos] 	= dest;
    }
};

template<typename T>
struct convertToCOOSorted {
    eoff_t *d_positions;
    T* srcArray;
    T* destArray;

	OPERATOR(Vertex& vertex_src, Edge& edge){
        auto    src 	= vertex_src.id();
        auto    dest 	= edge.dst_id();
        auto 	pos		= atomicAdd(d_positions+src,1);
        srcArray[pos] 	= src;
        destArray[pos] 	= dest;
    }
};

template<typename T>
struct convertToCSR {
    eoff_t *d_positions;
    T* destArray;

	OPERATOR(Vertex& vertex_src, Edge& edge){
        auto    src 	= vertex_src.id();
        auto    dest 	= edge.dst_id();
        auto 	pos 	= atomicAdd(d_positions+src,1);
        destArray[pos] 	= dest;
    }
};

/*
typedef struct gdf_column_{
    void *data;                       
    gdf_valid_type *valid;            
    gdf_size_type size;               
    gdf_dtype dtype;                  
    gdf_size_type null_count;         
    gdf_dtype_extra_info dtype_info;
    char *			col_name;
} gdf_column;

*/



template <typename T>
gdf_error allocateGdfColumn(gdf_column *gdf, size_t elements) {

	gdf->size=(gdf_size_type)elements;

	// int elementSize=0;
	if (typeid(T) == typeid(int32_t)){
		gdf->dtype = GDF_INT32;
	} else 	if (typeid(T) == typeid(int64_t)){
		gdf->dtype = GDF_INT32;
	}
	else {
		printf("Unsupported data type in conversion to gdf column\n");
		return GDF_UNSUPPORTED_DTYPE; // Graph vertices can be only 32 or 64 bit
	}

	gpu::allocate((T*&)(gdf->data),elements);

	size_t num_bitmaps = ((gdf_size_type)elements + 31) / 8;			// 8 bytes per bitmap
	gpu::allocate((gdf_valid_type*&)(gdf->valid),num_bitmaps);
	cudaMemset(gdf->valid, 0XFF, (sizeof(gdf_valid_type) 	* num_bitmaps));

	gdf->null_count=0;
	return gdf_error::GDF_SUCCESS;
}

gdf_error freeGdfColumn(gdf_column *gdf) {
	gpu::free(gdf->data);
	gpu::free(gdf->valid);
	return gdf_error::GDF_SUCCESS;
}

gdf_column* HornetToCOO(HornetGPU& hornet_gpu){

    Timer<DEVICE> TM(3);
    TM.start();


	eoff_t *d_positions,*d_edgeCounter,*d_prefixPositions;
	gpu::allocate(d_positions,hornet_gpu.nV()+1);
	gpu::allocate(d_prefixPositions,hornet_gpu.nV()+1);
	gpu::allocate(d_edgeCounter,1);

	vid_t nE = hornet_gpu.nE();

    gpu::memsetZero(d_edgeCounter, 1);

	forAllVertices(hornet_gpu, getPosition { d_positions,d_edgeCounter });
	
	// gdf_column sourceColumn,destColumn;
	gdf_column* retCols = new gdf_column[2];


	allocateGdfColumn<vid_t>(retCols, (size_t)nE);
	allocateGdfColumn<vid_t>(retCols+1, (size_t)nE);

    load_balancing::BinarySearch 	load_balancing(hornet_gpu);

	forAllEdges(hornet_gpu, convertToCOO<vid_t> { d_positions, (vid_t*)retCols[0].data,(vid_t*)retCols[1].data},load_balancing);


	gpu::free(d_positions);
	gpu::free(d_prefixPositions);
	gpu::free(d_edgeCounter);

    TM.stop();
    TM.print("Conversion from Hornet to COO : ");

		// freeGdfColumn(&sourceColumn);
		// freeGdfColumn(&destColumn);
    return retCols;

}



gdf_column* HornetToCOOSorted(HornetGPU& hornet_gpu){

    Timer<DEVICE> TM(3);
    TM.start();


	vid_t *d_sizes,*d_prefixPositions;
	gpu::allocate(d_sizes,hornet_gpu.nV()+1);
	gpu::allocate(d_prefixPositions,hornet_gpu.nV()+1);

    gpu::memsetZero(d_sizes, hornet_gpu.nV()+1);

	forAllVertices(hornet_gpu, getPositionSorted { d_sizes });

	vid_t nE = hornet_gpu.nE();

	void *_d_temp_storage=nullptr; size_t _temp_storage_bytes=0;
	_d_temp_storage=nullptr; _temp_storage_bytes=0;
	cub::DeviceScan::ExclusiveSum(_d_temp_storage, _temp_storage_bytes,d_sizes, d_prefixPositions, hornet_gpu.nV()+1);
	cudaMalloc(&_d_temp_storage, _temp_storage_bytes);
	cub::DeviceScan::ExclusiveSum(_d_temp_storage, _temp_storage_bytes,d_sizes, d_prefixPositions, hornet_gpu.nV()+1);
	gpu::free(_d_temp_storage);

	gdf_column* retCols = new gdf_column[2];

	allocateGdfColumn<vid_t>(retCols, (size_t)nE);
	allocateGdfColumn<vid_t>(retCols+1, (size_t)nE);

    load_balancing::BinarySearch 	load_balancing(hornet_gpu);

	forAllEdges(hornet_gpu, convertToCOOSorted<vid_t> { d_prefixPositions, (vid_t*)retCols[0].data,(vid_t*)retCols[1].data},load_balancing);

	gpu::free(d_sizes);
	gpu::free(d_prefixPositions);

    TM.stop();
    TM.print("Conversion from Hornet to (semi)-Sorted  (by src) COO : ");

    return retCols;
}



gdf_column* HornetToCSR(HornetGPU& hornet_gpu){
    Timer<DEVICE> TM(3);
    TM.start();

	vid_t *d_sizes,*d_prefixPositions;
	gpu::allocate(d_sizes,hornet_gpu.nV()+1);
	gpu::allocate(d_prefixPositions,hornet_gpu.nV()+1);

    gpu::memsetZero(d_sizes, hornet_gpu.nV()+1);

	forAllVertices(hornet_gpu, getPositionSorted { d_sizes });

	vid_t nE = hornet_gpu.nE();

	void *_d_temp_storage=nullptr; size_t _temp_storage_bytes=0;
	_d_temp_storage=nullptr; _temp_storage_bytes=0;
	cub::DeviceScan::ExclusiveSum(_d_temp_storage, _temp_storage_bytes,d_sizes, d_prefixPositions, hornet_gpu.nV()+1);
	cudaMalloc(&_d_temp_storage, _temp_storage_bytes);
	cub::DeviceScan::ExclusiveSum(_d_temp_storage, _temp_storage_bytes,d_sizes, d_prefixPositions, hornet_gpu.nV()+1);
	gpu::free(_d_temp_storage);

	gdf_column* retCols = new gdf_column[2];

	// gdf_column offsetColumn,indColumn;

	allocateGdfColumn<vid_t>(retCols, (size_t)(hornet_gpu.nV()+1));
	allocateGdfColumn<vid_t>(retCols+1, (size_t)nE);
	gpu::copyToDevice(d_prefixPositions,hornet_gpu.nV()+1,(vid_t*)retCols[0].data);

    load_balancing::BinarySearch 	load_balancing(hornet_gpu);

	forAllEdges(hornet_gpu, convertToCSR<vid_t> { d_prefixPositions, (vid_t*)retCols[1].data},load_balancing);

	gpu::free(d_sizes);
	gpu::free(d_prefixPositions);

    TM.stop();
    TM.print("Conversion from Hornet to CSR : ");

    return retCols;

}









struct AlgoArgs{
  vid_t* arrayA;
  vid_t* arrayB;
  vid_t* arrayC;
  int32_t lengthA;
  int32_t lengthB;
};


__device__  __forceinline__ void serialMerge(AlgoArgs &arg) {
    int32_t Aindex = 0;
    int32_t Bindex = 0;
    int32_t Cindex = 0;

    while(Aindex < arg.lengthA && Bindex < arg.lengthB) {
        arg.arrayC[Cindex++] = arg.arrayA[Aindex] < arg.arrayB[Bindex] ? arg.arrayA[Aindex++] : arg.arrayB[Bindex++];
    }
    while(Aindex < arg.lengthA) arg.arrayC[Cindex++] = arg.arrayA[Aindex++];
    while(Bindex < arg.lengthB) arg.arrayC[Cindex++] = arg.arrayB[Bindex++];
}


__device__ void iterativeMergeSort(vid_t* array, int32_t length, vid_t* temp) {
    vid_t* c_sort = array;
    vid_t* temp_array = temp;
    int32_t array_length = length;

    // We can't return different pointers because
    // this will cause memory issues.
    // Therefore we might need to copy in the last step
    int numberOfSwaps = 0;

    //now do actual iterative merge sort
    for (int32_t currentSubArraySize = 1; currentSubArraySize < array_length; currentSubArraySize = 2 * currentSubArraySize)
    {
      for (int32_t A_start = 0; A_start < array_length; A_start += 2 * currentSubArraySize)
      {
          int32_t A_end = min(A_start + currentSubArraySize, array_length - 1);
          int32_t B_start = A_end;
          int32_t B_end = min(A_start + 2 * currentSubArraySize, array_length);
          int32_t A_length = A_end - A_start;
          int32_t B_length = B_end - B_start;

          struct AlgoArgs mergeArgs;
          mergeArgs.arrayA = c_sort + A_start;
          mergeArgs.lengthA = A_length;
          mergeArgs.arrayB = c_sort + B_start;
          mergeArgs.lengthB = B_length;
          mergeArgs.arrayC = temp_array + A_start;

          serialMerge(mergeArgs);
          // serialMerge(c_sort+A_start,A_length, c_sort+B_start, B_length, temp_array);
      }
      // Pointer swap for C
      vid_t* tmp = c_sort;
      c_sort = temp_array;
      temp_array = tmp;
      numberOfSwaps++;
    }

    if (numberOfSwaps%2 == 1) {
      for(int32_t i=0; i<length; i++){
        c_sort[i]=temp_array[i];
      } 
    }
}


struct launchMergeSort {
  vid_t      *offset;
  vid_t      *edges;
  vid_t      *edgesTemp;

  OPERATOR(int i) {

    int32_t adjSize=offset[i+1]-offset[i];

    if(adjSize<=1)
      return;
    iterativeMergeSort(edges+offset[i], adjSize,edgesTemp+offset[i]);

  }
};

struct binCount {
  vid_t      *offset;
  int32_t     *bins;
  vid_t      *edgesTemp;

  OPERATOR(int i) {
  	__shared__ int32_t localBins[33];
  	int id = threadIdx.x;
  	if(id<33){
  		localBins[id]=0;
  	}
  	__syncthreads();

    int32_t adjSize=offset[i+1]-offset[i];
    int myBin  = __clz(adjSize);

    atomicAdd(localBins+myBin, 1);

	__syncthreads();
  	if(id<33){
    	atomicAdd(bins+id, localBins[id]);
  	}


    // atomicAdd(bins+myBin, 1);

  }
};


struct binPrefix {
  int32_t     *bins;
  int32_t     *d_binsPrefix;

  OPERATOR(int i) {
  	if(i>=1){
  		printf("*");
  		return;
  	}
  		d_binsPrefix[0]=0;
  		for(int b=0; b<33; b++){
  			d_binsPrefix[b+1]=d_binsPrefix[b]+bins[b];
  			// printf("%d, %d\n",33-b,d_binsPrefix[b]);
  			// if(b==32)
	  		// 	printf("%d, %d\n",33,d_binsPrefix[33]);
  		}


  	}
};

struct rebin{
  vid_t     *offset;
  int32_t   *d_binsPrefix;
  vid_t  	*d_reOrg;
  vid_t     *newSize;
  vid_t     *d_start;
  vid_t     *d_stop;

  OPERATOR(int i) {

  	__shared__ int32_t localBins[33];
  	__shared__ int32_t localPos[33];
  	int id = threadIdx.x;
  	if(id<33){
  		localBins[id]=0;
  		localPos[id]=0;
  	}
  	__syncthreads();

    int32_t adjSize=offset[i+1]-offset[i];
    int myBin  = __clz(adjSize);

    atomicAdd(localBins+myBin, 1);

	__syncthreads();
  	if(id<33){
    	localPos[id]=atomicAdd(d_binsPrefix+id, localBins[id]);
  	}
	__syncthreads();


    int pos = atomicAdd(localPos+myBin, 1);
    d_reOrg[pos]=i;
    newSize[pos]=adjSize;
    d_start[pos]=offset[i];
    d_stop[pos] =offset[i+1];

    // if(i==6832){
    // 	printf("6832 : %d %d %d %d\n",pos,adjSize,offset[i],offset[i+1]);
    // }

  }
};


// #define BUBBLE(temp, val1,val2 ) temp=val1; val1=val2; val2=temp;
__device__ void bubbleSort(int32_t size, vid_t *edges){
  vid_t temp; 
  for(int32_t i=0; i<(size-1); i++){
	int32_t min_idx=i;
	for(int32_t j=i+1; j<(size); j++){
	  if(edges[j]<edges[min_idx])
		min_idx=j;
	}
	temp          = edges[min_idx];
	edges[min_idx]  = edges[i];
	edges[i]        = temp;
  }
}


struct sortSmall{
  vid_t  	*edges;
  vid_t  	*newEdges;
  vid_t  	*d_reOrg;
  // vid_t     *d_start;
  // vid_t     *d_stop;
  vid_t     *offset;
  int32_t pos;

  OPERATOR(int i) {

  	vid_t v=d_reOrg[i+pos];

    // int32_t adjSize=d_stop[i+pos]-d_start[i+pos];


    int32_t adjSize=offset[v+1]-offset[v];
  	if(adjSize==0){
  		return;
  	}
  	else if (adjSize==1){
    	newEdges[offset[v]]=edges[offset[v]]; 
  		return;
  	}


    // int32_t temp1[32],temp2[32];
 //    for(int32_t d=0; d<adjSize;d++){
 //    	temp1[d]=edges[offset[v]+d];
 //    }

	// iterativeMergeSort(temp1, adjSize,temp2);

 //    for(int32_t d=0; d<adjSize;d++){
 //    	newEdges[offset[v]+d]=temp1[d];
 //    }

    for(int32_t d=0; d<adjSize;d++){
    	newEdges[offset[v]+d]=edges[offset[v]+d];
	}

    bubbleSort(adjSize,newEdges+offset[v]);

 //    int32_t temp1[32];

 //    for(int32_t d=0; d<adjSize;d++){
 //    	temp1[d]=edges[offset[v]+d];
	// }

 //    bubbleSort(adjSize,temp1);

 //    for(int32_t d=0; d<adjSize;d++){
 //    	newEdges[offset[v]+d]=temp1[d];
	// }


  }
};

template <int threads, int elements_per_thread,int  total_elements>
__global__ void sortOneSize(
		int32_t	posReorg,
		vid_t  	*d_reOrg,
		vid_t   *offset,
		vid_t  	*edges,
		vid_t  	*newEdges,
		vid_t   *newSize,
		vid_t   *d_start,
		vid_t   *d_stop
	  )
{
	typedef cub::BlockRadixSort<vid_t, threads, elements_per_thread> BlockRadixSort;
	
	__shared__ typename BlockRadixSort::TempStorage temp_storage;
	__shared__ vid_t sharedEdges[total_elements];

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	vid_t v=d_reOrg[posReorg+bid];

    int32_t adjSize=offset[v+1]-offset[v];

  //   if(tid==0 && bid<=4)
		// printf("%d %d %d %d %d %d\n",posReorg+bid,  v,adjSize,newSize[posReorg+bid],d_stop[posReorg+bid],d_start[posReorg+bid]);		
	for(int i=tid; i<total_elements; i+=threads){
		if(i<adjSize)
			sharedEdges[i]=edges[offset[v]+i];
		else
      // sharedEdges[i]=INT_MAX-1;
      sharedEdges[i]=cub::Traits<vid_t>::MAX_KEY;

	}

	// if(bid==0 && tid==0){
	// 	for(int i=0;i<adjSize; i++)
	// 		printf("%d,", sharedEdges[i]);
	// 	printf("@@\n");
	// }

	__syncthreads();

	vid_t mine[elements_per_thread];
	// for(int i=tid; i<total_elements; i+=threads)
	// 	mine[s++]=sharedEdges[i];

	int pos=tid*elements_per_thread;
	for(int s=0; s<elements_per_thread; s++){
		mine[s]=sharedEdges[pos+s];
	}


	__syncthreads();

	BlockRadixSort(temp_storage).Sort(mine);
	__syncthreads();


 	pos=tid*elements_per_thread;
	for(int s=0; s<elements_per_thread; s++){
		sharedEdges[pos+s]=mine[s];
	}
	__syncthreads();

	// if(bid==0 && tid==0){
	// 	for(int i=0;i<adjSize; i++)
	// 		printf("%d,", sharedEdges[i]);
	// 	printf("@@\n");
	// }


	for(int i=tid; i<adjSize; i+=threads){
		newEdges[offset[v]+i]=sharedEdges[i];
	}

}

struct compareEdges{
  vid_t     *firstSet;
  vid_t     *secondSet;
  vid_t 	*countIdentical;

  OPERATOR(int i) {
  	if(firstSet[i]==secondSet[i]){
  		atomicAdd(countIdentical,1);
  		// printf("*");
  	}
  }
};

// struct reorder{
//   vid_t     *offset;
//   vid_t     *edges;
//   vid_t  	*d_reOrg;
//   vid_t     *newOffset;
//   vid_t     *newEdges;

//   OPERATOR(int i) {

//   	v = d_reOrg[i];

//     int32_t adjSize=offset[v+1]-offset[v];

//     for (d=0; d<adjSize;d++){
//     	newEdges[d_newOffset[i]+d]=edges[offset[v]+d];
//     }

//   }
// };







// struct launchReorgMergeSort {
//   vid_t      *offset;
//   vid_t      *edges;
//   vid_t      *edgesTemp;
//   vid_t   	 *d_reOrg;

//   OPERATOR(int i) {
//   	vid_t v = d_reOrg[i];
//     int32_t adjSize=offset[v+1]-offset[v];

//     if(adjSize<=1)
//       return;

//   	if(adjSize>=(1<<8))
//   		return;


//   	if(adjSize<32)
//   		return;
//   		// bubbleSort(adjSize,edges+offset[v]);
//   	else
// 	    iterativeMergeSort(edges+offset[v], adjSize,edgesTemp+offset[v]);

//   }
// };


struct print{
  vid_t     *newOffset;

  OPERATOR(int i) {

  		for(int b=0; b<10; b++){
  			printf("%d, ",newOffset[b]);
  		}
  }
};

struct printOneBefore{
  vid_t     *offset;
  vid_t     *edges;

  OPERATOR(int i) {

  		int32_t adjSize=offset[1]-offset[0];

  		if(adjSize<2)
  			return;
  		for(int e=0; e<adjSize; e++){
  			printf("%d, ",edges[e]);
  		}
  		vid_t temp;
  		temp = edges[0];
  		edges[0] = edges[1];
  		edges[1] = temp;
  		printf("\n");
  		for(int e=0; e<adjSize; e++){
  			printf("%d, ",edges[e]);
  		}
  		printf("\n");

  }
};


struct printOneAfter{
  vid_t     *offset;
  vid_t     *edges;

  OPERATOR(int i) {

  		int32_t adjSize=offset[1]-offset[0];

  		if(adjSize<2)
  			return;
  		for(int e=0; e<adjSize; e++){
  			printf("%d, ",edges[e]);
  		}
  		printf("\n");
  }
};



void exec(int argc, char* argv[]) {
    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;
    xlib::device_info();

    graph::GraphStd<vid_t, eoff_t> graph;
    graph.read(argv[1]);

    standard_context_t context;

    // auto weights = new int[graph.nE()];
    // std::iota(weights, weights + graph.nE(), 0);
    //--------------------------------------------------------------------------
    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges());

    HornetGPU hornet_gpu(hornet_init);

    {
		gdf_column* cols = HornetToCOO(hornet_gpu);

		freeGdfColumn(cols);
		freeGdfColumn(cols+1);
		delete [] cols;
    }


    {
		gdf_column* cols = HornetToCOOSorted(hornet_gpu);

		freeGdfColumn(cols);
		freeGdfColumn(cols+1);
		delete [] cols;
    }

    {
		gdf_column* cols = HornetToCSR(hornet_gpu);

		freeGdfColumn(cols);
		freeGdfColumn(cols+1);
		delete [] cols;
    }



    {
		gdf_column* cols = HornetToCOO(hornet_gpu);

		vid_t nE = hornet_gpu.nE();
		gdf_column 	sortedKeys, sortedVals; 
		allocateGdfColumn<vid_t>(&sortedKeys, (size_t)nE);
		allocateGdfColumn<vid_t>(&sortedVals, (size_t)nE);

	    Timer<DEVICE> TM(3);
	    TM.start();


		void     *d_temp_storage = NULL;
		size_t   temp_storage_bytes = 0;
		cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, (vid_t*)cols[0].data, (vid_t*)sortedKeys.data, (vid_t*)cols[1].data, (vid_t*)sortedVals.data, nE);
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
		// cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, (vid_t*)cols[0].data, (vid_t*)sortedKeys.data, (vid_t*)cols[1].data, (vid_t*)sortedVals.data, nE);


		// Two phase sorting. First sort based on the destination. Then sort based on the  source.
		// The final output is in the gdf columns
		cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, (vid_t*)cols[1].data, 	  (vid_t*)sortedVals.data, (vid_t*)cols[0].data, (vid_t*)sortedKeys.data, nE);
		cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, (vid_t*)sortedKeys.data,  (vid_t*)cols[0].data, (vid_t*)sortedVals.data, (vid_t*)cols[1].data, nE);


		// Run sorting operation
	    cudaFree(d_temp_storage);

	    TM.stop();
	    TM.print("Time to sort COO: two phases- destination then source");


		freeGdfColumn(cols);
		freeGdfColumn(cols+1);
		freeGdfColumn(&sortedKeys);
		freeGdfColumn(&sortedVals);
		delete [] cols;
    }

    {
		gdf_column* cols = HornetToCSR(hornet_gpu);

		vid_t nV = hornet_gpu.nV();
		vid_t nE = hornet_gpu.nE();
		gdf_column 	extraStorage; 
		allocateGdfColumn<vid_t>(&extraStorage, (size_t)nE);

	    Timer<DEVICE> TM(3);
	    // TM.start();

		   //  forAll (nV,launchMergeSort{(vid_t*)cols[0].data, (vid_t*)cols[1].data, (vid_t*)extraStorage.data});

	    // TM.stop();
	    // TM.print("Time to sort CSR using merge-sort per adjacency");


	    int32_t *d_bins, *d_binsPrefix,*d_binsPrefixTemp;
	    vid_t *d_reOrg,*d_newOffset,*d_newSize,*modernGPUref;
		gpu::allocate((int32_t*&)(d_bins),33);
		gpu::allocate((int32_t*&)(d_binsPrefix),34);
		gpu::allocate((int32_t*&)(d_binsPrefixTemp),34);
		gpu::allocate((vid_t*&)(d_reOrg),nV);
		gpu::allocate((vid_t*&)(d_newSize),nV+2);
		gpu::allocate((vid_t*&)(d_newOffset),nV+2);

		gpu::allocate((vid_t*&)(modernGPUref),nE+1);

		// gpu::allocate((vid_t*&)(d_newEdges),nE);


		cudaMemcpy(modernGPUref,(vid_t*)cols[1].data,sizeof(vid_t)*nE,cudaMemcpyDeviceToDevice);

    	// forAll (1,printOneBefore{(vid_t*)cols[0].data,(vid_t*)cols[1].data});

	    TM.start();

		segmented_sort((vid_t*)modernGPUref, nE, (vid_t*)cols[0].data, nV, less_equal_t<vid_t>(), context);
		// segmented_sort((vid_t*)cols[1].data, nE, (vid_t*)cols[0].data, nV, less_equal_t<vid_t>(), context);


		    // forAll (nV,launchReorgMergeSort{(vid_t*)cols[0].data, (vid_t*)cols[1].data, (vid_t*)extraStorage.data,d_reOrg});


	    TM.stop();
	    TM.print("Time to sort CSR using moderngpu SegmentSort");

    	// forAll (1,printOneAfter{(vid_t*)cols[0].data,(vid_t*)cols[1].data});

    	forAll (1,printOneBefore{(vid_t*)cols[0].data,(vid_t*)cols[1].data});

	    TM.start();

		void     *d_temp_storage = NULL;
		size_t   temp_storage_bytes = 0;


		cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, (vid_t*)cols[1].data, (vid_t*)extraStorage.data,
    	nE, nV, (vid_t*)cols[0].data, (vid_t*)(cols[0].data) + 1);

		cudaMalloc(&d_temp_storage, temp_storage_bytes);
		cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, (vid_t*)cols[1].data, (vid_t*)extraStorage.data,
    	nE, nV, (vid_t*)cols[0].data, (vid_t*)(cols[0].data) + 1);
	    cudaFree(d_temp_storage);

	    TM.stop();
	    TM.print("Time to sort CSR using Cub's DeviceSegmentedRadixSort");

    	forAll (1,printOneAfter{(vid_t*)cols[0].data,(vid_t*)extraStorage.data});


	    vid_t *d_start,*d_stop;
		gpu::allocate((vid_t*&)(d_start),nV);
		gpu::allocate((vid_t*&)(d_stop),nV);

		int sortRadix=20;
		int32_t pos=0;
		int sortRadixSmall=28;		
		int32_t posSmall=0;
		// int32_t posZero=0;
		int32_t h_binsPrefix[34];

	    TM.start();
		    gpu::memsetZero(d_bins, 33);
		    gpu::memsetZero(d_newSize, nV+2);
		    forAll (nV,binCount{(vid_t*)cols[0].data, d_bins});
		    forAll (1,binPrefix{d_bins,d_binsPrefix});

		    cudaMemcpy(d_binsPrefixTemp,d_binsPrefix,sizeof(int32_t)*34, cudaMemcpyDeviceToDevice);

		    forAll (nV,rebin{(vid_t*)cols[0].data,d_binsPrefixTemp,d_reOrg,d_newSize,d_start,d_stop});

		    cudaMemcpy(&pos,d_binsPrefix+sortRadix,sizeof(int32_t), cudaMemcpyDeviceToHost);
		    cudaMemcpy(&posSmall,d_binsPrefix+sortRadixSmall,sizeof(int32_t), cudaMemcpyDeviceToHost);


      // void     *d_temp_storage = NULL;
      // size_t   temp_storage_bytes = 0;

      d_temp_storage = NULL;
      temp_storage_bytes = 0;

			cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, (vid_t*)cols[1].data, (vid_t*)extraStorage.data,
	    	nE, pos, d_start, d_stop);
			cudaMalloc(&d_temp_storage, temp_storage_bytes);
			cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, (vid_t*)cols[1].data, (vid_t*)extraStorage.data,
	    	nE, pos, d_start, d_stop);
		    cudaFree(d_temp_storage);



	    TM.stop();
	    TM.print("Time to sort with LRB");

	    TM.start();

		    cudaMemcpy(h_binsPrefix,d_binsPrefix,sizeof(int32_t)*34, cudaMemcpyDeviceToHost);

		    // for(int e=1; e<34; e++)
		    // 	printf("%d, %d\n",33-e, h_binsPrefix[e]- h_binsPrefix[e-1]);

		    if(h_binsPrefix[21]-h_binsPrefix[20]>0)
			    sortOneSize<256,16,4096>	<<<h_binsPrefix[21]-h_binsPrefix[20],256>>>		(h_binsPrefix[20],d_reOrg,(vid_t*)cols[0].data,(vid_t*)cols[1].data, (vid_t*)extraStorage.data,d_newSize,d_start,d_stop);
		    if(h_binsPrefix[22]-h_binsPrefix[21]>0)
			    sortOneSize<128,16,2048>	<<<h_binsPrefix[22]-h_binsPrefix[21],128>>>		(h_binsPrefix[21],d_reOrg,(vid_t*)cols[0].data,(vid_t*)cols[1].data, (vid_t*)extraStorage.data,d_newSize,d_start,d_stop);
		    if(h_binsPrefix[23]-h_binsPrefix[22]>0)
			    sortOneSize<128,8,1024>		<<<h_binsPrefix[23]-h_binsPrefix[22],128>>>		(h_binsPrefix[22],d_reOrg,(vid_t*)cols[0].data,(vid_t*)cols[1].data, (vid_t*)extraStorage.data,d_newSize,d_start,d_stop);
		    if(h_binsPrefix[24]-h_binsPrefix[23]>0)
			    sortOneSize<128,4,512>		<<<h_binsPrefix[24]-h_binsPrefix[23],128>>>		(h_binsPrefix[23],d_reOrg,(vid_t*)cols[0].data,(vid_t*)cols[1].data, (vid_t*)extraStorage.data,d_newSize,d_start,d_stop);
		    if(h_binsPrefix[25]-h_binsPrefix[24]>0)
			    sortOneSize<64,4,256>		<<<h_binsPrefix[25]-h_binsPrefix[24],64>>>		(h_binsPrefix[24],d_reOrg,(vid_t*)cols[0].data,(vid_t*)cols[1].data, (vid_t*)extraStorage.data,d_newSize,d_start,d_stop);
		    if(h_binsPrefix[26]-h_binsPrefix[25]>0)
			    sortOneSize<64,2,128>		<<<h_binsPrefix[26]-h_binsPrefix[25],64>>>		(h_binsPrefix[25],d_reOrg,(vid_t*)cols[0].data,(vid_t*)cols[1].data, (vid_t*)extraStorage.data,d_newSize,d_start,d_stop);
		    if(h_binsPrefix[27]-h_binsPrefix[26]>0)
			    sortOneSize<64,1,64>		<<<h_binsPrefix[27]-h_binsPrefix[26],64>>>		(h_binsPrefix[26],d_reOrg,(vid_t*)cols[0].data,(vid_t*)cols[1].data, (vid_t*)extraStorage.data,d_newSize,d_start,d_stop);
		    if(h_binsPrefix[28]-h_binsPrefix[27]>0)
			    sortOneSize<32,1,32>		<<<h_binsPrefix[28]-h_binsPrefix[27],32>>>		(h_binsPrefix[27],d_reOrg,(vid_t*)cols[0].data,(vid_t*)cols[1].data, (vid_t*)extraStorage.data,d_newSize,d_start,d_stop);


			if(nV-posSmall>0)
			    forAll (nV-posSmall,sortSmall{(vid_t*)cols[1].data,(vid_t*)extraStorage.data,d_reOrg,(vid_t*)cols[0].data,posSmall});

///////		    // sortOneSize<32,1,16>	<<<h_binsPrefix[28]-h_binsPrefix[27],32>>>(h_binsPrefix[27],d_reOrg,(vid_t*)cols[0].data,(vid_t*)cols[1].data, (vid_t*)extraStorage.data,d_newSize);



	    TM.stop();
	    TM.print("CUB Device Radix");

	    // int x;
	    // scanf("%d",&x);

	    fflush(stdout);

	    printf("The number of edges is %d %d\n", nE, h_binsPrefix[33]);
	    vid_t* countIdentical;
	    cudaMallocManaged((void**)&countIdentical,sizeof(int));
	    *countIdentical=0;
		// forAll (nE,compareEdges{(vid_t*)extraStorage.data,modernGPUref,countIdentical});
		// forAll (nE,compareEdges{modernGPUref,modernGPUref,countIdentical});
	    fflush(stdout);

		printf("Number of identical is %d\n",*countIdentical);

	    fflush(stdout);

	    printf("The number of adjacency lists sorted is %d\n",h_binsPrefix[33]);
	    // printf("The number of adjacency lists sorted is %d\n",pos+ (nV-posSmall));

			void *_d_temp_storage=nullptr; size_t _temp_storage_bytes=0;
			cub::DeviceScan::ExclusiveSum(_d_temp_storage, _temp_storage_bytes,d_newSize, d_newOffset, nV+1);
			cudaMalloc(&_d_temp_storage, _temp_storage_bytes);
			cub::DeviceScan::ExclusiveSum(_d_temp_storage, _temp_storage_bytes,d_newSize, d_newOffset, nV+1);
			gpu::free(_d_temp_storage);

		int32_t sortedEdges=0;
		cudaMemcpy(&sortedEdges,d_newOffset+h_binsPrefix[33],sizeof(int32_t), cudaMemcpyDeviceToHost);
		// cudaMemcpy(&sortedEdgesSmall,d_newOffset+posSmall,sizeof(int32_t), cudaMemcpyDeviceToHost);
		// sortedEdgesSmall = nE-sortedEdgesSmall;
	    printf("The number of sorted edges is %d\n",sortedEdges);


	    gpu::free(modernGPUref);
	    gpu::free(d_start);
	    gpu::free(d_stop);	    
	    gpu::free(d_newOffset);
	    gpu::free(d_newSize);
		gpu::free(d_bins);
		gpu::free(d_binsPrefix);
		gpu::free(d_binsPrefixTemp);
		gpu::free(d_reOrg);
		freeGdfColumn(cols);
		freeGdfColumn(cols+1);
		freeGdfColumn(&extraStorage);

		delete [] cols;
    }


	// {
	// 	eoff_t *h_emptyoffsets;
	// 	host::allocate(h_emptyoffsets,hornet_gpu.nV()+1);
	//     host::memsetZero(h_emptyoffsets, hornet_gpu.nV()+1);

	//     HornetInit hornet_init3(graph.nV(), 0, h_emptyoffsets,NULL);

	//     HornetGPU hornet_gpu3(hornet_init3);

	//     gpu::BatchUpdate batch_update((vid_t*)sourceColumn.data, (vid_t*)destColumn.data, nE);

	//     hornet_gpu3.reserveBatchOpResource(nE);

	// 	std::cout << "Number of edges in new Hornet " << hornet_gpu3.nE() << std::endl;

	//     TM.start();
	//     hornet_gpu3.insertEdgeBatch(batch_update);

	//     TM.stop();
		
	// 	TM.print("Batch insertion time:");


	// 	std::cout << "Number of edges in new Hornet " << hornet_gpu3.nE() << std::endl;
	//     host::free(h_emptyoffsets);
	// 	freeGdfColumn(&sourceColumn);
	// 	freeGdfColumn(&destColumn);
 // 	}







}




/*
This code can be used to check correctness later on

	    vid_t* edges; 
	    eoff_t* offsets;

		size_t s_NV = hornet_gpu.nV(), s_NE=hornet_gpu.nE();
		host::allocate(offsets,s_NV+1);
		host::allocate(edges,s_NE);


		gpu::copyToHost((eoff_t*)offsetColumn.data,s_NV+1,offsets);
		gpu::copyToHost((vid_t*)indColumn.data,s_NE,edges);

	    HornetInit hornet_init2(s_NV, s_NE, offsets,edges);
	    HornetGPU hornet_gpu2(hornet_init2);

		host::free(offsets);
		host::free(edges);

		freeGdfColumn(&offsetColumn);
		freeGdfColumn(&indColumn);

*/
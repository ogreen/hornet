
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


#include <unordered_map>
#include <algorithm>

#include "main.hpp"


using namespace std;


void BatchUpdateData::copyHostToHost(BatchUpdateData &hBUA){
	if (isHost && hBUA.isHost){
		copyArrayHostToHost(hBUA.mem, mem, numberBytes, sizeof(int8_t));
	}
	else{
		CUSTINGER_ERROR(string("Failure to copy host array to host array in ") + string(typeid(*this).name())+ string(". One of the ARRAY is not a host array"));
	}
}

void BatchUpdateData::copyHostToDevice(BatchUpdateData &hBUA){
	if (!isHost && hBUA.isHost){
		copyArrayHostToDevice(hBUA.mem, mem, numberBytes, sizeof(int8_t));
	}
	else{
		CUSTINGER_ERROR(string("Failure to copy host array to host array in ") + string(typeid(*this).name())+ string(". One of the ARRAY is not a host array"));
	}
}
void BatchUpdateData::copyDeviceToHost(BatchUpdateData &dBUA){
	if (isHost && !dBUA.isHost){	
		copyArrayDeviceToHost(dBUA.mem, mem, numberBytes, sizeof(int8_t));
	}
	else{
		CUSTINGER_ERROR(string("Failure to copy host array to host array in ") + string(typeid(*this).name())+ string(". One of the ARRAY is not a host array"));
	}
}

void BatchUpdateData::copyDeviceToHostDupCount(BatchUpdateData &dBUD){
	if (isHost && !dBUD.isHost){
		copyArrayDeviceToHost(dBUD.dupCount,dupCount,1,sizeof(length_t));
	}
	else{
		CUSTINGER_ERROR(string("Failure to copy device array to host array in ") + string(typeid(*this).name())+ string(". One array is not the right type"));
	}
}

void BatchUpdateData::copyDeviceToHostIncCount(BatchUpdateData &dBUD)
{
	if (isHost && !dBUD.isHost){
		copyArrayDeviceToHost(dBUD.incCount,incCount,1,sizeof(length_t));
	}
	else{
		CUSTINGER_ERROR(string("Failure to copy device array to host array in ") + string(typeid(*this).name())+ string(". One array is not the right type"));
	}
}





BatchUpdate::BatchUpdate(BatchUpdateData &h_bua){

	cout << "Made it this far" << endl << flush;
	if(!h_bua.getisHost()){
		CUSTINGER_ERROR(string(typeid(*this).name()) + string(" expects to receive an update list that is host size"));
	}

	length_t batchSize = *(h_bua.getBatchSize());
	cout << "The batch size is :" << batchSize << endl;


	hData = new BatchUpdateData(batchSize,true);
	cout << "The batch size is :" << batchSize << endl;
	dData = new BatchUpdateData(batchSize,false);
	cout << "Made it this far" << endl << flush;

	hData->copyHostToHost(h_bua);
	dData->copyHostToDevice(h_bua);

	dPtr=(BatchUpdate*) allocDeviceArray(1,sizeof(BatchUpdate));
	copyArrayHostToDevice(this,dPtr,1, sizeof(BatchUpdate));
}

// BatchUpdate::BatchUpdate(int32_t batchSize_){
// 	h_edgeSrc       =  (int32_t*)allocHostArray(batchSize_,sizeof(int32_t));
// 	h_edgeDst       =  (int32_t*)allocHostArray(batchSize_,sizeof(int32_t));
// 	h_indIncomplete =  (int32_t*)allocHostArray(batchSize_,sizeof(int32_t));
// 	h_incCount      =  (int32_t*)allocHostArray(1,sizeof(int32_t));
// 	h_batchSize     =  (int32_t*)allocHostArray(1,sizeof(int32_t));
// 	h_indDuplicate  =  (int32_t*)allocHostArray(batchSize_,sizeof(int32_t));
// 	h_dupCount      =  (int32_t*)allocHostArray(1,sizeof(int32_t));
// 	h_dupRelPos     =  (int32_t*)allocHostArray(batchSize_,sizeof(int32_t));

// 	d_edgeSrc       =  (int32_t*)allocDeviceArray(batchSize_,sizeof(int32_t));
// 	d_edgeDst       =  (int32_t*)allocDeviceArray(batchSize_,sizeof(int32_t));
// 	d_indIncomplete =  (int32_t*)allocDeviceArray(batchSize_,sizeof(int32_t));
// 	d_incCount      =  (int32_t*)allocDeviceArray(1,sizeof(int32_t));
// 	d_batchSize     =  (int32_t*)allocDeviceArray(1,sizeof(int32_t));
// 	d_indDuplicate  =  (int32_t*)allocDeviceArray(batchSize_,sizeof(int32_t));
// 	d_dupCount      =  (int32_t*)allocDeviceArray(1,sizeof(int32_t));
// 	d_dupRelPos     =  (int32_t*)allocDeviceArray(batchSize_,sizeof(int32_t));

// 	h_batchSize[0]=batchSize_;

// 	resetHostIncCount();
// 	resetHostDuplicateCount();


// 	d_batchUpdate=(BatchUpdate*) allocDeviceArray(1,sizeof(BatchUpdate));
// 	copyArrayHostToDevice(this,d_batchUpdate,1, sizeof(BatchUpdate));
// }


BatchUpdate::~BatchUpdate(){
	// freeHostArray(h_edgeSrc);
	// freeHostArray(h_edgeDst);
	// freeHostArray(h_incCount);
	// freeHostArray(h_batchSize);
	// freeHostArray(h_indDuplicate);
	// freeHostArray(h_dupCount);
	// freeHostArray(h_dupRelPos);

	// freeDeviceArray(d_edgeSrc);
	// freeDeviceArray(d_edgeDst);
	// freeDeviceArray(d_incCount);
	// freeDeviceArray(d_batchSize);
	// freeDeviceArray(d_indDuplicate);
	// freeDeviceArray(d_dupCount);
	// freeDeviceArray(d_dupRelPos);

	freeDeviceArray(dPtr);

	delete hData;
	delete dData;
}

// void BatchUpdate::copyHostToDevice(){
// 	copyArrayHostToDevice(h_batchSize, d_batchSize, 1, sizeof(int32_t));

// 	copyArrayHostToDevice(h_edgeSrc, d_edgeSrc, h_batchSize[0], sizeof(int32_t));
// 	copyArrayHostToDevice(h_edgeDst, d_edgeDst, h_batchSize[0], sizeof(int32_t));
// 	copyArrayHostToDevice(h_indIncomplete, d_indIncomplete, h_batchSize[0], sizeof(int32_t));
// 	copyArrayHostToDevice(h_incCount, d_incCount, 1, sizeof(int32_t));
// 	copyArrayHostToDevice(h_indDuplicate, d_indDuplicate, h_batchSize[0], sizeof(int32_t));
// 	copyArrayHostToDevice(h_dupRelPos, d_dupRelPos, h_batchSize[0], sizeof(int32_t));
// 	copyArrayHostToDevice(h_dupCount, d_dupCount, 1, sizeof(int32_t));
// }

// void BatchUpdate::copyDeviceToHost(){
// 	copyArrayDeviceToHost(d_edgeSrc, h_edgeSrc, h_batchSize[0], sizeof(int32_t));
// 	copyArrayDeviceToHost(d_edgeDst, h_edgeDst, h_batchSize[0], sizeof(int32_t));
// 	copyArrayDeviceToHost(d_indIncomplete, h_indIncomplete, h_batchSize[0], sizeof(int32_t));
// 	copyArrayDeviceToHost(d_incCount, h_incCount, 1, sizeof(int32_t));
// 	copyArrayDeviceToHost(d_indDuplicate, h_indDuplicate, h_batchSize[0], sizeof(int32_t));
// 	copyArrayDeviceToHost(d_dupRelPos, h_dupRelPos, h_batchSize[0], sizeof(int32_t));
// 	copyArrayDeviceToHost(d_dupCount, h_dupCount, 1, sizeof(int32_t));
// }





void BatchUpdate::reAllocateMemoryAfterSweep1(cuStinger &custing)
{
	cout << "Oded, still need to fix reallocation function" << endl;	


	// int32_t sum=0, *tempsrc=getHostSrc(),*tempdst=getHostDst();
	// int32_t *incomplete = getHostIndIncomplete();	
	// int32_t incCount = getHostIncCount();

	// unordered_map <int32_t, int32_t> h_hmap;

	// int32_t* h_requireUpdates=(int32_t*)allocHostArray(getHostBatchSize(), sizeof(int32_t));
	// int32_t* h_overLimit=(int32_t*)allocHostArray(getHostBatchSize(), sizeof(int32_t));

	// for (int32_t i=0; i<incCount; i++){
	// 	int32_t temp = tempsrc[incomplete[i]];
	// 	h_hmap[temp]++;
	// }

	// int countUnique=0;
	// for (int32_t i=0; i<incCount; i++){
	// 	int32_t temp = tempsrc[incomplete[i]];
	// 	if(h_hmap[temp]!=0){
	// 		h_requireUpdates[countUnique]=temp;
	// 		h_overLimit[countUnique]=h_hmap[temp];
	// 		countUnique++;
	// 		h_hmap[temp]=0;
	// 	}
	// }

	// custing.copyDeviceToHost();

	// if(countUnique>0){
	// 	int32_t ** h_tempAdjacency = (int32_t**) allocHostArray(custing.nv,sizeof(int32_t*));
	// 	int32_t ** d_tempAdjacency = (int32_t**) allocDeviceArray(custing.nv,sizeof(int32_t*));
	// 	int32_t * d_requireUpdates = (int32_t*) allocDeviceArray(countUnique, sizeof(int32_t));

	// 	for (int32_t i=0; i<countUnique; i++){
	// 		int32_t tempVertex = h_requireUpdates[i];
	// 		int32_t newMax = custing.updateVertexAllocator(custing.h_max[tempVertex] ,h_overLimit[i]);
	// 		h_tempAdjacency[tempVertex] = (int32_t*)allocDeviceArray(newMax, sizeof(int32_t));
	// 		custing.h_max[tempVertex] = newMax;
	// 	}

	// 	sort(h_requireUpdates, h_requireUpdates + countUnique);
	// 	copyArrayHostToDevice(h_requireUpdates,d_requireUpdates,countUnique,sizeof(int32_t));
	// 	copyArrayHostToDevice(h_tempAdjacency,d_tempAdjacency, custing.nv, sizeof(int32_t*));

	// 	custing.copyMultipleAdjacencies(d_tempAdjacency,d_requireUpdates,countUnique);

	// 	for (int32_t i=0; i<countUnique; i++){
	// 		int32_t tempVertex = h_requireUpdates[i];
	// 		freeDeviceArray(custing.h_adj[tempVertex]);
	// 		custing.h_adj[tempVertex] = h_tempAdjacency[tempVertex];
	// 	}

	// 	custing.copyHostToDevice();

	// 	freeDeviceArray(d_requireUpdates);
	// 	freeHostArray(h_tempAdjacency);
	// 	freeDeviceArray(d_tempAdjacency);
	// }

	// freeHostArray(h_requireUpdates);
	// freeHostArray(h_overLimit);
}

	
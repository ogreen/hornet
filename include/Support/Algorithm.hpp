/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
 *
 * @copyright Copyright © 2017 cuStinger. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 *
 * @file
 */
#pragma once

#include "Support/HostDevice.hpp"
#include <unordered_map>    //std::unordered_map

namespace xlib {

template<class FUN_T, typename... T>
inline void Funtion_TO_multiThreads(bool MultiCore, FUN_T FUN, T... Args);

/**
 * @brief return the old value if exits
 */
template<typename T, typename R = T>
class UniqueMap : public std::unordered_map<T, R> {
static_assert(std::is_integral<R>::value,
              "UniqueMap accept only Integral types");
public:
    R insertValue(T id);
};

template<bool FAULT = true, class iteratorA_t, class iteratorB_t>
bool equal(iteratorA_t start_A, iteratorA_t end_A, iteratorB_t start_B);

template<bool FAULT = true, class iteratorA_t, class iteratorB_t>
bool equal(iteratorA_t start_A, iteratorA_t end_A, iteratorB_t start_B,
        bool (*equalFunction)(
                typename std::iterator_traits<iteratorA_t>::value_type,
                typename std::iterator_traits<iteratorB_t>::value_type));

template<bool FAULT = true, class iteratorA_t, class iteratorB_t>
bool equalSorted(iteratorA_t start_A, iteratorA_t end_A, iteratorB_t start_B);

/**
 * required auxilary space: O(|end -start| * 2)
 */
template<typename T, typename... RArgs>
void sort_by_key(T* start, T* end, RArgs... data_packed);

//==============================================================================

template<typename T, typename R>
HOST_DEVICE
R lower_bound_left(const T* mem, R size, T searched);

template<typename T, typename R>
HOST_DEVICE
R lower_bound_right(const T* mem, R size, T searched);

template<typename T, typename R>
HOST_DEVICE
R upper_bound_left(const T* mem, R size, T searched);

template<typename T, typename R>
HOST_DEVICE
R upper_bound_right(const T* mem, R size, T searched);

template<typename T, typename R>
HOST_DEVICE
R binary_search(const T* mem, R size, T searched);

//------------------------------------------------------------------------------

template<typename T, typename S>
HOST_DEVICE
void merge(const T* left, S size_left, const T* right, S size_right, T* merge);

template<typename T, typename S>
HOST_DEVICE
void inplace_merge(T* left, S size_left, const T* right, S size_right);

} // namespace xlib

#include "impl/Algorithm.i.hpp"
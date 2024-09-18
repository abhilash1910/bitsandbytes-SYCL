// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_extras/dpcpp_extensions.h>
#include "kernel_quant.h"
#include <dpct/dpl_utils.hpp>
//#include <mma.h>
#include <cmath>

#define FLT_MAX std::numeric_limits<float>::max()
#define FLT_MIN std::numeric_limits<float>::min()


#define HLF_MAX 65504
#define TH 1024
#define NUM 4
#define NUM_BLOCK 4096

/// Load linear segment items into block format across threads
/// Helper for Block Load
namespace dpct{
namespace group{
/*
enum load_algorithm {

  BLOCK_LOAD_DIRECT,
  BLOCK_LOAD_STRIPED,
  // To-do: BLOCK_LOAD_WARP_TRANSPOSE

};

// loads a linear segment of workgroup items into a blocked arrangement.
template <size_t ITEMS_PER_WORK_ITEM, typename InputT,
          typename InputIteratorT, typename Item>
 __dpct_inline__ void load_blocked(const Item &item, InputIteratorT block_itr,
                                  InputT (&items)[ITEMS_PER_WORK_ITEM]) {

  // This implementation does not take in account range loading across
  // workgroup items To-do: Decide whether range loading is required for group
  // loading
  size_t linear_tid = item.get_local_linear_id();
  int ltid = int(linear_tid);
  uint32_t workgroup_offset = linear_tid * ITEMS_PER_WORK_ITEM;
  //static const CONSTANT char FMT[] = "n: %u\n";
  //sycl::ext::oneapi::experimental::printf(FMT,ltid);
#pragma unroll
  for (size_t idx = 0; idx < ITEMS_PER_WORK_ITEM; idx++) {
    items[idx] = block_itr[workgroup_offset + idx];
  }
}

// loads a linear segment of workgroup items into a striped arrangement.
template <size_t ITEMS_PER_WORK_ITEM, typename InputT,
          typename InputIteratorT, typename Item>
 __dpct_inline__ void load_striped(const Item &item, InputIteratorT block_itr,
                                  InputT (&items)[ITEMS_PER_WORK_ITEM]) {

  // This implementation does not take in account range loading across
  // workgroup items To-do: Decide whether range loading is required for group
  // loading
  size_t linear_tid = item.get_local_linear_id();
  size_t group_work_items = item.get_local_range().size();
  //static const CONSTANT char FMT[] = "n: %u\n";
  //sycl::ext::oneapi::experimental::printf(FMT,linear_tid);
  //sycl::ext::oneapi::experimental::printf("y: %u\n",group_work_items);
  //sycl::ext::oneapi::experimental::printf("items_per_wi: %u\n",ITEMS_PER_WORK_ITEM);
#pragma unroll
  
  
  for (size_t idx = 0; idx < ITEMS_PER_WORK_ITEM; idx++) {
    items[idx] = block_itr[linear_tid + (idx * group_work_items)];
  }
}

// loads a linear segment of workgroup items into a subgroup striped
// arrangement. Created as free function until exchange mechanism is
// implemented.
// To-do: inline this function with BLOCK_LOAD_WARP_TRANSPOSE mechanism
template <size_t ITEMS_PER_WORK_ITEM, typename InputT, typename InputIteratorT,
          typename Item>
__dpct_inline__ void
uninitialized_load_subgroup_striped(const Item &item, InputIteratorT block_itr,
                                    InputT (&items)[ITEMS_PER_WORK_ITEM]) {

  // This implementation does not take in account range loading across
  // workgroup items To-do: Decide whether range loading is required for group
  // loading
  // This implementation uses unintialized memory for loading linear segments
  // into warp striped arrangement.
  uint32_t subgroup_offset = item.get_sub_group().get_local_linear_id();
  uint32_t subgroup_size = item.get_sub_group().get_local_linear_range();
  uint32_t subgroup_idx = item.get_sub_group().get_group_linear_id();
  uint32_t initial_offset =
      (subgroup_idx * ITEMS_PER_WORK_ITEM * subgroup_size) + subgroup_offset;
#pragma unroll
  for (size_t idx = 0; idx < ITEMS_PER_WORK_ITEM; idx++) {
    new (&items[idx]) InputT(block_itr[initial_offset + (idx * subgroup_size)]);
  }
}

template <size_t ITEMS_PER_WORK_ITEM, load_algorithm ALGORITHM, typename InputT,
          typename InputIteratorT, typename Item>
class workgroup_load {
public:
  static size_t get_local_memory_size(size_t group_work_items) { return 0; }
  workgroup_load(uint8_t *local_memory) : _local_memory(local_memory) {}

  __dpct_inline__ void load(const Item &item, InputIteratorT block_itr,
                            InputT (&items)[ITEMS_PER_WORK_ITEM]) {

    if constexpr (ALGORITHM == BLOCK_LOAD_DIRECT) {
      //sycl::ext::oneapi::experimental::printf(" in direct ");
      load_blocked<ITEMS_PER_WORK_ITEM, InputT>(item, block_itr, items);
    } if constexpr (ALGORITHM == BLOCK_LOAD_STRIPED) {
      //sycl::ext::oneapi::experimental::printf(" in striped ");
      load_striped<ITEMS_PER_WORK_ITEM, InputT>(item, block_itr, items);
    }
  }

private:
  uint8_t *_local_memory;
};



*/








enum store_algorithm {

  BLOCK_STORE_DIRECT,
  BLOCK_STORE_STRIPED,
  // To-do: BLOCK_STORE_WARP_TRANSPOSE
  // To-do: BLOCK_STORE_VECTORIZE

};

/// Stores a blocked arrangement of work items linear segment of items.
template <size_t ITEMS_PER_WORK_ITEM, typename InputT,
          typename OutputIteratorT, typename Item>
__dpct_inline__ void store_blocked(const Item &item, OutputIteratorT block_itr,
                                  InputT (&items)[ITEMS_PER_WORK_ITEM]) {

  // This implementation does not take in account range storage across
  // workgroup items To-do: Decide whether range storage is required for group
  // storage
  size_t linear_tid = item.get_local_linear_id();
  OutputIteratorT workitem_itr = block_itr + (linear_tid * ITEMS_PER_WORK_ITEM);
#pragma unroll
  for (uint32_t idx = 0; idx < ITEMS_PER_WORK_ITEM; idx++) {
    workitem_itr[idx] = items[idx];
  }
}

/// Stores a striped arrangement of work items linear segment of items.
template <size_t ITEMS_PER_WORK_ITEM, typename InputT,
          typename OutputIteratorT, typename Item>
__dpct_inline__ void store_striped(const Item &item, OutputIteratorT block_itr,
                                  InputT (&items)[ITEMS_PER_WORK_ITEM]) {

  // This implementation does not take in account range storage across
  // workgroup items To-do: Decide whether range storage is required for group
  // storage
  size_t linear_tid = item.get_local_linear_id();
  OutputIteratorT workitem_itr = block_itr + linear_tid; 
  size_t GROUP_WORK_ITEMS = item.get_global_range().size();
#pragma unroll
  for (uint32_t idx = 0; idx < ITEMS_PER_WORK_ITEM; idx++) {
    workitem_itr[(idx * GROUP_WORK_ITEMS)] = items[idx];
  }
}

/// Stores a warp-striped arrangement of work items linear segment of items.
// Created as free function until exchange mechanism is
// implemented.
// To-do: inline this function with BLOCK_STORE_WARP_TRANSPOSE mechanism
template <size_t ITEMS_PER_WORK_ITEM, typename InputT, typename OutputIteratorT,
          typename Item>
__dpct_inline__ void
store_subgroup_striped(const Item &item, OutputIteratorT block_itr,
                                    InputT (&items)[ITEMS_PER_WORK_ITEM]) {

  // This implementation does not take in account range loading across
  // workgroup items To-do: Decide whether range loading is required for group
  // loading
  // This implementation uses unintialized memory for loading linear segments
  // into warp striped arrangement.
  uint32_t subgroup_offset = item.get_sub_group().get_local_linear_id();
  uint32_t subgroup_size = item.get_sub_group().get_local_linear_range();
  uint32_t subgroup_idx = item.get_sub_group().get_group_linear_id();
  uint32_t initial_offset =
      (subgroup_idx * ITEMS_PER_WORK_ITEM * subgroup_size) + subgroup_offset;
  OutputIteratorT workitem_itr = block_itr + initial_offset;
#pragma unroll
  for (uint32_t idx = 0; idx < ITEMS_PER_WORK_ITEM; idx++) {
    workitem_itr[(idx * subgroup_size)] = items[idx];
  }
}

// template parameters :
// ITEMS_PER_WORK_ITEM: size_t variable controlling the number of items per
// thread/work_item
// ALGORITHM: store_algorithm variable controlling the type of store operation.
// InputT: type for input sequence.
// OutputIteratorT:  output iterator type
// Item : typename parameter resembling sycl::nd_item<3> .
template <size_t ITEMS_PER_WORK_ITEM, store_algorithm ALGORITHM, typename InputT,
          typename OutputIteratorT, typename Item>
class workgroup_store {
public:
  static size_t get_local_memory_size(size_t group_work_items) { return 0; }
  workgroup_store(uint8_t *local_memory) : _local_memory(local_memory) {}
  
  __dpct_inline__ void store(const Item &item, OutputIteratorT block_itr,
                            InputT (&items)[ITEMS_PER_WORK_ITEM]) {

    if constexpr (ALGORITHM == BLOCK_STORE_DIRECT) {
      store_blocked<ITEMS_PER_WORK_ITEM>(item, block_itr, (&items)[ITEMS_PER_WORK_ITEM]);
    } else if constexpr (ALGORITHM == BLOCK_STORE_STRIPED) {
      store_striped<ITEMS_PER_WORK_ITEM>(item, block_itr, (&items)[ITEMS_PER_WORK_ITEM]);
    }
  }
  
private:
  uint8_t *_local_memory;

};









}
}



template <typename T, int VALUES_PER_THREAD> class exchange {
public:
  static size_t get_local_memory_size(size_t group_threads) {
    size_t padding_values =
        (INSERT_PADDING)
            ? ((group_threads * VALUES_PER_THREAD) >> LOG_LOCAL_MEMORY_BANKS)
            : 0;
    return (group_threads * VALUES_PER_THREAD + padding_values) * sizeof(T);
  }

  exchange(uint8_t *local_memory) : _local_memory(local_memory) {}
public:
  // TODO: Investigate if padding is required for performance,
  // and if specializations are required for specific target hardware.
  static size_t adjust_by_padding(size_t offset) {

    if constexpr (INSERT_PADDING) {
      offset = dpct::group::detail::shr_add(offset, LOG_LOCAL_MEMORY_BANKS, offset);
    }
    return offset;
  }

  struct get_blocked_offset {
    template <typename Item> size_t operator()(Item item, size_t i) {
      size_t offset = item.get_local_id(0) * VALUES_PER_THREAD + i;
      return adjust_by_padding(offset);
    }
  };

  struct get_striped_offset {
    template <typename Item> size_t operator()(Item item, size_t i) {
      size_t offset = i * item.get_local_range(2) * item.get_local_range(1) *
                          item.get_local_range(0) +
                      item.get_local_id(0);
      return adjust_by_padding(offset);
    }
  };

  template <typename Iterator> struct get_scatter_offset {
    Iterator begin;
    get_scatter_offset(const int (&ranks)[VALUES_PER_THREAD]) { begin = ranks; }
    template <typename Item> size_t operator()(Item item, size_t i) const {
      // iterator i is expected to be within bounds [0,VALUES_PER_THREAD)
      return adjust_by_padding(begin[i]);
    }
  };

  template <typename Item, typename offsetFunctorTypeFW,
            typename offsetFunctorTypeRV>
  __dpct_inline__ void helper_exchange(Item item, T (&keys)[VALUES_PER_THREAD],
                                       offsetFunctorTypeFW &offset_functor_fw,
                                       offsetFunctorTypeRV &offset_functor_rv) {

    T *buffer = reinterpret_cast<T *>(_local_memory);

    #pragma unroll
    for (size_t i = 0; i < VALUES_PER_THREAD; i++) {
      size_t offset = offset_functor_fw(item, i);
      buffer[offset] = keys[i];
    }

    item.barrier(sycl::access::fence_space::local_space);

    #pragma unroll
    for (size_t i = 0; i < VALUES_PER_THREAD; i++) {
      size_t offset = offset_functor_rv(item, i);
      keys[i] = buffer[offset];
    }
  }
public:
  /// Rearrange elements from blocked order to striped order
  template <typename Item>
  __dpct_inline__ void blocked_to_striped(Item item,
                                          T (&keys)[VALUES_PER_THREAD]) {

    get_striped_offset getStripedOffset;
    get_blocked_offset getBlockedOffset;
    helper_exchange(item, keys, getStripedOffset, getBlockedOffset);
  }

  /// Rearrange elements from striped order to blocked order
  template <typename Item>
  __dpct_inline__ void striped_to_blocked(Item item,
                                          T (&keys)[VALUES_PER_THREAD]) {

    get_blocked_offset getBlockedOffset;
    get_striped_offset getStripedOffset;
    helper_exchange(item, keys, getBlockedOffset, getStripedOffset);
  }

  /// Rearrange elements from rank order to blocked order
  template <typename Item>
  __dpct_inline__ void scatter_to_blocked(Item item,
                                          T (&keys)[VALUES_PER_THREAD],
                                          int (&ranks)[VALUES_PER_THREAD]) {

    get_scatter_offset<int *> getScatterOffset(ranks);
    get_blocked_offset getBlockedOffset;
    helper_exchange(item, keys, getScatterOffset, getBlockedOffset);
  }

  /// Rearrange elements from scatter order to striped order
  template <typename Item>
  __dpct_inline__ void scatter_to_striped(Item item,
                                          T (&keys)[VALUES_PER_THREAD],
                                          int (&ranks)[VALUES_PER_THREAD]) {

    get_scatter_offset<int *> getScatterOffset(ranks);
    get_striped_offset getStripedOffset;
    helper_exchange(item, keys, getScatterOffset, getStripedOffset);
  }

private:
  static constexpr int LOG_LOCAL_MEMORY_BANKS = 4;
  static constexpr bool INSERT_PADDING =
      (VALUES_PER_THREAD > 4) &&
      (dpct::group::detail::power_of_two<VALUES_PER_THREAD>::VALUE);

  uint8_t *_local_memory;
};


template <typename T, int VALUES_PER_THREAD, bool DESCENDING = false>
class radix_sort {
public:
  static size_t get_local_memory_size(size_t group_threads) {
    size_t ranks_size =
        dpct::group::detail::radix_rank<RADIX_BITS>::get_local_memory_size(group_threads);
    size_t exchange_size =
        dpct::group::exchange<T, VALUES_PER_THREAD>::get_local_memory_size(group_threads);
    return sycl::max(ranks_size, exchange_size);
  }

  radix_sort(uint8_t *local_memory) : _local_memory(local_memory) {}

  template <typename Item>
  __dpct_inline__ void
  helper_sort(const Item &item, T (&keys)[VALUES_PER_THREAD], int begin_bit = 0,
              int end_bit = 8 * sizeof(T), bool is_striped = false) {

    uint32_t(&unsigned_keys)[VALUES_PER_THREAD] =
        reinterpret_cast<uint32_t(&)[VALUES_PER_THREAD]>(keys);

   #pragma unroll
    for (int i = 0; i < VALUES_PER_THREAD; ++i) {
      unsigned_keys[i] = dpct::group::detail::traits<T>::twiddle_in(unsigned_keys[i]);
    }

    for (int i = begin_bit; i < end_bit; i += RADIX_BITS) {
      int pass_bits = sycl::min(RADIX_BITS, end_bit - begin_bit);

      int ranks[VALUES_PER_THREAD];
      dpct::group::detail::radix_rank<RADIX_BITS, DESCENDING>(_local_memory)
          .template rank_keys(item, unsigned_keys, ranks, i, pass_bits);

      item.barrier(sycl::access::fence_space::local_space);

      bool last_iter = i + RADIX_BITS > end_bit;
      if (last_iter && is_striped) {
        dpct::group::exchange<T, VALUES_PER_THREAD>(_local_memory)
            .scatter_to_striped(item, keys, ranks);

      } else {
        dpct::group::exchange<T, VALUES_PER_THREAD>(_local_memory)
            .scatter_to_blocked(item, keys, ranks);
      }

      item.barrier(sycl::access::fence_space::local_space);
    }

   #pragma unroll
    for (int i = 0; i < VALUES_PER_THREAD; ++i) {
      unsigned_keys[i] = dpct::group::detail::traits<T>::twiddle_out(unsigned_keys[i]);
    }
  }

  template <typename Item>
  __dpct_inline__ void
  sort_blocked(const Item &item, T (&keys)[VALUES_PER_THREAD],
               int begin_bit = 0, int end_bit = 8 * sizeof(T)) {
    helper_sort(item, keys, begin_bit, end_bit, false);
  }

  template <typename Item>
  __dpct_inline__ void
  sort_blocked_to_striped(const Item &item, T (&keys)[VALUES_PER_THREAD],
                          int begin_bit = 0, int end_bit = 8 * sizeof(T)) {
    helper_sort(item, keys, begin_bit, end_bit, true);
  }

private:
  static constexpr int RADIX_BITS = 4;

  uint8_t *_local_memory;
};







// source: https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
// sign function for lion
// taken from https://stackoverflow.com/a/4609795, but not sure if there's a proper way to do this in CUDA

template <typename T> int sgn(T val)
{
  return (T(0) < val) - (val < T(0));
}

template <int STOCHASTIC>
unsigned char dQuantize(float* smem_code, const float rand, float x)
{
    int pivot = 127;
    int upper_pivot = 255;
    int lower_pivot = 0;

    float lower = -1.0f;
    float upper = 1.0f;

    float val = smem_code[pivot];
    // i>>=1 = {32, 16, 8, 4, 2, 1}
    for(int i = 64; i > 0; i>>=1)
    {
        if(x > val)
        {
            lower_pivot = pivot;
            lower = val;
            pivot+=i;
        }
        else
        {
            upper_pivot = pivot;
            upper = val;
            pivot-=i;
        }
        val = smem_code[pivot];
    }

    if(upper_pivot == 255)
        upper = smem_code[upper_pivot];
    if(lower_pivot == 0)
        lower = smem_code[lower_pivot];

    if(!STOCHASTIC)
    {
      if(x > val)
      {
        float midpoint = (upper+val)*0.5f;
        if(x > midpoint)
        {
          return upper_pivot;
        }
        else
          return pivot;
      }
      else
      {
        float midpoint = (lower+val)*0.5f;
        if(x < midpoint)
          return lower_pivot;
        else
          return pivot;
      }
    }
    else
    {
      if(x > val)
      {
        float dist_to_upper = sycl::fabs(upper-x);
        float dist_full = upper-val;
        if(rand >= dist_to_upper/dist_full) return upper_pivot;
        else return pivot;
      }
      else
      {
        float dist_to_lower = sycl::fabs(lower-x);
        float dist_full = val-lower;
        if(rand >= dist_to_lower/dist_full) return lower_pivot;
        else return pivot;
      }
    }
}

template <int SIGNED>
__dpct_inline__ unsigned char quantize_2D(float *__restrict__ quadrants, float x)
{
    int pivot = 127;
    int upper_pivot = 255;
    int lower_pivot = 0;

    float lower = SIGNED ? -1.0f : 0.0f;
    float upper = 1.0f;
    float midpoint;
    float val = quadrants[1];
    int local_pivot = 1;
    int offset = 1;

    // i>>=1 = {32, 16, 8, 4, 2, 1}
    for(int i = 64; i > 0; i>>=1)
    {
        if(x > val)
        {
            lower_pivot = pivot;
            lower = val;
            pivot+=i;
            //val = i == 64 ? quadrants[2] : smem_code[pivot];
            local_pivot += offset;
        }
        else
        {
            upper_pivot = pivot;
            upper = val;
            pivot-=i;
            //val = i == 64 ? quadrants[0] : smem_code[pivot];
            local_pivot -= offset;
        }
        val = i >= 64 ? quadrants[local_pivot] : 0;//smem_code[pivot];
        offset -= 1;
    }

    if(x > val)
    {
      midpoint = (upper+val)*0.5f;
      if(x > midpoint)
        return upper_pivot;
      else
        return pivot;
    }
    else
    {
      midpoint = (lower+val)*0.5f;
      if(x < midpoint)
        return lower_pivot;
      else
        return pivot;
    }
}

template <int SIGNED>
__dpct_inline__ unsigned char quantize_quadrant(int QUADRANT, float *__restrict__ const smem_code, float x, float lower, float midpoint, float upper)
{
    int lower_pivot = QUADRANT*16-1 - 0;
    int pivot = QUADRANT*16-1 + 16;
    int upper_pivot = QUADRANT*16-1 + 31;

    float val = midpoint;

    // i>>=1 = {32, 16, 8, 4, 2, 1}
    for(int i = 16; i > 0; i>>=1)
    {
        if(x > val)
        {
            lower_pivot = pivot;
            lower = val;
            pivot+=i;
        }
        else
        {
            upper_pivot = pivot;
            upper = val;
            pivot-=i;
        }
        val = smem_code[pivot];
    }

    if(x > val)
    {
      midpoint = (upper+val)*0.5f;
      if(x > midpoint)
        return upper_pivot;
      else
        return pivot;
    }
    else
    {
      midpoint = (lower+val)*0.5f;
      if(x < midpoint)
        return lower_pivot;
      else
        return pivot;
    }
}

SYCL_EXTERNAL void kHistogramScatterAdd2D(float* histogram, int *index1, int *index2, float *src, const int maxidx1, const int n,
                            const sycl::nd_item<3> &item_ct1)
{
  const int tid = item_ct1.get_local_id(2) + (item_ct1.get_local_range(2)*item_ct1.get_group(2));
  const int numThreads = item_ct1.get_local_range(2)*item_ct1.get_group_range(2);

  for(int i = tid; i < n; i+=numThreads)
  {
      int idx = (index1[i]*maxidx1) + index2[i];
      dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&histogram[idx], src[i]);
  }
}

template<typename T, int BLOCK_SIZE, int NUM_MAX>
void kCompressMax(T * __restrict__ const A, T* out, unsigned char* out_idx, const int n,
                  const sycl::nd_item<3> &item_ct1, int *smem_max_indices,
                  float *smem_max_values)
{
  
  
  const int warp_idx = item_ct1.get_local_id(2)/32;
  const int valid_items = n - (item_ct1.get_group(2)*BLOCK_SIZE) > BLOCK_SIZE ? BLOCK_SIZE : n - (item_ct1.get_group(2)*BLOCK_SIZE);

  //  BLOCK_SIZE/32 == number of warps
  T values[8];
  T max1 = -64000.0f;
  T max2 = -64000.0f;
  int max_idx1 = -1;
  int max_idx2 = -1;
  int sign1 = -1;
  int sign2 = -1;

  sycl::buffer<T, 1>  buff_indices(smem_max_indices, sycl::range<1>(8*BLOCK_SIZE/32));
  sycl::buffer<T, 1>  buff_values(smem_max_values, sycl::range<1>(8*BLOCK_SIZE/32));
  sycl::buffer<T, 1>  buff_A(A,sycl::range<1>(8*BLOCK_SIZE/32));
  
  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
 
      using group_load = dpct::group::workgroup_load<8, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, int,  int *, sycl::nd_item<3>>;
      size_t temp_storage_size = group_load::get_local_memory_size(8*BLOCK_SIZE/32);
      sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
      sycl::accessor dacc_A(buff_A[(item_ct1.get_local_id(2)*BLOCK_SIZE)], cgh, sycl::read_write);
      
      // 1. load 8 values per thread
      // 2. compute 2-max in registers (64 max per warp)
      // 3. do warp reduction + broadcast back
      // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
      // 5. Repeat (3) 8 times for top 8 values in 256
      // 6. store with byte index
      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item) {
          auto *d_A = dacc_A.template get_multi_ptr<sycl::access::decorated::yes>().get();
          auto *tmp = tacc.get_multi_ptr<sycl::access::decorated::yes>().get();
          group_load(tmp).load(item_ct1, d_A, values);
        
        });
      
  });
   
  #pragma unroll 8
  for(int i = 0; i < 8; i++)
  {
    T absval = fabsf(values[i]);
    if(absval > max1)
    {
      max1 = values[i];
      sign1 = signbit(values[i]);
      max_idx1 = 8*item_ct1.get_local_id(2) + i;
    }
    else if(absval > max2)
    {
      max2 = values[i];
      sign2 = signbit(values[i]);
      max_idx2 = 8*item_ct1.get_local_id(2) + i;
    }
  }
  
  float warp_max;
  sycl::host_accessor hacc_values{buff_values};
  sycl::host_accessor hacc_indices{buff_indices};
  for(int i = 0; i < 8; i++)
  {
    // 3. do warp reduction + broadcast back
    
    auto output = sycl::reduce_over_group(item_ct1.get_sub_group(), max1, sycl::maximum<>());
    warp_max = item_ct1.get_sub_group().shuffle(warp_max, 0);

    // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
    if(warp_max == max1)
    {
	
      hacc_values[warp_idx*8 + i] = sign1 != 0 ? -max1 : max1;
      hacc_indices[warp_idx*8 + i] = max_idx1;

      sign1 = sign2;
      max1 = max2;
      max_idx1 = max_idx2;

      max2 = -64000.0f;
    }
    sycl::group_barrier(item_ct1.get_sub_group());
  }

  if(item_ct1.get_local_id(2) % 32 < 8)
  {
    // offset: 8 values per 256 input values
    //
    int offset = BLOCK_SIZE*item_ct1.get_group(2)*BLOCK_SIZE/32*8;
  }

}

#define THREADS_ESTIMATE 512
#define NUM_ESTIMATE 8
#define BLOCK_ESTIMATE 4096


//================typedefs===================================

typedef sycl::local_accessor<uint8_t, 1> sycl_la;
typedef sycl::accessor<int, 1> sycl_dacc;
typedef sycl::accessor<float, 1> sycl_dacc_float;
typedef sycl::accessor<unsigned char, 1> sycl_dacc_uc;

//======================gemm device=====================================

#define WARPS 3 

template <typename T, int BITS, int THREADS> SYCL_EXTERNAL void gemm_device(int M, int N, int K, T * __restrict__ const A,  T* B,  T * out,  int lda, int ldb, int ldc, const sycl::nd_item<3> &item_ct1, T *smem_A, T *smem_B, const sycl::accessor<T, 1> &dacc_A, const sycl::accessor<T, 1> &dacc_B, const sycl::accessor<T, 1> &dacc_out)
{

#if DPCT_COMPATIBILITY_TEMP >= 750
	
  int col_offset = item_ct1.get_group(2) *32;
  const int warp_id = item_ct1.get_local_id(2) / 32;
  const int half_warp_id = item_ct1.get_local_id(2) / 16;
  const int half_warp_lane = item_ct1.get_local_id(2) % 16;
  const int batch_size_warps = (WARPS-1)*2;
  const int val_per_iter = item_ct1.get_local_range(2)-32;

  T local_A[4];
  T local_B[128];

  const int a_tile_offset = 16;
  const int b_tile_offset = (16*32 + 16);

  auto d_A = dacc_A.template get_multi_ptr<sycl::access::decorated::yes>();
  auto d_B = dacc_B.template get_multi_ptr<sycl::access::decorated::yes>();
  auto sg_size = item_ct1.get_sub_group();
  
   
   sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, T, sycl::ext::oneapi::experimental::matrix::use::a, 8, 16, sycl::ext::oneapi::experimental::matrix::layout::row_major> a_frag{};
   sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, T, sycl::ext::oneapi::experimental::matrix::use::b, 16, 32, sycl::ext::oneapi::experimental::matrix::layout::col_major> b_frag{};
   sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, T, sycl::ext::oneapi::experimental::matrix::use::accumulator, 8, 32> c_frag{};
   sycl::ext::oneapi::experimental::matrix::joint_matrix_fill(item_ct1.get_sub_group(), c_frag, 0.0f);

  //wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> a_frag;
  //wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> b_frag;
  //wmma::fragment<wmma::accumulator, 8, 32, 16, half> c_frag; 
  //wmma::fill_fragment(c_frag, 0.0f);
  
  int ticktock = 0;
  int idx = 0 + item_ct1.get_local_id(2);
  int loaded_values = 0;
  // prefetch
  if(idx < K && warp_id < (WARPS-1))
  {
    if(loaded_values == 0)
    {
      local_A[0] = dacc_A[idx];
      local_A[1] = dacc_A[idx+(1*val_per_iter)];
      local_A[2] = dacc_A[idx+(2*val_per_iter)];
      local_A[3] = dacc_A[idx+(3*val_per_iter)];

      #pragma unroll 32
      for(int col = 0; col < 32; col++)
      {
        local_B[col] = dacc_B[(col_offset+col)*ldb+idx];
        local_B[col+32] = dacc_B[(col_offset+col)*ldb+idx+(1*val_per_iter)];
        local_B[col+64] = dacc_B[(col_offset+col)*ldb+idx+(2*val_per_iter)];
        local_B[col+96] = dacc_B[(col_offset+col)*ldb+idx+(3*val_per_iter)];
      }
      loaded_values = 3;
    }
    else
    {

      if(loaded_values == 3)
      {
        local_A[0] = local_A[1];
        #pragma unroll 32
        for(int col = 0; col < 32; col++)
          local_B[col] = local_B[col+(32)];
      }
      else if(loaded_values == 2)
      {
        local_A[0] = local_A[2];
        #pragma unroll 32
        for(int col = 0; col < 32; col++)
          local_B[col] = local_B[col+(64)];
      }
      else
      {
        local_A[0] = local_A[3];
        #pragma unroll 32
        for(int col = 0; col < 32; col++)
          local_B[col] = local_B[col+(96)];
      }
      loaded_values--;
    }

    smem_A[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*a_tile_offset)] = local_A[0];

    #pragma unroll 32
    for(int col = 0; col < 32; col++)
        smem_B[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*b_tile_offset) + (col*16)] = local_B[col];
  }
  else if(warp_id < (WARPS-1))
  {
    local_A[0] = T(0.0);
    smem_A[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*a_tile_offset)] =  0.0f;

    #pragma unroll 32
    for(int col = 0; col < 32; col++)
      local_B[col] = 0.0f;

    #pragma unroll 32
    for(int col = 0; col < 32; col++)
      smem_B[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*b_tile_offset) + (col*16)] = 0.0f;
  }
  ticktock = ticktock == 0 ? 1 : 0;

  //for(int base_idx = blockDim.x-32; base_idx < K; base_idx+=blockDim.x-32)
  for(int base_idx = item_ct1.get_local_range(2)-32; base_idx < K; base_idx+=item_ct1.get_local_range(2)-32)
  {
    idx = base_idx + item_ct1.get_local_id(2);

    item_ct1.barrier(sycl::access::fence_space::local_space);
    if(idx < K && warp_id < (WARPS-1))
    {
      //local_A[0] = A[idx];

      //#pragma unroll 32
      //for(int col = 0; col < 32; col++)
      //  local_B[col] = B[(col_offset+col)*ldb+idx];
      if(loaded_values == 0)
      {
        local_A[0] = dacc_A[idx];
        local_A[1] = dacc_A[idx+(1*val_per_iter)];
        local_A[2] = dacc_A[idx+(2*val_per_iter)];
        local_A[3] = dacc_A[idx+(3*val_per_iter)];

        #pragma unroll 32
        for(int col = 0; col < 32; col++)
        {
          local_B[col] = dacc_B[(col_offset+col)*ldb+idx];
          local_B[col+32] = dacc_B[(col_offset+col)*ldb+idx+(1*val_per_iter)];
          local_B[col+64] = dacc_B[(col_offset+col)*ldb+idx+(2*val_per_iter)];
          local_B[col+96] = dacc_B[(col_offset+col)*ldb+idx+(3*val_per_iter)];
        }
        loaded_values = 3;

      }
      else
      {

        if(loaded_values == 3)
        {
          local_A[0] = local_A[1];
          #pragma unroll 32
          for(int col = 0; col < 32; col++)
            local_B[col] = local_B[col+(32)];
        }
        else if(loaded_values == 2)
        {
          local_A[0] = local_A[2];
          #pragma unroll 32
          for(int col = 0; col < 32; col++)
            local_B[col] = local_B[col+(64)];
        }
        else
        {
          local_A[0] = local_A[3];
          #pragma unroll 32
          for(int col = 0; col < 32; col++)
            local_B[col] = local_B[col+(96)];
        }
        loaded_values--;
      }

      smem_A[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*a_tile_offset)] = local_A[0];

      #pragma unroll 32
      for(int col = 0; col < 32; col++)
          smem_B[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*b_tile_offset) + (col*16)] = local_B[col];
    }
    else if(warp_id < (WARPS-1))
    {
      local_A[0] = T(0.0);
      smem_A[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*a_tile_offset)] =  0.0f;

      #pragma unroll 32
      for(int col = 0; col < 32; col++)
        local_B[col] = 0.0f;

      #pragma unroll 32
      for(int col = 0; col < 32; col++)
        smem_B[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*b_tile_offset) + (col*16)] = 0.0f;
    }
    ticktock = ticktock == 0 ? 1 : 0;
    
    
    if(warp_id == (WARPS-1))
      for(int k = 0; k < batch_size_warps; k++)
      {
        
        //wmma::load_matrix_sync(a_frag, &(smem_A[(ticktock*batch_size_warps + k)*a_tile_offset]), 16); //  111 mu
        
        dacc_A[(ticktock*batch_size_warps + k)*a_tile_offset] = smem_A[(ticktock*batch_size_warps + k)*a_tile_offset];
        dacc_B[(ticktock*batch_size_warps + k)*b_tile_offset] = smem_B[(ticktock*batch_size_warps + k)*b_tile_offset];
        d_A = dacc_A.template get_multi_ptr<sycl::access::decorated::yes>();
        d_B = dacc_B.template get_multi_ptr<sycl::access::decorated::yes>();
    
        sycl::ext::oneapi::experimental::matrix::joint_matrix_load(sg_size, a_frag, d_A, 16);
    
        
        //wmma::load_matrix_sync(b_frag, &(smem_B[(ticktock*batch_size_warps + k)*b_tile_offset]), 16); // 35 mu
        sycl::ext::oneapi::experimental::matrix::joint_matrix_load(sg_size, a_frag,  d_B, 16);
    
        //wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        sycl::ext::oneapi::experimental::matrix::joint_matrix_mad(sg_size, c_frag, a_frag, b_frag, c_frag);
    }
   
  }
  
  item_ct1.barrier(sycl::access::fence_space::local_space);
  if(warp_id != (WARPS-1)){ return; }
  // only warp_id == (WARPS-1) from here
  int warp_lane = item_ct1.get_local_id(2) % 32;

  ticktock = ticktock == 0 ? 1 : 0;
  for(int k = 0; k < batch_size_warps; k++)
  {
      dacc_A[(ticktock*batch_size_warps + k)*a_tile_offset] = smem_A[(ticktock*batch_size_warps + k)*a_tile_offset];
      dacc_B[(ticktock*batch_size_warps + k)*b_tile_offset] = smem_B[(ticktock*batch_size_warps + k)*b_tile_offset];
      d_A = dacc_A.template get_multi_ptr<sycl::access::decorated::yes>();
      d_B = dacc_B.template get_multi_ptr<sycl::access::decorated::yes>();
      
     //wmma::load_matrix_sync(a_frag, &(smem_A[(ticktock*batch_size_warps + k)*a_tile_offset]), 16); //  111 mu
     sycl::ext::oneapi::experimental::matrix::joint_matrix_load(sg_size, a_frag, d_A, 16);
     
     //wmma::load_matrix_sync(b_frag, &(smem_B[(ticktock*batch_size_warps + k)*b_tile_offset]), 16); // 35 mu
     sycl::ext::oneapi::experimental::matrix::joint_matrix_load(sg_size, b_frag, d_B, 16);                
      
     //wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
     sycl::ext::oneapi::experimental::matrix::joint_matrix_mad(sg_size, c_frag, a_frag, b_frag, c_frag);
  }

  // 129 mu
  if(warp_id == (WARPS-1))
     
    //wmma::store_matrix_sync(&(smem_A[0]), c_frag, 32, wmma::mem_row_major);
    sycl::ext::oneapi::experimental::matrix::joint_matrix_store(sg_size, c_frag, d_A, (size_t)32, sycl::ext::oneapi::experimental::matrix::layout::row_major);
    
  if(col_offset + warp_lane < M)
    dacc_out[col_offset + warp_lane] = smem_A[warp_lane];
#endif
}
//====================================================================================================

//=======================================4 bit gemm===============================

const float nf4_data[16] = {-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0};



template <typename T, int THREADS> SYCL_EXTERNAL void kgemm_4bit_inference(int M, int N, int K, T * __restrict__ const A, unsigned char *B,  float *absmax, T * out,  int lda, int ldb, int ldc, int blocksize, const sycl::nd_item<3> &item_ct1, T *smem_A, unsigned char *smem_B, T *smem_C,
const sycl::accessor<T, 1> &dacc_A, const sycl_dacc_uc &dacc_B, const sycl::accessor<T, 1> &dacc_out)
{

#if DPCT_COMPATIBILITY_TEMP >= 750
	
  int col_offset = item_ct1.get_group(2) *32;
  const int warp_id = item_ct1.get_local_id(2) / 32;
  const int warp_idx = item_ct1.get_local_id(2) % 32;
  const int half_warp_id = item_ct1.get_local_id(2) / 16;
  const int half_warp_lane = item_ct1.get_local_id(2) % 16;
  const int batch_size_warps = (WARPS-1)*2;

  T quant_map[16];

  #pragma unroll 16
  for(int i = 0; i < 16; i++)
    quant_map[i] = nf4_data[i];
  //__shared__ T quant_map[16*160];

  T local_A[2];
  T local_B[64];
  unsigned char local_B_4bit[32];


  const int a_tile_offset = 16;
  const int b_tile_offset = (16*32 + 16);

  auto d_A = dacc_A.template get_multi_ptr<sycl::access::decorated::yes>();
  auto d_B = dacc_B.get_multi_ptr<sycl::access::decorated::yes>();
  auto sg_size = item_ct1.get_sub_group();
  
   
   sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, T, sycl::ext::oneapi::experimental::matrix::use::a, 8, 16, sycl::ext::oneapi::experimental::matrix::layout::row_major> a_frag{};
   sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, unsigned char, sycl::ext::oneapi::experimental::matrix::use::b, 16, 32, sycl::ext::oneapi::experimental::matrix::layout::col_major> b_frag{};
   sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, T, sycl::ext::oneapi::experimental::matrix::use::accumulator, 8, 32> c_frag{};
   sycl::ext::oneapi::experimental::matrix::joint_matrix_fill(sg_size, c_frag, 0.0f); 
   
  

  //wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> a_frag;
  //wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> b_frag;
  //wmma::fragment<wmma::accumulator, 8, 32, 16, half> c_frag; 
  //wmma::fill_fragment(c_frag, 0.0f);
   
  for(int i = item_ct1.get_local_id(2); i < (8*32); i+=item_ct1.get_local_range(2))
    smem_C[i] = 0.0f;

  item_ct1.barrier(sycl::access::fence_space::local_space);

  int ticktock = 0;
  int idx = 0 + item_ct1.get_local_id(2);
  int loaded_values = 0;
  // prefetch
  if(idx < K && warp_id < (WARPS-1))
  {
    if(loaded_values == 0)
    {
      local_A[0] = dacc_A[idx];
      local_A[1] = dacc_A[idx+item_ct1.get_local_range(2)-32];

      #pragma unroll 32
      for(int col = 0; col < 32; col++)
        local_B_4bit[col] = dacc_B[(col_offset+col)*ldb+idx];

      loaded_values = 1;
    }
    else
    {
      local_A[0] = local_A[1];
      loaded_values--;

      #pragma unroll 64
      for(int col = 0; col < 64; col+=2)
      {
        //local_B[col] = dhDequantizeNF4(local_B_4bit[col/2] >> 4)*T(1.0f);
        //local_B[col+1] = dhDequantizeNF4(local_B_4bit[col/2] & 0x0F)*T(1.0f);
        //local_B[col] = d2DequantizeFP4(local_B_4bit[col/2] >> 4)*(float)(17.0);
        //local_B[col+1] = d2DequantizeFP4(local_B_4bit[col/2] & 0x0F)*(float)(17.0);
        //local_B[col] = 127*(local_B_4bit[col/2] >> 4)*(float)(17.0);
        //local_B[col+1] = 127*(local_B_4bit[col/2] & 0x0F)*(float)(17.0);

        //local_B[col] = quant_map[(local_B_4bit[col/2] >> 4)]*T(17.0);
        //local_B[col+1] = quant_map[(local_B_4bit[col/2] & 0x0F)]*T(17.0);
        local_B[col] = quant_map[160*(local_B_4bit[col/2] >> 4)+warp_idx]*T(17.0);
        local_B[col+1] = quant_map[160*(local_B_4bit[col/2] & 0x0F)+warp_idx]*T(17.0);
      }
    }

    smem_A[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*a_tile_offset)] = local_A[0];

    #pragma unroll 32
    for(int col = 0; col < 32; col++)
        smem_B[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*b_tile_offset) + (col*16)] = local_B[col];
  }
  else if(warp_id < (WARPS-1))
  {
    local_A[0] = T(0.0);
    smem_A[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*a_tile_offset)] =  0.0f;

    #pragma unroll 32
    for(int col = 0; col < 32; col++)
      local_B[col] = 0.0f;

    #pragma unroll 32
    for(int col = 0; col < 32; col++)
      smem_B[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*b_tile_offset) + (col*16)] = 0.0f;
  }
  ticktock = ticktock == 0 ? 1 : 0;
    //if(threadIdx.x == 0)
      //printf("aa %i %i\n", idx, loaded_values);

  //for(int base_idx = blockDim.x-32; base_idx < K; base_idx+=blockDim.x-32)
  for(int base_idx = item_ct1.get_local_range(2)-32; base_idx < K; base_idx+=item_ct1.get_local_range(2)-32)
  {
    idx = base_idx + item_ct1.get_local_id(2);
    //if(threadIdx.x == 0)
      //printf("%i %i\n", idx, loaded_values);

    //__syncthreads();
    if(idx < K && warp_id < (WARPS-1))
    {
      if(loaded_values == 0)
      {
        local_A[0] = dacc_A[idx];
        local_A[1] = dacc_A[idx+item_ct1.get_local_range(2)-32];

        #pragma unroll 32
        for(int col = 0; col < 32; col++)
        {
          local_B_4bit[col] = dacc_B[(col_offset+col)*ldb+idx];
          local_B_4bit[col+16] = dacc_B[(col_offset+col)*ldb+idx];
        }

        loaded_values = 1;
      }
      else
      {
        local_A[0] = local_A[1];
        loaded_values--;

        int absidx = (idx + col_offset)/blocksize;
        
        sycl::half local_absmax =  absmax[absidx];

        #pragma unroll 64
        for(int col = 0; col < 64; col+=2)
        {
          //local_B[col] = dhDequantizeNF4(local_B_4bit[col/2] >> 4)*T(absidx);
          //local_B[col+1] = dhDequantizeNF4(local_B_4bit[col/2] & 0x0F)*T(absidx);
          //local_B[col] = T(127)*T(local_B_4bit[col/2] >> 4)*T(absidx);
          //local_B[col+1] = T(127)*T(local_B_4bit[col/2] & 0x0F)*T(absidx);

          //local_B[col] = quant_map[160*(local_B_4bit[col/2] >> 4)+warp_idx]*T(local_absmax);
          //local_B[col+1] = quant_map[160*(local_B_4bit[col/2] & 0x0F)+warp_idx]*T(local_absmax);
          local_B[col] = quant_map[(local_B_4bit[col/2] >> 4)]*T(absidx);
          local_B[col+1] = quant_map[(local_B_4bit[col/2] & 0x0F)]*T(absidx);
        }
        //printnonzero<T>(local_B, 128, "");
      }

      smem_A[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*a_tile_offset)] = local_A[0];

      #pragma unroll 32
      for(int col = 0; col < 32; col++)
          smem_B[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*b_tile_offset) + (col*16)] = local_B[col];
    }
    else if(warp_id < (WARPS-1))
    {
      local_A[0] = T(0.0);
      smem_A[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*a_tile_offset)] =  0.0f;

      #pragma unroll 32
      for(int col = 0; col < 32; col++)
        local_B[col] = 0.0f;

      #pragma unroll 32
      for(int col = 0; col < 32; col++)
        smem_B[half_warp_lane + (((batch_size_warps*ticktock)+half_warp_id)*b_tile_offset) + (col*16)] = 0.0f;
    }
    ticktock = ticktock == 0 ? 1 : 0;

    if(warp_id == (WARPS-1))
      for(int k = 0; k < batch_size_warps; k++)
      {
        
        dacc_A[(ticktock*batch_size_warps + k)*a_tile_offset] = smem_A[(ticktock*batch_size_warps + k)*a_tile_offset];
        dacc_B[(ticktock*batch_size_warps + k)*b_tile_offset] = smem_B[(ticktock*batch_size_warps + k)*b_tile_offset];
        d_A = dacc_A.template get_multi_ptr<sycl::access::decorated::yes>();
        d_B = dacc_B.get_multi_ptr<sycl::access::decorated::yes>();
        
        //wmma::load_matrix_sync(a_frag, &(smem_A[(ticktock*batch_size_warps + k)*a_tile_offset]), 16); //  111 mu
        sycl::ext::oneapi::experimental::matrix::joint_matrix_load(sg_size, a_frag, d_A, 16);
    
        
        //wmma::load_matrix_sync(b_frag, &(smem_B[(ticktock*batch_size_warps + k)*b_tile_offset]), 16); // 35 mu
        sycl::ext::oneapi::experimental::matrix::joint_matrix_load(sg_size, b_frag, d_B, 16);
    
        
        //wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        sycl::ext::oneapi::experimental::matrix::joint_matrix_mad(sg_size, c_frag, a_frag, b_frag, c_frag);
        
      }
  }

  item_ct1.barrier(sycl::access::fence_space::local_space);
  //if(threadIdx.x == 0)
  //{
  //  printnonzero<T>(smem_A, 8*16 + (2*16*(batch_size_warps-1)), "A: ");
  //  printnonzero<T>(smem_B, 2*batch_size_warps*16*32 + (2*16*(batch_size_warps-1)), "B: ");
  //}
  if(warp_id != (WARPS-1)){ return; }
  // only warp_id == (WARPS-1) from here
  int warp_lane = item_ct1.get_local_id(2) % 32;

  ticktock = ticktock == 0 ? 1 : 0;
  for(int k = 0; k < batch_size_warps; k++)
  {
    //if(warp_lane == 0)
      //printf("%i %i %i %i\n", (ticktock*batch_size_warps + k)*a_tile_offset, k, ticktock, threadIdx.x);
    
    dacc_A[(ticktock*batch_size_warps + k)*a_tile_offset] = smem_A[(ticktock*batch_size_warps + k)*a_tile_offset];
    dacc_B[(ticktock*batch_size_warps + k)*b_tile_offset] = smem_B[(ticktock*batch_size_warps + k)*b_tile_offset];
    d_A = dacc_A.template get_multi_ptr<sycl::access::decorated::yes>();
    d_B = dacc_B.get_multi_ptr<sycl::access::decorated::yes>();
    
    //wmma::load_matrix_sync(a_frag, &(smem_A[(ticktock*batch_size_warps + k)*a_tile_offset]), 16); //  111 mu
    sycl::ext::oneapi::experimental::matrix::joint_matrix_load(sg_size, a_frag, d_A, 16);

    //wmma::load_matrix_sync(b_frag, &(smem_B[(ticktock*batch_size_warps + k)*b_tile_offset]), 16); // 35 mu
    sycl::ext::oneapi::experimental::matrix::joint_matrix_load(sg_size, b_frag, d_B, 16);

    
    //wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    sycl::ext::oneapi::experimental::matrix::joint_matrix_mad(sg_size, c_frag, a_frag, b_frag, c_frag);
    
  }

  // 129 mu
  if(warp_id == (WARPS-1))
    
    //wmma::store_matrix_sync(&(smem_C[0]), c_frag, 32, wmma::mem_row_major);
    sycl::ext::oneapi::experimental::matrix::joint_matrix_store(sg_size, c_frag, d_A, 32, sycl::ext::oneapi::experimental::matrix::layout::row_major);

  //printnonzero<T>(smem_C, 32, "");  
  
  if(col_offset + warp_lane < M)
    // use smem_A itself
    dacc_out[col_offset + warp_lane] = dacc_A[warp_lane];
#endif
}


//=========================================4 bit gemm naive===============


#define num_values_4bit 32

template <typename T, int THREADS, int BITS> SYCL_EXTERNAL void kgemm_4bit_inference_naive(int M, int N, int K, T * __restrict__ const A, unsigned char *B,  float *absmax, const float *datatype, T * out,  int lda, int ldb, int ldc, int blocksize, const sycl::nd_item<3> &item_ct1, T *quant_map, const sycl::accessor<T, 1> &dacc_A, const sycl_dacc_uc &dacc_B, const sycl::accessor<T, 1> &dacc_out, const sycl_dacc_float &dacc_absmax, const sycl_dacc_float &dacc_datatype)
{

  // per threadblock:
  // load step-by-step in chunks of [32,warps]: 1x32 * [32,warps] -> [1,warps]
  // 4 warps -> 4 loads per iter
  // 1x32 * 32x4 -> 1x4 outputs per thread block

  const int warp_idx = item_ct1.get_local_id(2) / 32;
  const int warp_lane = item_ct1.get_local_id(2) % 32;
  const int row_B = (THREADS/32)*item_ct1.get_group(2) + warp_idx;
  const int num_values_8bit = num_values_4bit/2;
  float local_C = 0.0f;

  unsigned char local_B_4bit[num_values_8bit];
  T local_B[num_values_4bit/4];
  T local_A[num_values_4bit/4];
  
	T local_absmax = T(0.0f);

  for(int i = item_ct1.get_local_id(2); i < 16; i++)
    quant_map[i] = T(dacc_datatype[i]);
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // A: [1, K]
  // B: [N, K]
  for(int inner_idx = warp_lane*num_values_4bit; inner_idx < K; inner_idx += 32*num_values_4bit)
  {
    int inner_idx_halved = inner_idx/2;
    int offset_B = ldb*row_B;
    int absidx = ((2*offset_B)+inner_idx)/blocksize;
	  
	  local_absmax =  dacc_absmax[absidx];

    if(row_B < M)
    {
      if((inner_idx_halved + num_values_8bit) < (K/2))
      {
        // this is the most important for performance considerations
        reinterpret_cast<sycl::int4(&)[num_values_8bit]>(local_B_4bit)[0] = reinterpret_cast<sycl::int4*>(B)[(offset_B+(inner_idx_halved))/(num_values_8bit)];
      }
      else
      {
        #pragma unroll
        for(int j = 0; j < (num_values_8bit); j++)
          if((inner_idx_halved) + j < (K/2))
            local_B_4bit[j] = dacc_B[offset_B+inner_idx_halved + j];
          else
            local_B_4bit[j] = 0b01110111;
      }
    }
    else
    {
      #pragma unroll
      for(int j = 0; j < (num_values_8bit); j++)
          local_B_4bit[j] = 0b01110111;
    }

    for(int i = 0; i < 4; i++)
    {
      #pragma unroll
      for(int k = 0; k < num_values_8bit/4; k++)
      {
        #if DPCT_COMPATIBILITY_TEMP >= 800
          local_B[k*2] = quant_map[local_B_4bit[(i*num_values_8bit/4) + k] >> 4]*local_absmax;
          local_B[k*2 + 1] = quant_map[local_B_4bit[(i*num_values_8bit/4) + k] & 0x0F]*local_absmax;
        #else
          // bf16 multipliation not supported
          local_B[k*2] = T((float)quant_map[local_B_4bit[(i*num_values_8bit/4) + k] >> 4]*(float)local_absmax);
          local_B[k*2 + 1] = T((float)quant_map[local_B_4bit[(i*num_values_8bit/4) + k] & 0x0F]*(float)local_absmax);
        #endif
      }

      if(inner_idx+(num_values_4bit/4) + (i*num_values_4bit/4) < K)
      {
        // this is also relatively important for performance
        if(BITS==16)
        {
          reinterpret_cast<sycl::int4(&)[num_values_4bit]>(local_A)[0] = reinterpret_cast<sycl::int4*>(A)[inner_idx/(num_values_4bit/4) + i];
        }
        else
        {
          reinterpret_cast<sycl::int4(&)[num_values_4bit]>(local_A)[0] = reinterpret_cast<sycl::int4*>(A)[inner_idx/(num_values_4bit/8) + (2*i) + 0];
          reinterpret_cast<sycl::int4(&)[num_values_4bit]>(local_A)[1] = reinterpret_cast<sycl::int4*>(A)[inner_idx/(num_values_4bit/8) + (2*i) + 1];
        }

      }
      else
        #pragma unroll
        for(int k = 0; k < num_values_4bit/4; k++)
          if(inner_idx + (i*num_values_4bit/4) + k < K)
            local_A[k] = dacc_A[inner_idx + k + (i*num_values_4bit/4)];
          else
            local_A[k] = T(0.0f);


      // accumulate in float; small performance hit for Ampere, but lower error for outputs
      #pragma unroll
      for(int k = 0; k < num_values_4bit/4; k++)
      {
        #if DPCT_COMPATIBILITY_TEMP >= 800
          local_C += (float)(local_A[k]*local_B[k]);
        #else
          // bf16 multipliation not supported
          local_C += ((float)local_A[k]*(float)local_B[k]);
        #endif
      }
    }
  }

  local_C = sycl::reduce_over_group(item_ct1.get_sub_group(), local_C, sycl::plus<>());

  if(row_B < M && warp_lane == 0)
    dacc_out[row_B] = T(local_C);

}


//=================================spm


//============================================k spmm sparse coo===============================================
#define DENORM 1.0f/127.0f
#define MAX_SPARSE_COUNT 32
#define SMEM_SIZE 8*256
template <typename T, int SPMM_ITEMS, int BITS>

SYCL_EXTERNAL void kspmm_coo_very_sparse_naive(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, sycl::half *values, T *B, sycl::half *out, float * __restrict__ const dequant_stats, int nnz, int rowsA, int rowsB, int colsB, const sycl::nd_item<3> &item_ct1,
 sycl::half *smem_dequant_stats, const sycl_dacc &dacc_max_count, const sycl_dacc &dacc_max_idx, const sycl_dacc &dacc_offset_rowidx, const sycl_dacc &dacc_rowidx, const sycl_dacc &dacc_colidx, const sycl::accessor<sycl::half, 1> &dacc_values, const sycl::accessor<T, 1> &dacc_B, const sycl::accessor<sycl::half, 1> &dacc_out, const sycl_dacc_float &dacc_dequant_stats)
{

  // 0. load balancing: We process rows with most columns first (count_vec)and we process one row per block
  //    If a block finishes, the next one is scheduled. Since the last blocks like have fewer
  //    elements they finish faster "fillin up" the gaps left by larger blocks

  // without tensor cores
  // 1. use rowidx_length to find what to load (as many blocks as there are rows)
  // 2. Load A into registers
  // 3. each warp loads all required rows of B but each warp is offset by k
  // 4. Do mma operations that accumulate into registers
  // 5. Each warp stores its output row into matrix C

  
  const int count = dacc_max_count[item_ct1.get_group(2)];
  const int local_max_idx = dacc_max_idx[item_ct1.get_group(2)];
  const int offset = local_max_idx == 0 ? 0 : dacc_offset_rowidx[local_max_idx-1];
  const int local_row_idx = dacc_rowidx[offset];

  const int warp_id = item_ct1.get_local_id(2) / 32;
  const int warp_idx = item_ct1.get_local_id(2) % 32;
  const int warp_offset = (warp_id*32)*SPMM_ITEMS;
  const int num_items = BITS == 8 ? 8 : 8;
  int idx_col_B = warp_offset;
  int local_idx_col_B_offset = 0;

  sycl::half local_valA[MAX_SPARSE_COUNT];
  int local_colidxA[MAX_SPARSE_COUNT];
  sycl::half local_valC[SPMM_ITEMS];
  T local_valsB[num_items];
  sycl::half local_valOut[num_items];
  // 128 byte loads per warp == 4 bytes per thread

  // 2. Load A into registers
  for(int j = 0; j < MAX_SPARSE_COUNT; j++)
  {
    local_valA[j] = j < count ? dacc_values[offset+j] : sycl::vec<float, 1>(0.0f).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    local_colidxA[j] = j < count ? dacc_colidx[offset+j] : 0;
  }

  // each thread processes SPMM_ITEMS=32 per iteration. We have 256 threads. 32*256=x192
  // we expect each warp to be SPMM_ITEMS*32 apart
  // we have a total of 128 bytes for the bank with a bank size of 4 bytes
  // added 3 bytes = 6 values between warps should reduce bank conflicts
  


  while(idx_col_B <  colsB)
  {

    if(dequant_stats != NULL)
    {
      for(int i = item_ct1.get_local_id(2); i < SMEM_SIZE; i+=item_ct1.get_local_range(2))
        if((idx_col_B+i-local_idx_col_B_offset) < colsB)
          smem_dequant_stats[i] = dacc_dequant_stats[idx_col_B+i-local_idx_col_B_offset];

      /*
      DPCT1065:204: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    #pragma unroll SPMM_ITEMS
    for(int j = 0; j < SPMM_ITEMS; j++)
      local_valC[j] = 0.0f;

    #pragma unroll
    for(int i = 0; i < count; i++)
    {
        // 3. each warp loads all required rows of B but each warp is offset by k
        int row_offset = colsB*local_colidxA[i];

        #pragma unroll SPMM_ITEMS
        for(int j = 0; j < SPMM_ITEMS; j+=num_items)
        {
          // 4. Multiply the tile -> accumulate outputs in shared memory until 128 bytes it reached
          int idx = idx_col_B + (warp_idx*SPMM_ITEMS) + j;
          if(idx >= colsB){ break; }
          if((idx+num_items < colsB))
          {
            if(BITS == 8)
              reinterpret_cast<sycl::float2(&)[num_items]>(local_valsB)[0] = reinterpret_cast<sycl::float2*>(B)[(row_offset+ idx)/num_items];
            else
              reinterpret_cast<sycl::float4(&)[num_items]>(local_valsB)[0] = reinterpret_cast<sycl::float4*>(B)[(row_offset+ idx)/num_items];
          }
          else
          {
            #pragma unroll num_items
            for(int k = 0; k < num_items; k++)
              if(idx+k < colsB)
                local_valsB[k] = dacc_B[row_offset+idx+k];
              else
                local_valsB[k] = 0.0f;
          }
          #pragma unroll num_items
          for(int k = 0; k < num_items; k++)
          {
            if(BITS == 8 && dequant_stats != NULL)
              // we do texture cache reads (__ldg) on dequant_stats which should be super fast
            {
              float valB = local_valsB[k];
              float valA = local_valA[i];
              if(valB != 0.0 && valA != 0.0)
                local_valC[j+k] = (float)local_valC[j+k] + ((float)smem_dequant_stats[idx+k-local_idx_col_B_offset])*DENORM*valB*valA;
            }
            else
              local_valC[j+k] = (float)local_valC[j+k] + (float)local_valsB[k]*(float)local_valA[i];
          }
        }
    }

    int idx_row_C = (colsB*local_row_idx);

    #pragma unroll SPMM_ITEMS
    for(int j = 0; j < SPMM_ITEMS; j+=num_items)
    {
      //int idx_col_C =  idx_col_B + (32*j) + warp_idx;
      int idx_col_C =  idx_col_B + warp_idx*SPMM_ITEMS + j;
      int idx_val = idx_col_C + idx_row_C;

      if(idx_col_C +num_items < colsB)
      {

          // load outputs to do inplace addition
          reinterpret_cast<sycl::float4(&)[num_items/4]>(local_valOut)[0] = reinterpret_cast<sycl::float4*>(out)[idx_val/num_items];

          #pragma unroll num_items
          for(int k = 0; k < num_items; k++)
            local_valC[(j/num_items) + k] = (float)local_valC[(j/num_items) + k] + (float)local_valOut[k];

          reinterpret_cast<sycl::float4*>(out)[idx_val/num_items] = reinterpret_cast<sycl::float4(&)[num_items]>(local_valC)[j/num_items];
      }
      else
      {
        #pragma unroll num_items
        for(int k = 0; k < num_items; k++)
         if(idx_col_C + k < colsB)
           dacc_out[idx_val+k] = (float)out[idx_val+k]+(float)local_valC[j+k];
      }
    }

    idx_col_B += item_ct1.get_local_range(2)*SPMM_ITEMS;
    local_idx_col_B_offset += item_ct1.get_local_range(2)*SPMM_ITEMS;
  }
}

//Template declarations

//these are not used and make no sense, but the compiler needs them
//template __global__ void gemm_device<float, 16, 128>(int M, int N, int K, float * __restrict__ const A,  float* B,  float * out,  int lda, int ldb, int ldc);
template SYCL_EXTERNAL void gemm_device<sycl::half, 32, 256>(int M, int N, int K, sycl::half * __restrict__ const A,  sycl::half* B,  sycl::half * out,  int lda, int ldb, int ldc,
                                         const sycl::nd_item<3> &item_ct1,
                                         sycl::half *smem_A, sycl::half *smem_B,
                                         const sycl::accessor<sycl::half, 1> &dacc_A, 
                                         const sycl::accessor<sycl::half, 1> &dacc_B, 
                                         const sycl::accessor<sycl::half, 1> &dacc_out);
                                         
template SYCL_EXTERNAL void gemm_device<sycl::half, 32, 192>(int M, int N, int K, sycl::half * __restrict__ const A,  sycl::half* B,  sycl::half * out,  int lda, int ldb, int ldc,
                                         const sycl::nd_item<3> &item_ct1,
                                         sycl::half *smem_A, sycl::half *smem_B,
                                         const sycl::accessor<sycl::half, 1> &dacc_A, 
                                         const sycl::accessor<sycl::half, 1> &dacc_B, 
                                         const sycl::accessor<sycl::half, 1> &dacc_out);
                                         
template SYCL_EXTERNAL void gemm_device<sycl::half, 32, 160>(int M, int N, int K, sycl::half * __restrict__ const A,  sycl::half* B,  sycl::half * out,  int lda, int ldb, int ldc,
                                         const sycl::nd_item<3> &item_ct1,
                                         sycl::half *smem_A, sycl::half *smem_B,
                                         const sycl::accessor<sycl::half, 1> &dacc_A, 
                                         const sycl::accessor<sycl::half, 1> &dacc_B, 
                                         const sycl::accessor<sycl::half, 1> &dacc_out);
template SYCL_EXTERNAL void gemm_device<sycl::half, 32, 128>(int M, int N, int K, sycl::half * __restrict__ const A,  sycl::half* B,  sycl::half * out,  int lda, int ldb, int ldc,
                                         const sycl::nd_item<3> &item_ct1,
                                         sycl::half *smem_A, sycl::half *smem_B,
                                         const sycl::accessor<sycl::half, 1> &dacc_A, 
                                         const sycl::accessor<sycl::half, 1> &dacc_B, 
                                         const sycl::accessor<sycl::half, 1> &dacc_out);
                                         
//template __global__ void gemm_device<float, 16, 32>(int M, int N, int K, float * __restrict__ const A,  float* B,  float * out,  int lda, int ldb, int ldc);
template SYCL_EXTERNAL void gemm_device<sycl::half, 32, 32>(int M, int N, int K, sycl::half * __restrict__ const A,  sycl::half* B,  sycl::half * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        sycl::half *smem_A, sycl::half *smem_B,
                                        const sycl::accessor<sycl::half, 1> &dacc_A, 
                                         const sycl::accessor<sycl::half, 1> &dacc_B, 
                                         const sycl::accessor<sycl::half, 1> &dacc_out);
                                         
template SYCL_EXTERNAL void gemm_device<sycl::half, 32, 64>(int M, int N, int K, sycl::half * __restrict__ const A,  sycl::half* B,  sycl::half * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        sycl::half *smem_A, sycl::half *smem_B,
                                        const sycl::accessor<sycl::half, 1> &dacc_A, 
                                         const sycl::accessor<sycl::half, 1> &dacc_B, 
                                         const sycl::accessor<sycl::half, 1> &dacc_out);
                                         
template SYCL_EXTERNAL void gemm_device<sycl::half, 32, 96>(int M, int N, int K, sycl::half * __restrict__ const A,  sycl::half* B,  sycl::half * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        sycl::half *smem_A, sycl::half *smem_B,
                                        const sycl::accessor<sycl::half, 1> &dacc_A, 
                                         const sycl::accessor<sycl::half, 1> &dacc_B, 
                                         const sycl::accessor<sycl::half, 1> &dacc_out);

////

template SYCL_EXTERNAL void gemm_device<float, 32, 256>(int M, int N, int K, float * __restrict__ const A,  float* B,  float * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        float *smem_A, float *smem_B,
                                        const sycl::accessor<float, 1> &dacc_A, 
                                         const sycl::accessor<float, 1> &dacc_B, 
                                         const sycl::accessor<float, 1> &dacc_out);
                                         

template SYCL_EXTERNAL void gemm_device<float, 32, 192>(int M, int N, int K, float * __restrict__ const A,  float* B,  float * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        float *smem_A, float *smem_B,
                                        const sycl::accessor<float, 1> &dacc_A, 
                                         const sycl::accessor<float, 1> &dacc_B, 
                                         const sycl::accessor<float, 1> &dacc_out);
                                         

template SYCL_EXTERNAL void gemm_device<float, 32, 160>(int M, int N, int K, float * __restrict__ const A,  float* B,  float * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        float *smem_A, float *smem_B,
                                        const sycl::accessor<float, 1> &dacc_A, 
                                         const sycl::accessor<float, 1> &dacc_B, 
                                         const sycl::accessor<float, 1> &dacc_out);

template SYCL_EXTERNAL void gemm_device<float, 32, 128>(int M, int N, int K, float * __restrict__ const A,  float* B,  float * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        float *smem_A, float *smem_B,
                                        const sycl::accessor<float, 1> &dacc_A, 
                                         const sycl::accessor<float, 1> &dacc_B, 
                                         const sycl::accessor<float, 1> &dacc_out);
                                         
//template __global__ void gemm_device<float, 16, 32>(int M, int N, int K, float * __restrict__ const A,  float* B,  float * out,  int lda, int ldb, int ldc);

template SYCL_EXTERNAL void gemm_device<float, 32, 32>(int M, int N, int K, float * __restrict__ const A,  float* B,  float * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        float *smem_A, float *smem_B,
                                        const sycl::accessor<float, 1> &dacc_A, 
                                         const sycl::accessor<float, 1> &dacc_B, 
                                         const sycl::accessor<float, 1> &dacc_out);                                         

template SYCL_EXTERNAL void gemm_device<float, 32, 64>(int M, int N, int K, float * __restrict__ const A,  float* B,  float * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        float *smem_A, float *smem_B,
                                        const sycl::accessor<float, 1> &dacc_A, 
                                         const sycl::accessor<float, 1> &dacc_B, 
                                         const sycl::accessor<float, 1> &dacc_out);                                         
template SYCL_EXTERNAL void gemm_device<float, 32, 96>(int M, int N, int K, float * __restrict__ const A,  float* B,  float * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        float *smem_A, float *smem_B,
                                        const sycl::accessor<float, 1> &dacc_A, 
                                         const sycl::accessor<float, 1> &dacc_B, 
                                         const sycl::accessor<float, 1> &dacc_out);

///
template SYCL_EXTERNAL void gemm_device<sycl::ext::oneapi::bfloat16, 32, 256>(int M, int N, int K, sycl::ext::oneapi::bfloat16 * __restrict__ const A,  sycl::ext::oneapi::bfloat16* B,  sycl::ext::oneapi::bfloat16 * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        sycl::ext::oneapi::bfloat16 *smem_A, sycl::ext::oneapi::bfloat16 *smem_B,
                                        const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_A, 
                                         const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_B, 
                                         const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_out);
                                         

template SYCL_EXTERNAL void gemm_device<sycl::ext::oneapi::bfloat16, 32, 192>(int M, int N, int K, sycl::ext::oneapi::bfloat16 * __restrict__ const A,  sycl::ext::oneapi::bfloat16* B,  sycl::ext::oneapi::bfloat16 * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        sycl::ext::oneapi::bfloat16 *smem_A, sycl::ext::oneapi::bfloat16 *smem_B,
                                        const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_A, 
                                         const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_B, 
                                         const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_out);
                                         

template SYCL_EXTERNAL void gemm_device<sycl::ext::oneapi::bfloat16, 32, 160>(int M, int N, int K, sycl::ext::oneapi::bfloat16 * __restrict__ const A,  sycl::ext::oneapi::bfloat16* B,  sycl::ext::oneapi::bfloat16 * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        sycl::ext::oneapi::bfloat16 *smem_A, sycl::ext::oneapi::bfloat16 *smem_B,
                                        const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_A, 
                                         const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_B, 
                                         const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_out);

template SYCL_EXTERNAL void gemm_device<sycl::ext::oneapi::bfloat16, 32, 128>(int M, int N, int K, sycl::ext::oneapi::bfloat16 * __restrict__ const A,  sycl::ext::oneapi::bfloat16* B,  sycl::ext::oneapi::bfloat16 * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        sycl::ext::oneapi::bfloat16 *smem_A, sycl::ext::oneapi::bfloat16 *smem_B,
                                        const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_A, 
                                         const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_B, 
                                         const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_out);
                                         
//template __global__ void gemm_device<float, 16, 32>(int M, int N, int K, float * __restrict__ const A,  float* B,  float * out,  int lda, int ldb, int ldc);

template SYCL_EXTERNAL void gemm_device<sycl::ext::oneapi::bfloat16, 32, 32>(int M, int N, int K, sycl::ext::oneapi::bfloat16 * __restrict__ const A,  sycl::ext::oneapi::bfloat16* B,  sycl::ext::oneapi::bfloat16 * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        sycl::ext::oneapi::bfloat16 *smem_A, sycl::ext::oneapi::bfloat16 *smem_B,
                                        const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_A, 
                                         const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_B, 
                                         const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_out);
template SYCL_EXTERNAL void gemm_device<sycl::ext::oneapi::bfloat16, 32, 64>(int M, int N, int K, sycl::ext::oneapi::bfloat16 * __restrict__ const A,  sycl::ext::oneapi::bfloat16* B,  sycl::ext::oneapi::bfloat16 * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        sycl::ext::oneapi::bfloat16 *smem_A, sycl::ext::oneapi::bfloat16 *smem_B,
                                        const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_A, 
                                         const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_B, 
                                         const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_out);


template SYCL_EXTERNAL void gemm_device<sycl::ext::oneapi::bfloat16, 32, 96>(int M, int N, int K, sycl::ext::oneapi::bfloat16 * __restrict__ const A,  sycl::ext::oneapi::bfloat16* B,  sycl::ext::oneapi::bfloat16 * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        sycl::ext::oneapi::bfloat16 *smem_A, sycl::ext::oneapi::bfloat16 *smem_B,
                                        const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_A, 
                                         const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_B, 
                                         const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_out);

///

// these are not used and make no sense, but the compiler needs them

//template __global__ void gemm_device<float, 32, 128>(int M, int N, int K, float * __restrict__ const A,  float* B,  float * out,  int lda, int ldb, int ldc);
template SYCL_EXTERNAL void gemm_device<sycl::half, 16, 256>(int M, int N, int K, sycl::half * __restrict__ const A,  sycl::half* B,  sycl::half * out,  int lda, int ldb, int ldc,
                                         const sycl::nd_item<3> &item_ct1,
                                         sycl::half *smem_A, sycl::half *smem_B,
                                         const sycl::accessor<sycl::half, 1> &dacc_A, 
                                         const sycl::accessor<sycl::half, 1> &dacc_B, 
                                         const sycl::accessor<sycl::half, 1> &dacc_out);
                                         
template SYCL_EXTERNAL void gemm_device<sycl::half, 16, 192>(int M, int N, int K, sycl::half * __restrict__ const A,  sycl::half* B,  sycl::half * out,  int lda, int ldb, int ldc,
                                         const sycl::nd_item<3> &item_ct1,
                                         sycl::half *smem_A, sycl::half *smem_B,
                                         const sycl::accessor<sycl::half, 1> &dacc_A, 
                                         const sycl::accessor<sycl::half, 1> &dacc_B, 
                                         const sycl::accessor<sycl::half, 1> &dacc_out);
template SYCL_EXTERNAL void gemm_device<sycl::half, 16, 160>(int M, int N, int K, sycl::half * __restrict__ const A,  sycl::half* B,  sycl::half * out,  int lda, int ldb, int ldc,
                                         const sycl::nd_item<3> &item_ct1,
                                         sycl::half *smem_A, sycl::half *smem_B,
                                         const sycl::accessor<sycl::half, 1> &dacc_A, 
                                         const sycl::accessor<sycl::half, 1> &dacc_B, 
                                         const sycl::accessor<sycl::half, 1> &dacc_out);;
template SYCL_EXTERNAL void gemm_device<sycl::half, 16, 128>(int M, int N, int K, sycl::half * __restrict__ const A,  sycl::half* B,  sycl::half * out,  int lda, int ldb, int ldc,
                                         const sycl::nd_item<3> &item_ct1,
                                         sycl::half *smem_A, sycl::half *smem_B,
                                         const sycl::accessor<sycl::half, 1> &dacc_A, 
                                         const sycl::accessor<sycl::half, 1> &dacc_B, 
                                         const sycl::accessor<sycl::half, 1> &dacc_out);
//template __global__ void gemm_device<float, 32, 32>(int M, int N, int K, float * __restrict__ const A,  float* B,  float * out,  int lda, int ldb, int ldc);
template SYCL_EXTERNAL void gemm_device<sycl::half, 16, 32>(int M, int N, int K, sycl::half * __restrict__ const A,  sycl::half* B,  sycl::half * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        sycl::half *smem_A, sycl::half *smem_B,
                                        const sycl::accessor<sycl::half, 1> &dacc_A, 
                                         const sycl::accessor<sycl::half, 1> &dacc_B, 
                                         const sycl::accessor<sycl::half, 1> &dacc_out);
                                        
template SYCL_EXTERNAL void gemm_device<sycl::half, 16, 64>(int M, int N, int K, sycl::half * __restrict__ const A,  sycl::half* B,  sycl::half * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        sycl::half *smem_A, sycl::half *smem_B,
                                        const sycl::accessor<sycl::half, 1> &dacc_A, 
                                         const sycl::accessor<sycl::half, 1> &dacc_B, 
                                         const sycl::accessor<sycl::half, 1> &dacc_out);
                                        
template SYCL_EXTERNAL void gemm_device<sycl::half, 16, 96>(int M, int N, int K, sycl::half * __restrict__ const A,  sycl::half* B,  sycl::half * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        sycl::half *smem_A, sycl::half *smem_B,
                                        const sycl::accessor<sycl::half, 1> &dacc_A, 
                                         const sycl::accessor<sycl::half, 1> &dacc_B, 
                                         const sycl::accessor<sycl::half, 1> &dacc_out);



/////


template SYCL_EXTERNAL void gemm_device<float, 16, 256>(int M, int N, int K, float * __restrict__ const A,  float* B,  float * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        float *smem_A, float *smem_B,
                                        const sycl::accessor<float, 1> &dacc_A, 
                                         const sycl::accessor<float, 1> &dacc_B, 
                                         const sycl::accessor<float, 1> &dacc_out);
                                         

template SYCL_EXTERNAL void gemm_device<float, 16, 192>(int M, int N, int K, float * __restrict__ const A,  float* B,  float * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        float *smem_A, float *smem_B,
                                        const sycl::accessor<float, 1> &dacc_A, 
                                         const sycl::accessor<float, 1> &dacc_B, 
                                         const sycl::accessor<float, 1> &dacc_out);
                                         

template SYCL_EXTERNAL void gemm_device<float, 16, 160>(int M, int N, int K, float * __restrict__ const A,  float* B,  float * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        float *smem_A, float *smem_B,
                                        const sycl::accessor<float, 1> &dacc_A, 
                                         const sycl::accessor<float, 1> &dacc_B, 
                                         const sycl::accessor<float, 1> &dacc_out);

template SYCL_EXTERNAL void gemm_device<float, 16, 128>(int M, int N, int K, float * __restrict__ const A,  float* B,  float * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        float *smem_A, float *smem_B,
                                        const sycl::accessor<float, 1> &dacc_A, 
                                         const sycl::accessor<float, 1> &dacc_B, 
                                         const sycl::accessor<float, 1> &dacc_out);
                                         
//template __global__ void gemm_device<float, 16, 32>(int M, int N, int K, float * __restrict__ const A,  float* B,  float * out,  int lda, int ldb, int ldc);

template SYCL_EXTERNAL void gemm_device<float, 16, 32>(int M, int N, int K, float * __restrict__ const A,  float* B,  float * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        float *smem_A, float *smem_B,
                                        const sycl::accessor<float, 1> &dacc_A, 
                                         const sycl::accessor<float, 1> &dacc_B, 
                                         const sycl::accessor<float, 1> &dacc_out);                                         

template SYCL_EXTERNAL void gemm_device<float, 16, 64>(int M, int N, int K, float * __restrict__ const A,  float* B,  float * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        float *smem_A, float *smem_B,
                                        const sycl::accessor<float, 1> &dacc_A, 
                                         const sycl::accessor<float, 1> &dacc_B, 
                                         const sycl::accessor<float, 1> &dacc_out);                                         
template SYCL_EXTERNAL void gemm_device<float, 16, 96>(int M, int N, int K, float * __restrict__ const A,  float* B,  float * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        float *smem_A, float *smem_B,
                                        const sycl::accessor<float, 1> &dacc_A, 
                                         const sycl::accessor<float, 1> &dacc_B, 
                                         const sycl::accessor<float, 1> &dacc_out);

///
template SYCL_EXTERNAL void gemm_device<sycl::ext::oneapi::bfloat16, 16, 256>(int M, int N, int K, sycl::ext::oneapi::bfloat16 * __restrict__ const A,  sycl::ext::oneapi::bfloat16* B,  sycl::ext::oneapi::bfloat16 * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        sycl::ext::oneapi::bfloat16 *smem_A, sycl::ext::oneapi::bfloat16 *smem_B,
                                        const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_A, 
                                         const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_B, 
                                         const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_out);
                                         

template SYCL_EXTERNAL void gemm_device<sycl::ext::oneapi::bfloat16, 16, 192>(int M, int N, int K, sycl::ext::oneapi::bfloat16 * __restrict__ const A,  sycl::ext::oneapi::bfloat16* B,  sycl::ext::oneapi::bfloat16 * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        sycl::ext::oneapi::bfloat16 *smem_A, sycl::ext::oneapi::bfloat16 *smem_B,
                                        const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_A, 
                                         const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_B, 
                                         const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_out);
                                         

template SYCL_EXTERNAL void gemm_device<sycl::ext::oneapi::bfloat16, 16, 160>(int M, int N, int K, sycl::ext::oneapi::bfloat16 * __restrict__ const A,  sycl::ext::oneapi::bfloat16* B,  sycl::ext::oneapi::bfloat16 * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        sycl::ext::oneapi::bfloat16 *smem_A, sycl::ext::oneapi::bfloat16 *smem_B,
                                        const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_A, 
                                         const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_B, 
                                         const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_out);

template SYCL_EXTERNAL void gemm_device<sycl::ext::oneapi::bfloat16, 16, 128>(int M, int N, int K, sycl::ext::oneapi::bfloat16 * __restrict__ const A,  sycl::ext::oneapi::bfloat16* B,  sycl::ext::oneapi::bfloat16 * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        sycl::ext::oneapi::bfloat16 *smem_A, sycl::ext::oneapi::bfloat16 *smem_B,
                                        const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_A, 
                                         const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_B, 
                                         const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_out);
                                         
//template __global__ void gemm_device<float, 16, 32>(int M, int N, int K, float * __restrict__ const A,  float* B,  float * out,  int lda, int ldb, int ldc);

template SYCL_EXTERNAL void gemm_device<sycl::ext::oneapi::bfloat16, 16, 32>(int M, int N, int K, sycl::ext::oneapi::bfloat16 * __restrict__ const A,  sycl::ext::oneapi::bfloat16* B,  sycl::ext::oneapi::bfloat16 * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        sycl::ext::oneapi::bfloat16 *smem_A, sycl::ext::oneapi::bfloat16 *smem_B,
                                        const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_A, 
                                         const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_B, 
                                         const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_out);
template SYCL_EXTERNAL void gemm_device<sycl::ext::oneapi::bfloat16, 16, 64>(int M, int N, int K, sycl::ext::oneapi::bfloat16 * __restrict__ const A,  sycl::ext::oneapi::bfloat16* B,  sycl::ext::oneapi::bfloat16 * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        sycl::ext::oneapi::bfloat16 *smem_A, sycl::ext::oneapi::bfloat16 *smem_B,
                                        const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_A, 
                                         const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_B, 
                                         const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_out);


template SYCL_EXTERNAL void gemm_device<sycl::ext::oneapi::bfloat16, 16, 96>(int M, int N, int K, sycl::ext::oneapi::bfloat16 * __restrict__ const A,  sycl::ext::oneapi::bfloat16* B,  sycl::ext::oneapi::bfloat16 * out,  int lda, int ldb, int ldc,
                                        const sycl::nd_item<3> &item_ct1,
                                        sycl::ext::oneapi::bfloat16 *smem_A, sycl::ext::oneapi::bfloat16 *smem_B,
                                        const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_A, 
                                         const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_B, 
                                         const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_out);




/////
///
template SYCL_EXTERNAL void kgemm_4bit_inference<sycl::half, 96>(int M, int N, int K, sycl::half * __restrict__ const A, unsigned char *B,  float *absmax, sycl::half * out,  int lda, int ldb, int ldc, int blocksize,
                                             const sycl::nd_item<3> &item_ct1,
                                             sycl::half *smem_A,
                                             unsigned char *smem_B,
                                             sycl::half *smem_C,
                                             const sycl::accessor<sycl::half, 1> &dacc_A,
                                             const sycl_dacc_uc &dacc_B, 
                                             const sycl::accessor<sycl::half, 1> &dacc_out);
                                            
template SYCL_EXTERNAL void kgemm_4bit_inference<sycl::half, 128>(int M, int N, int K, sycl::half * __restrict__ const A, unsigned char *B,  float *absmax, sycl::half * out,  int lda, int ldb, int ldc, int blocksize,
                                              const sycl::nd_item<3> &item_ct1,
                                              sycl::half *smem_A,
                                              unsigned char *smem_B,
                                              sycl::half *smem_C,
                                              const sycl::accessor<sycl::half, 1> &dacc_A,
                                              const sycl_dacc_uc &dacc_B, 
                                              const sycl::accessor<sycl::half, 1> &dacc_out);
                                              
template SYCL_EXTERNAL void kgemm_4bit_inference<sycl::half, 160>(int M, int N, int K, sycl::half * __restrict__ const A, unsigned char *B,  float *absmax, sycl::half * out,  int lda, int ldb, int ldc, int blocksize,
                                              const sycl::nd_item<3> &item_ct1,
                                              sycl::half *smem_A,
                                              unsigned char *smem_B,
                                              sycl::half *smem_C,
                                              const sycl::accessor<sycl::half, 1> &dacc_A,
                                              const sycl_dacc_uc &dacc_B, 
                                              const sycl::accessor<sycl::half, 1> &dacc_out);
                                              
template SYCL_EXTERNAL void kgemm_4bit_inference<sycl::half, 256>(int M, int N, int K, sycl::half * __restrict__ const A, unsigned char *B,  float *absmax, sycl::half * out,  int lda, int ldb, int ldc, int blocksize,
                                              const sycl::nd_item<3> &item_ct1,
                                              sycl::half *smem_A,
                                              unsigned char *smem_B,
                                              sycl::half *smem_C,
                                              const sycl::accessor<sycl::half, 1> &dacc_A,
                                              const sycl_dacc_uc &dacc_B, 
                                              const sycl::accessor<sycl::half, 1> &dacc_out);

                                                        
/////
template SYCL_EXTERNAL void kgemm_4bit_inference<float, 96>(int M, int N, int K, float * __restrict__ const A, unsigned char *B,  float *absmax, float * out,  int lda, int ldb, int ldc, int blocksize,
                                             const sycl::nd_item<3> &item_ct1,
                                             float *smem_A,
                                             unsigned char *smem_B,
                                             float *smem_C,
                                             const sycl::accessor<float, 1> &dacc_A,
                                             const sycl_dacc_uc &dacc_B, 
                                             const sycl::accessor<float, 1> &dacc_out);
                                            
template SYCL_EXTERNAL void kgemm_4bit_inference<float, 128>(int M, int N, int K, float * __restrict__ const A, unsigned char *B,  float *absmax, float * out,  int lda, int ldb, int ldc, int blocksize,
                                             const sycl::nd_item<3> &item_ct1,
                                             float *smem_A,
                                             unsigned char *smem_B,
                                             float *smem_C,
                                             const sycl::accessor<float, 1> &dacc_A,
                                             const sycl_dacc_uc &dacc_B, 
                                             const sycl::accessor<float, 1> &dacc_out);

template SYCL_EXTERNAL void kgemm_4bit_inference<float, 160>(int M, int N, int K, float * __restrict__ const A, unsigned char *B,  float *absmax, float * out,  int lda, int ldb, int ldc, int blocksize,
                                             const sycl::nd_item<3> &item_ct1,
                                             float *smem_A,
                                             unsigned char *smem_B,
                                             float *smem_C,
                                             const sycl::accessor<float, 1> &dacc_A,
                                             const sycl_dacc_uc &dacc_B, 
                                             const sycl::accessor<float, 1> &dacc_out);

template SYCL_EXTERNAL void kgemm_4bit_inference<float, 256>(int M, int N, int K, float * __restrict__ const A, unsigned char *B,  float *absmax, float * out,  int lda, int ldb, int ldc, int blocksize,
                                             const sycl::nd_item<3> &item_ct1,
                                             float *smem_A,
                                             unsigned char *smem_B,
                                             float *smem_C,
                                             const sycl::accessor<float, 1> &dacc_A,
                                             const sycl_dacc_uc &dacc_B, 
                                             const sycl::accessor<float, 1> &dacc_out);
                                              
template SYCL_EXTERNAL void kgemm_4bit_inference<sycl::ext::oneapi::bfloat16, 96>(int M, int N, int K, sycl::ext::oneapi::bfloat16 * __restrict__ const A, unsigned char *B,  float *absmax, sycl::ext::oneapi::bfloat16 * out,  int lda, int ldb, int ldc, int blocksize,
                                             const sycl::nd_item<3> &item_ct1,
                                             sycl::ext::oneapi::bfloat16 *smem_A,
                                             unsigned char *smem_B,
                                             sycl::ext::oneapi::bfloat16 *smem_C,
                                             const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_A,
                                             const sycl_dacc_uc &dacc_B, 
                                             const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_out);


////
template SYCL_EXTERNAL void kgemm_4bit_inference_naive<sycl::half, 128, 16>(int M, int N, int K, sycl::half * __restrict__ const A, unsigned char *B,  float *absmax, const float *datatype, sycl::half * out,  int lda, int ldb, int ldc, int blocksize,
                                                       const sycl::nd_item<3> &item_ct1,
                                                       sycl::half *quant_map,
                                                       const sycl::accessor<sycl::half, 1> &dacc_A, 
                                                        const sycl_dacc_uc &dacc_B, 
                                                        const sycl::accessor<sycl::half, 1> &dacc_out,
                                                        const sycl_dacc_float &dacc_absmax,
                                                        const sycl_dacc_float &dacc_datatype);
                                                        
template SYCL_EXTERNAL void kgemm_4bit_inference_naive<float, 128, 32>(int M, int N, int K, float * __restrict__ const A, unsigned char *B,  float *absmax, const float *datatype, float * out,  int lda, int ldb, int ldc, int blocksize,
                                                         const sycl::nd_item<3> &item_ct1,
                                                         float *quant_map,
                                                         const sycl::accessor<float, 1> &dacc_A, 
                                                        const sycl_dacc_uc &dacc_B, 
                                                        const sycl::accessor<float, 1> &dacc_out,
                                                        const sycl_dacc_float &dacc_absmax,
                                                        const sycl_dacc_float &dacc_datatype);
                                                        
template SYCL_EXTERNAL void kgemm_4bit_inference_naive<sycl::ext::oneapi::bfloat16, 128, 16>(int M, int N, int K, sycl::ext::oneapi::bfloat16 * __restrict__ const A, unsigned char *B,  float *absmax, const float *datatype, sycl::ext::oneapi::bfloat16 * out,  int lda, int ldb, int ldc, int blocksize,
                                                       const sycl::nd_item<3> &item_ct1,
                                                       sycl::ext::oneapi::bfloat16 *quant_map,
                                                       const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_A, 
                                                        const sycl_dacc_uc &dacc_B, 
                                                        const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_out,
                                                        const sycl_dacc_float &dacc_absmax,
                                                        const sycl_dacc_float &dacc_datatype);


                                                        
template SYCL_EXTERNAL void kspmm_coo_very_sparse_naive<signed char, 8, 16>(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, sycl::half *values, signed char *B, sycl::half *out, float * __restrict__ const dequant_stats, int nnz, int rowsA, int rowsB, int colsB, const sycl::nd_item<3> &item_ct1,
 sycl::half *smem_dequant_stats, const sycl_dacc &dacc_max_count, const sycl_dacc &dacc_max_idx, const sycl_dacc &dacc_offset_rowidx, const sycl_dacc &dacc_rowidx, const sycl_dacc &dacc_colidx, const sycl::accessor<sycl::half, 1> &dacc_values, const sycl::accessor<signed char, 1> &dacc_B, const sycl::accessor<sycl::half, 1> &dacc_out, const sycl_dacc_float &dacc_dequant_stats);
                                                       
template SYCL_EXTERNAL void kspmm_coo_very_sparse_naive<signed char, 16, 16>(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, sycl::half *values, signed char *B, sycl::half *out, float * __restrict__ const dequant_stats, int nnz, int rowsA, int rowsB, int colsB, const sycl::nd_item<3> &item_ct1,
 sycl::half *smem_dequant_stats, const sycl_dacc &dacc_max_count, const sycl_dacc &dacc_max_idx, const sycl_dacc &dacc_offset_rowidx, const sycl_dacc &dacc_rowidx, const sycl_dacc &dacc_colidx, const sycl::accessor<sycl::half, 1> &dacc_values, const sycl::accessor<signed char, 1> &dacc_B, const sycl::accessor<sycl::half, 1> &dacc_out, const sycl_dacc_float &dacc_dequant_stats);
                                                       

template SYCL_EXTERNAL void kspmm_coo_very_sparse_naive<signed char, 32, 16>(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, sycl::half *values, signed char *B, sycl::half *out, float * __restrict__ const dequant_stats, int nnz, int rowsA, int rowsB, int colsB, const sycl::nd_item<3> &item_ct1,
 sycl::half *smem_dequant_stats, const sycl_dacc &dacc_max_count, const sycl_dacc &dacc_max_idx, const sycl_dacc &dacc_offset_rowidx, const sycl_dacc &dacc_rowidx, const sycl_dacc &dacc_colidx, const sycl::accessor<sycl::half, 1> &dacc_values, const sycl::accessor<signed char, 1> &dacc_B, const sycl::accessor<sycl::half, 1> &dacc_out, const sycl_dacc_float &dacc_dequant_stats);

                                                       
template SYCL_EXTERNAL void kspmm_coo_very_sparse_naive<signed char, 32, 8>(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, sycl::half *values, signed char *B, sycl::half *out, float * __restrict__ const dequant_stats, int nnz, int rowsA, int rowsB, int colsB, const sycl::nd_item<3> &item_ct1,
 sycl::half *smem_dequant_stats, const sycl_dacc &dacc_max_count, const sycl_dacc &dacc_max_idx, const sycl_dacc &dacc_offset_rowidx, const sycl_dacc &dacc_rowidx, const sycl_dacc &dacc_colidx, const sycl::accessor<sycl::half, 1> &dacc_values, const sycl::accessor<signed char, 1> &dacc_B, const sycl::accessor<sycl::half, 1> &dacc_out, const sycl_dacc_float &dacc_dequant_stats);
 
template SYCL_EXTERNAL void kspmm_coo_very_sparse_naive<signed char, 16, 8>(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, sycl::half *values, signed char *B, sycl::half *out, float * __restrict__ const dequant_stats, int nnz, int rowsA, int rowsB, int colsB, const sycl::nd_item<3> &item_ct1,
 sycl::half *smem_dequant_stats, const sycl_dacc &dacc_max_count, const sycl_dacc &dacc_max_idx, const sycl_dacc &dacc_offset_rowidx, const sycl_dacc &dacc_rowidx, const sycl_dacc &dacc_colidx, const sycl::accessor<sycl::half, 1> &dacc_values, const sycl::accessor<signed char, 1> &dacc_B, const sycl::accessor<sycl::half, 1> &dacc_out, const sycl_dacc_float &dacc_dequant_stats);

template SYCL_EXTERNAL void kspmm_coo_very_sparse_naive<signed char, 8, 8>(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, sycl::half *values, signed char *B, sycl::half *out, float * __restrict__ const dequant_stats, int nnz, int rowsA, int rowsB, int colsB, const sycl::nd_item<3> &item_ct1,
 sycl::half *smem_dequant_stats, const sycl_dacc &dacc_max_count, const sycl_dacc &dacc_max_idx, const sycl_dacc &dacc_offset_rowidx, const sycl_dacc &dacc_rowidx, const sycl_dacc &dacc_colidx, const sycl::accessor<sycl::half, 1> &dacc_values, const sycl::accessor<signed char, 1> &dacc_B, const sycl::accessor<sycl::half, 1> &dacc_out, const sycl_dacc_float &dacc_dequant_stats);
 
 

///
template SYCL_EXTERNAL void kspmm_coo_very_sparse_naive<sycl::half, 8, 16>(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, sycl::half *values, sycl::half *B, sycl::half *out, float * __restrict__ const dequant_stats, int nnz, int rowsA, int rowsB, int colsB, const sycl::nd_item<3> &item_ct1,
 sycl::half *smem_dequant_stats, const sycl_dacc &dacc_max_count, const sycl_dacc &dacc_max_idx, const sycl_dacc &dacc_offset_rowidx, const sycl_dacc &dacc_rowidx, const sycl_dacc &dacc_colidx, const sycl::accessor<sycl::half, 1> &dacc_values, const sycl::accessor<sycl::half, 1> &dacc_B, const sycl::accessor<sycl::half, 1> &dacc_out, const sycl_dacc_float &dacc_dequant_stats);
                                                       
template SYCL_EXTERNAL void kspmm_coo_very_sparse_naive<sycl::half, 16, 16>(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, sycl::half *values, sycl::half *B, sycl::half *out, float * __restrict__ const dequant_stats, int nnz, int rowsA, int rowsB, int colsB, const sycl::nd_item<3> &item_ct1,
 sycl::half *smem_dequant_stats, const sycl_dacc &dacc_max_count, const sycl_dacc &dacc_max_idx, const sycl_dacc &dacc_offset_rowidx, const sycl_dacc &dacc_rowidx, const sycl_dacc &dacc_colidx, const sycl::accessor<sycl::half, 1> &dacc_values, const sycl::accessor<sycl::half, 1> &dacc_B, const sycl::accessor<sycl::half, 1> &dacc_out, const sycl_dacc_float &dacc_dequant_stats);
                                                       

template SYCL_EXTERNAL void kspmm_coo_very_sparse_naive<sycl::half, 32, 16>(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, sycl::half *values, sycl::half *B, sycl::half *out, float * __restrict__ const dequant_stats, int nnz, int rowsA, int rowsB, int colsB, const sycl::nd_item<3> &item_ct1,
 sycl::half *smem_dequant_stats, const sycl_dacc &dacc_max_count, const sycl_dacc &dacc_max_idx, const sycl_dacc &dacc_offset_rowidx, const sycl_dacc &dacc_rowidx, const sycl_dacc &dacc_colidx, const sycl::accessor<sycl::half, 1> &dacc_values, const sycl::accessor<sycl::half, 1> &dacc_B, const sycl::accessor<sycl::half, 1> &dacc_out, const sycl_dacc_float &dacc_dequant_stats);

                                                       
template SYCL_EXTERNAL void kspmm_coo_very_sparse_naive<sycl::half, 32, 8>(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, sycl::half *values, sycl::half *B, sycl::half *out, float * __restrict__ const dequant_stats, int nnz, int rowsA, int rowsB, int colsB, const sycl::nd_item<3> &item_ct1,
 sycl::half *smem_dequant_stats, const sycl_dacc &dacc_max_count, const sycl_dacc &dacc_max_idx, const sycl_dacc &dacc_offset_rowidx, const sycl_dacc &dacc_rowidx, const sycl_dacc &dacc_colidx, const sycl::accessor<sycl::half, 1> &dacc_values, const sycl::accessor<sycl::half, 1> &dacc_B, const sycl::accessor<sycl::half, 1> &dacc_out, const sycl_dacc_float &dacc_dequant_stats);

template SYCL_EXTERNAL void kspmm_coo_very_sparse_naive<sycl::half, 16, 8>(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, sycl::half *values, sycl::half *B, sycl::half *out, float * __restrict__ const dequant_stats, int nnz, int rowsA, int rowsB, int colsB, const sycl::nd_item<3> &item_ct1,
 sycl::half *smem_dequant_stats, const sycl_dacc &dacc_max_count, const sycl_dacc &dacc_max_idx, const sycl_dacc &dacc_offset_rowidx, const sycl_dacc &dacc_rowidx, const sycl_dacc &dacc_colidx, const sycl::accessor<sycl::half, 1> &dacc_values, const sycl::accessor<sycl::half, 1> &dacc_B, const sycl::accessor<sycl::half, 1> &dacc_out, const sycl_dacc_float &dacc_dequant_stats);

template SYCL_EXTERNAL void kspmm_coo_very_sparse_naive<sycl::half, 8, 8>(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, sycl::half *values, sycl::half *B, sycl::half *out, float * __restrict__ const dequant_stats, int nnz, int rowsA, int rowsB, int colsB, const sycl::nd_item<3> &item_ct1,
 sycl::half *smem_dequant_stats, const sycl_dacc &dacc_max_count, const sycl_dacc &dacc_max_idx, const sycl_dacc &dacc_offset_rowidx, const sycl_dacc &dacc_rowidx, const sycl_dacc &dacc_colidx, const sycl::accessor<sycl::half, 1> &dacc_values, const sycl::accessor<sycl::half, 1> &dacc_B, const sycl::accessor<sycl::half, 1> &dacc_out, const sycl_dacc_float &dacc_dequant_stats);
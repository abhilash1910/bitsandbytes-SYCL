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
float atomicMax(float* address, float val) {
  int* address_as_i = reinterpret_cast<int*>(address);
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = dpct::atomic_compare_exchange_strong<sycl::access::address_space::generic_space>(reinterpret_cast<int*>(address), assumed, sycl::bit_cast<int>(sycl::fmax(val, sycl::bit_cast<float>(assumed))));
  } while (assumed != old);
  return sycl::bit_cast<float>(old);
}

float atomicMin(float* address, float val) {
  int* address_as_i = reinterpret_cast<int*>(address);
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = dpct::atomic_compare_exchange_strong<sycl::access::address_space::generic_space>(reinterpret_cast<int*>(address), assumed, sycl::bit_cast<int>(sycl::fmin(val, sycl::bit_cast<float>(assumed))));
  } while (assumed != old);
  return sycl::bit_cast<float>(old);
}

float dDequantizeFP4(unsigned char val, float absmax)
{
  float sign = (val & 0b1000) == 8 ? -1.0f : 1.0f;
  if((val & 0b0110) == 0)
  {
    // subnormal
    if((val & 0b0001) == 0)
      return 0.0f;
    else
      return sign*0.0625f*absmax;
  }
  else
  {
    // normal
    float exponent = ((val & 0b0100) == 4 ? 2.0f : 8.0f) + ((val & 0b0010) == 2 ? 0.0f : 2.0f);
    float fraction = (val & 0b0001) == 1 ? 1.5f : 1.0f;

    return sign*exponent*fraction*absmax;
  }
}

float d2DequantizeFP4(unsigned char val)
{
  float sign = (val & 0b1000) == 8 ? -1.0f : 1.0f;
  if((val & 0b0110) == 0)
  {
    // subnormal
    if((val & 0b0001) == 0)
      return 0.0f;
    else
      return sign*0.0625f;
  }
  else
  {
    // normal
    float exponent = ((val & 0b0100) == 4 ? 2.0f : 8.0f) + ((val & 0b0010) == 2 ? 0.0f : 2.0f);
    float fraction = (val & 0b0001) == 1 ? 1.5f : 1.0f;

    return sign*exponent*fraction;
  }
}

float dDequantizeFP4Tree(unsigned char val, float absmax)
{
  float sign = (val & 0b1000) == 8 ? -1.0f : 1.0f;
  if((val & 0b0100) == 4) // 0
    if((val & 0b0010) == 2) //01
      if((val & 0b0001) == 1) // 111
        return 0.25000000f*absmax*sign; // 1111
      else
        return 0.16666667f*absmax*sign; // 1110
    else
      if((val & 0b0001) == 1) // 110
        return 0.50000000f*absmax*sign; // 1101
      else
        return 0.33333333f*absmax*sign; // 1100
  else
    if((val & 0b0010) == 2) //10
      if((val & 0b0001) == 1) // 101
        return 1.00000000f*absmax*sign; // 1011
      else
        return 0.66666667f*absmax*sign; // 1010
    else
      if((val & 0b0001) == 1) // 100
        return 5.208333333e-03f*absmax*sign; // 1001
      else
        return 0.00000000f*absmax*sign; // 1000
}

unsigned char dQuantizeFP4(float x)
{
  // FP4 with bias of 3
  // first bit is a sign
  // subnormals
  // 0b000 = 0
  // 0b001 = 0.0625
  // 0b110 = 2
  // 0b111 = 3
  // 0b100 = 4
  // 0b101 = 6
  // 0b010 = 8
  // 0b011 = 12


  // we do a binary search
  // the pivots are divided by 12 (the FP4 absmax)
  // since we assume input data is in [-1.0, 1.0]

  // !be careful here, its easy to make a mistake
  // that is difficult to notice if you add an extra
  // zero somewhere!

  int sign = x < 0 ? 0b1000 : 0b0000;
  x = sycl::fabs(x);
  if(x > 0.29166667f)
    if( x > 0.583333f)
      if( x > 0.8333333f)
        return 0b0011+sign;
      else
        return 0b0010+sign;
    else
      if(x > 0.4166667f)
        return 0b101+sign;
      else
        return 0b100+sign;
  else
    if(x > 0.0859375f)
      if(x > 0.20833333f)
        return 0b0111+sign;
      else
        return 0b0110+sign;
    else
      if(x > 0.00260417f)
        return 0b0001+sign;
      else
        return 0b0000+sign;
}

sycl::half dhDequantizeNF4(unsigned char val)
{
  // the values for this tree was generated by test_normal_map_tree
  // in the file tests/test_functional.py
  if((val & 0b1000) == 8)
    if((val & 0b0100) == 4) // 1
      if((val & 0b0010) == 2) // 11
        if((val & 0b0001) == 1) // 111
          return 1.0f;
        else
          return 0.7229568362236023f;
      else
        if((val & 0b0001) == 1) // 110
          return 0.5626170039176941f;
        else
          return 0.44070982933044434f;
    else
      if((val & 0b0010) == 2) //10
        if((val & 0b0001) == 1) // 101
          return 0.33791524171829224f;
        else
          return 0.24611230194568634f;
      else
        if((val & 0b0001) == 1) // 100
          return 0.16093020141124725f;
        else
          return 0.07958029955625534f;

  else
    if((val & 0b0100) == 4) // 0
      if((val & 0b0010) == 2) //01
        if((val & 0b0001) == 1) // 011
          return 0.0f;
        else
          return -0.09105003625154495f;
      else
        if((val & 0b0001) == 1) // 010
          return -0.18477343022823334f;
        else
          return -0.28444138169288635f;
    else
      if((val & 0b0010) == 2) //00
        if((val & 0b0001) == 1) // 001
          return -0.39491748809814453f;
        else
          return -0.5250730514526367f;
      else
        if((val & 0b0001) == 1) // 000
          return -0.6961928009986877f;
        else
          return -1.0f;

}

float dDequantizeNF4(unsigned char val)
{

  // the values for this tree was generated by test_normal_map_tree
  // in the file tests/test_functional.py
  if((val & 0b1000) == 8)
    if((val & 0b0100) == 4) // 1
      if((val & 0b0010) == 2) // 11
        if((val & 0b0001) == 1) // 111
          return 1.0f;
        else
          return 0.7229568362236023f;
      else
        if((val & 0b0001) == 1) // 110
          return 0.5626170039176941f;
        else
          return 0.44070982933044434f;
    else
      if((val & 0b0010) == 2) //10
        if((val & 0b0001) == 1) // 101
          return 0.33791524171829224f;
        else
          return 0.24611230194568634f;
      else
        if((val & 0b0001) == 1) // 100
          return 0.16093020141124725f;
        else
          return 0.07958029955625534f;

  else
    if((val & 0b0100) == 4) // 0
      if((val & 0b0010) == 2) //01
        if((val & 0b0001) == 1) // 011
          return 0.0f;
        else
          return -0.09105003625154495f;
      else
        if((val & 0b0001) == 1) // 010
          return -0.18477343022823334f;
        else
          return -0.28444138169288635f;
    else
      if((val & 0b0010) == 2) //00
        if((val & 0b0001) == 1) // 001
          return -0.39491748809814453f;
        else
          return -0.5250730514526367f;
      else
        if((val & 0b0001) == 1) // 000
          return -0.6961928009986877f;
        else
          return -1.0f;

}

unsigned char dQuantizeNF4(float x)
{

  // the values for this tree was generated by test_normal_map_tree
  // in the file tests/test_functional.py
  if(x > 0.03979014977812767f)
    if(x > 0.3893125355243683f) // 1
      if(x > 0.6427869200706482f) // 11
        if(x > 0.8614784181118011f) // 111
          return 0b1111;
        else
          return 0b1110;
      else
        if(x > 0.5016634166240692f) // 110
          return 0b1101;
        else
          return 0b1100;
    else
      if(x > 0.2035212516784668f) // 10
        if(x > 0.2920137718319893f) // 101
          return 0b1011;
        else
          return 0b1010;
      else
        if(x > 0.1202552504837513f) // 100
          return 0b1001;
        else
          return 0b1000;
  else
    if(x > -0.33967943489551544f) // 0
      if(x > -0.13791173323988914f) // 01
        if(x > -0.045525018125772476f) // 011
          return 0b0111;
        else
          return 0b0110;
      else
        if(x > -0.23460740596055984f) // 010
          return 0b0101;
        else
          return 0b0100;
    else
      if(x > -0.6106329262256622f) // 00
        if(x > -0.4599952697753906f) // 001
          return 0b0011;
        else
          return 0b0010;
      else
        if(x > -0.8480964004993439f) // 000
          return 0b0001;
        else
          return 0b0000;
}
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
                            const sycl::nd_item<3> &item_ct1, const sycl_dacc_float &dacc_histogram, const sycl_dacc &dacc_index1, 
                            const sycl_dacc &dacc_index2, const sycl_dacc_float &dacc_src)
{
  const int tid = item_ct1.get_local_id(2) + (item_ct1.get_local_range(2)*item_ct1.get_group(2));
  const int numThreads = item_ct1.get_local_range(2)*item_ct1.get_group_range(2);

  for(int i = tid; i < n; i+=numThreads)
  {
      int idx = (dacc_index1[i]*maxidx1) + dacc_index2[i];
      dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&dacc_histogram[idx], dacc_src[i]);
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

//======================estimte quantiles=====================================

template<typename T>
SYCL_EXTERNAL 
void kEstimateQuantiles(const T *A, float *code, const float offset, const T max_val, const int n,
                        const sycl::nd_item<3> &item_ct1, sycl_la tacc, const sycl::accessor<T, 1> &dacc_A, const sycl_dacc_float &dacc_code)
{
  const int n_full = (BLOCK_ESTIMATE*(n/BLOCK_ESTIMATE)) + (n % BLOCK_ESTIMATE == 0 ? 0 : BLOCK_ESTIMATE);
  int valid_items = (item_ct1.get_group(2)+1 == item_ct1.get_group_range(2)) ? n - (item_ct1.get_group(2)*BLOCK_ESTIMATE) : BLOCK_ESTIMATE;
  const int base_idx = (item_ct1.get_group(2) * BLOCK_ESTIMATE);
  const float reciprocal_num_blocks = 1.0f/(n < 4096 ? 1.0f : (n/BLOCK_ESTIMATE));
  
  using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, int,  int *, sycl::nd_item<3>>;
  //using group_radix_sort = dpct::group::radix_sort<int, NUM_ESTIMATE>;
        
  T vals[NUM_ESTIMATE];
  auto *d_A = dacc_A.template get_multi_ptr<sycl::access::decorated::yes>().get();
  
  
  int smem_qidx[BLOCK_ESTIMATE];
  
  for (unsigned int i = base_idx; i < n_full; i += item_ct1.get_group_range(2)*BLOCK_ESTIMATE)
  {
      valid_items = n - i > BLOCK_ESTIMATE ? BLOCK_ESTIMATE : n - i;

      // do not process half-blocks
      if(valid_items < BLOCK_ESTIMATE && n > BLOCK_ESTIMATE){ continue; }

      #pragma unroll 4
      for(int j = 0; j < NUM_ESTIMATE; j++)
          vals[j] = max_val;

     
      item_ct1.barrier(sycl::access::fence_space::local_space);
  
      // 1. load 8 values per thread
      // 2. compute 2-max in registers (64 max per warp)
      // 3. do warp reduction + broadcast back
      // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
      // 5. Repeat (3) 8 times for top 8 values in 256
      // 6. store with byte index
      auto *tmp = tacc.get_multi_ptr<sycl::access::decorated::yes>().get();
      group_load(tmp).load(item_ct1, d_A, vals);
      
      #pragma unroll 4
      for(int j = 0; j < NUM_ESTIMATE; j++)
          vals[j] = ((float)vals[j]) * reciprocal_num_blocks;


      
      item_ct1.barrier();
      // sort into striped pattern to mitigate bank conflicts
      // striped pattern index for thread 0 [0, 1024, 2048, 3096]
      // striped pattern index for thread 1 [1, 1025, 2049, 3097]
      
      //BlockRadixSort(temp_storage.sort).SortBlockedToStriped(vals);
      
      // 1. load 8 values per thread
      // 2. compute 2-max in registers (64 max per warp)
      // 3. do warp reduction + broadcast back
      // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
      // 5. Repeat (3) 8 times for top 8 values in 256
      // 6. store with byte index
      //group_radix_sort(tmp).sort_blocked_to_striped(item_ct1, vals);

      
      item_ct1.barrier(sycl::access::fence_space::local_space);
      for(int j = item_ct1.get_local_id(2); j < BLOCK_ESTIMATE; j+=item_ct1.get_local_range(2))
          smem_qidx[j] = -1;

      
      item_ct1.barrier(sycl::access::fence_space::local_space);

      if(item_ct1.get_local_id(2) < 256)
      {
          float q_interval = (1.0f-(2.0f*offset))/255.0f;
          
          int local_idx = sycl::round(((offset+(item_ct1.get_local_id(2)*q_interval))*(valid_items-1)));
          smem_qidx[local_idx] = item_ct1.get_local_id(2);
      }

      
      item_ct1.barrier(sycl::access::fence_space::local_space);

      for(int i = item_ct1.get_local_id(2); i < BLOCK_ESTIMATE; i+=item_ct1.get_local_range(2))
      {
          if(smem_qidx[i] != -1)
              dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&dacc_code[smem_qidx[i]], vals[i/THREADS_ESTIMATE]);
      }
      
  }
 
}

//====================================k quantize===========================================
SYCL_EXTERNAL 
void kQuantize(float * code, float * __restrict__ const A, unsigned char *out, const int n,
               const sycl::nd_item<3> &item_ct1, float* smem_code, const sycl_la &tacc, const sycl_dacc_float &dacc_A,
               const sycl_dacc_uc &dacc_out, const sycl_dacc_float &dacc_code)
{
  const int n_full = (NUM_BLOCK*(n/NUM_BLOCK)) + (n % NUM_BLOCK == 0 ? 0 : NUM_BLOCK);
  int valid_items = (item_ct1.get_group(2)+1 == item_ct1.get_group_range(2)) ? n - (item_ct1.get_group(2)*NUM_BLOCK) : NUM_BLOCK;
  const int base_idx = (item_ct1.get_group(2) * NUM_BLOCK);

  float vals[NUM];
  unsigned char qvals[NUM];
  //const int lane_id = threadIdx.x % 2;

  using group_load_float = dpct::group::workgroup_load<NUM, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;  
  using group_store_uc = dpct::group::workgroup_store<NUM, dpct::group::store_algorithm::BLOCK_STORE_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;  

  auto *d_A = dacc_A.template get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *d_out = dacc_out.get_multi_ptr<sycl::access::decorated::yes>().get();

  if(item_ct1.get_local_id(2) < 256)
  {
    smem_code[item_ct1.get_local_id(2)] = dacc_code[item_ct1.get_local_id(2)];
    //smem_code[0][threadIdx.x] = code[threadIdx.x];
    //smem_code[1][threadIdx.x] = smem_code[0][threadIdx.x];
  }


  for (unsigned int i = base_idx; i < n_full; i += item_ct1.get_group_range(2)*NUM_BLOCK)
  {
      // number of values already processed in blocks +
      // number of values already processed in this block +
      // rand_offset % mod value
      valid_items = n - i > NUM_BLOCK ? NUM_BLOCK : n - i;

     
      item_ct1.barrier(sycl::access::fence_space::local_space);
  
      // 1. load 8 values per thread
      // 2. compute 2-max in registers (64 max per warp)
      // 3. do warp reduction + broadcast back
      // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
      // 5. Repeat (3) 8 times for top 8 values in 256
      // 6. store with byte index
      auto *tmp = tacc.get_multi_ptr<sycl::access::decorated::yes>().get();
      group_load_float(tmp).load(item_ct1, d_A, vals);
      
      #pragma unroll 4
      for(int j = 0; j < NUM; j++)
          qvals[j] = dQuantize<0>(smem_code, 0.0f, vals[j]);

      
      item_ct1.barrier(sycl::access::fence_space::local_space);
      
      
      //1. load 8 values per thread
      // 2. compute 2-max in registers (64 max per warp)
      // 3. do warp reduction + broadcast back
      // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
      // 5. Repeat (3) 8 times for top 8 values in 256
      // 6. store with byte index
      group_store_uc(tmp).store(item_ct1, d_out, qvals);
 }     
}




//===========================k quantize blockwise================================

template<typename T, int BLOCK_SIZE, int NUM_PER_TH, int STOCHASTIC, int DATA_TYPE>
//__launch_bounds__(TH, 4)
SYCL_EXTERNAL void kQuantizeBlockwise(float * code, T * __restrict__ const A, float *absmax, unsigned char *out, float * __restrict__ const rand, const int rand_offset, const int n,
                        const sycl::nd_item<3> &item_ct1, float *smem_code,
                        float *smem_absmax_value,const sycl_la &tacc,const sycl::accessor<T, 1> &dacc_A,
                        const sycl_dacc_float &dacc_rand, const sycl_dacc_uc &dacc_out,
                        const sycl_dacc_float &dacc_code, const sycl_dacc_float &dacc_absmax)
{
  
  
  const int n_full = item_ct1.get_group_range(2) * BLOCK_SIZE;
  int valid_items = 0;
  const int base_idx = (item_ct1.get_group(2) * BLOCK_SIZE);

  T vals[NUM_PER_TH];
  float rand_vals[NUM_PER_TH];
  unsigned char qvals[(DATA_TYPE > 0) ? NUM_PER_TH/2 : NUM_PER_TH];
  //float local_abs_max = -FLT_MAX;
  float local_abs_max = 0.0f;
  int local_rand_idx = 0;
  
  using group_load = dpct::group::workgroup_load<NUM_PER_TH, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
  using group_load_float = dpct::group::workgroup_load<NUM_PER_TH, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;  
  using group_store_uc = dpct::group::workgroup_store<(DATA_TYPE > 0) ? NUM_PER_TH/2 : NUM_PER_TH, dpct::group::store_algorithm::BLOCK_STORE_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;  

  auto *d_A = dacc_A.template get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *d_rand = dacc_rand.get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *d_out = dacc_out.get_multi_ptr<sycl::access::decorated::yes>().get();

  //code //absmax
  
  if(DATA_TYPE == 0)
    for(int i = item_ct1.get_local_id(2); i < 256; i+=item_ct1.get_local_range(2))
      smem_code[i] = dacc_code[i];

  for (unsigned int i = base_idx; i < n_full; i += item_ct1.get_group_range(2)*BLOCK_SIZE)
  {
    valid_items = n - i > BLOCK_SIZE ? BLOCK_SIZE : n - i;
    local_abs_max = -FLT_MAX;

    
    item_ct1.barrier(sycl::access::fence_space::local_space);
    
    // 1. load 8 values per thread
    // 2. compute 2-max in registers (64 max per warp)
    // 3. do warp reduction + broadcast back
    // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
    // 5. Repeat (3) 8 times for top 8 values in 256
    // 6. store with byte index
    auto *tmp = tacc.get_multi_ptr<sycl::access::decorated::yes>().get();
    group_load(tmp).load(item_ct1, d_A, vals);
    
    // 1. compute local max
    // 2. broadcast local max
    // 3. normalize inputs and quantize

    #pragma unroll NUM_PER_TH
    for(int j = 0; j < NUM_PER_TH; j++)
       local_abs_max = sycl::fmax(local_abs_max, sycl::fabs((float)vals[j]));

    local_abs_max = sycl::reduce_over_group(item_ct1.get_group(), local_abs_max, sycl::maximum<>());

    if(item_ct1.get_local_id(2) == 0)
      smem_absmax_value[0] = local_abs_max;

    
    item_ct1.barrier(sycl::access::fence_space::local_space);

    if(item_ct1.get_local_id(2) == 0)
      dacc_absmax[i/BLOCK_SIZE] = local_abs_max;
    else
      local_abs_max = smem_absmax_value[0];

    sycl::group_barrier(item_ct1.get_sub_group());

    local_abs_max = 1.0f/local_abs_max;

    if(STOCHASTIC)
    {
      local_rand_idx = ((item_ct1.get_group(2)*NUM_BLOCK) + (item_ct1.get_local_id(2)*NUM) + rand_offset) % (1024-4);
      
      // 1. load 8 values per thread
      // 2. compute 2-max in registers (64 max per warp)
      // 3. do warp reduction + broadcast back
      // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
      // 5. Repeat (3) 8 times for top 8 values in 256
      // 6. store with byte index
      group_load_float(tmp).load(item_ct1, d_rand, rand_vals);
          
    }

    unsigned char packed_4bit = 0;
    switch(DATA_TYPE)
    {
        case 0:
            #pragma unroll NUM_PER_TH
            for(int j = 0; j < NUM_PER_TH; j++)
            {
                if(!STOCHASTIC)
                 qvals[j] = dQuantize<0>(smem_code, 0.0f, ((float)vals[j])*local_abs_max);
                else
                 qvals[j] = dQuantize<1>(smem_code, rand_vals[j], ((float)vals[j])*local_abs_max);
            }
            break;
        case 1:
            #pragma unroll NUM_PER_TH
            for(int j = 0; j < NUM_PER_TH/2; j++)
            {
              packed_4bit |= dQuantizeFP4(((float)vals[2*j])*local_abs_max) << 4;
              packed_4bit |= dQuantizeFP4(((float)vals[2*j+1])*local_abs_max);
              qvals[j] = packed_4bit;
            }
            break;
        case 2:
            #pragma unroll NUM_PER_TH
            for(int j = 0; j < NUM_PER_TH/2; j++)
            {
              packed_4bit |= dQuantizeNF4(((float)vals[2*j])*local_abs_max) << 4;
              packed_4bit |= dQuantizeNF4(((float)vals[2*j+1])*local_abs_max);
              qvals[j] = packed_4bit;
            }
            break;
    }

    
    item_ct1.barrier(sycl::access::fence_space::local_space);
    
    // 1. load 8 values per thread
    // 2. compute 2-max in registers (64 max per warp)
    // 3. do warp reduction + broadcast back
    // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
    // 5. Repeat (3) 8 times for top 8 values in 256
    // 6. store with byte index
    group_store_uc(tmp).store(item_ct1, d_out, qvals);
    
  }
}

//===========================k dequantize================================


template<typename T, int TILE_SIZE, int THREADS, int NUM_PER_TH, int DATA_TYPE>
SYCL_EXTERNAL void kDequantizeBlockwise(float *code, unsigned char * A, float * absmax, T *out, const int blocksize, const int n,
                          const sycl::nd_item<3> &item_ct1,const sycl_la &tacc,const  sycl_dacc_uc &dacc_A,const sycl::accessor<T, 1> &dacc_out, const sycl_dacc_float &dacc_code, const sycl_dacc_float &dacc_absmax )
{

  const int n_load = (item_ct1.get_group_range(2) * TILE_SIZE);
  int valid_items_load = 0;
  int valid_items_store = 0;
  const int base_idx = (item_ct1.get_group(2) * TILE_SIZE);

  T vals[NUM_PER_TH*((DATA_TYPE > 0) ? 2 : 1)];
  unsigned char qvals[NUM_PER_TH];
  float local_abs_max = -FLT_MAX;
  
  using group_load_uc = dpct::group::workgroup_load<NUM_PER_TH, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
  
  using group_store = dpct::group::workgroup_store<NUM_PER_TH*((DATA_TYPE > 0) ? 2 : 1), dpct::group::store_algorithm::BLOCK_STORE_DIRECT, T,  T *, sycl::nd_item<3>>;
  
  
  auto *d_A = dacc_A.template get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *d_out = dacc_out.template get_multi_ptr<sycl::access::decorated::yes>().get();
  //A //out //code //absmax
  
  //typedef cub::BlockLoad<unsigned char, THREADS, NUM_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadChar;
  //typedef cub::BlockStore<T, THREADS, NUM_PER_TH*((DATA_TYPE > 0) ? 2 : 1), cub::BLOCK_STORE_WARP_TRANSPOSE> StoreT;


  for (unsigned int i = base_idx; i < n_load; i += item_ct1.get_group_range(2)*TILE_SIZE)
  {
    if(DATA_TYPE > 0)
    {
      valid_items_load = (n+1)/2 - i > TILE_SIZE ? TILE_SIZE : (n+1)/2 - i;
      valid_items_store = n - i*2 > TILE_SIZE*2 ? TILE_SIZE*2 : n - i*2;
    }
    else
    {
      valid_items_load = n - i > TILE_SIZE ? TILE_SIZE : n - i;
      valid_items_store = n - i > TILE_SIZE ? TILE_SIZE : n - i;
    }
    
    local_abs_max =  dacc_absmax[(i+item_ct1.get_local_id(2)*NUM_PER_TH)/(blocksize)];//sycl::ext::oneapi::experimental::cuda::ldg(&absmax[(i+item_ct1.get_local_id(2)*NUM_PER_TH)/(blocksize)]);

    
    item_ct1.barrier(sycl::access::fence_space::local_space);
    
    //LoadChar(loadchar).Load(&(A[i]), qvals, valid_items_load, 128);
    // 1. load 8 values per thread
    // 2. compute 2-max in registers (64 max per warp)
    // 3. do warp reduction + broadcast back
    // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
    // 5. Repeat (3) 8 times for top 8 values in 256
    // 6. store with byte index
    auto *tmp = tacc.get_multi_ptr<sycl::access::decorated::yes>().get();
    group_load_uc(tmp).load(item_ct1, d_A, qvals);
    
      
    

    switch(DATA_TYPE)
    {
        case General8bit:
          // load code through read-only cache via __ldg
          #pragma unroll NUM_PER_TH
          for(int j = 0; j < NUM_PER_TH; j++)
            
            vals[j] =  dacc_code[qvals[j]]*local_abs_max;//sycl::ext::oneapi::experimental::cuda::ldg(&code[qvals[j]]*local_abs_max);
          break;
        case FP4:
          #pragma unroll NUM_PER_TH
          for(int j = 0; j < NUM_PER_TH; j++)
          {
            vals[j*2] = dDequantizeFP4Tree(qvals[j] >> 4, local_abs_max);
            vals[j*2 + 1] = dDequantizeFP4Tree(qvals[j] & 0x0F, local_abs_max);
          }
          break;
        case NF4:
          #pragma unroll NUM_PER_TH
          for(int j = 0; j < NUM_PER_TH; j++)
          {
            vals[j*2] = dDequantizeNF4(qvals[j] >> 4)* local_abs_max;
            vals[j*2 + 1] = dDequantizeNF4(qvals[j] & 0x0F)* local_abs_max;
          }
          break;
    }

   
    item_ct1.barrier();
   
    //StoreT(storet).Store(&(out[(DATA_TYPE > 0) ? i*2 : i]), vals, valid_items_store);
    
  
    // 1. load 8 values per thread
    // 2. compute 2-max in registers (64 max per warp)
    // 3. do warp reduction + broadcast back
    // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
    // 5. Repeat (3) 8 times for top 8 values in 256
    // 6. store with byte index

    group_store(tmp).store(item_ct1, d_out, vals);
            
  }
}

//======================k dequantize===============

SYCL_EXTERNAL void kDequantize(float *code, unsigned char *buff_A, float *buff_out, const int n,
                 const sycl::nd_item<3> &item_ct1, float *smem_code)
{
	const unsigned int numThreads = item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
	const int idx = (item_ct1.get_group(2) * item_ct1.get_local_range(2)) + item_ct1.get_local_id(2);

	
	if(item_ct1.get_local_id(2) < 256)
	{
		smem_code[item_ct1.get_local_id(2)] = code[item_ct1.get_local_id(2)];
	}

	item_ct1.barrier(sycl::access::fence_space::local_space);

	for (int i = idx;i < n; i += numThreads)
	{
		buff_out[i] = smem_code[buff_A[i]];
	}
}



//===================32 bit optimizer========================


template<typename T, int OPTIMIZER, int BLOCK_SIZE, int NUM_VALS>
/*
DPCT1110:1: The total declared local variable size in device function kPreconditionOptimizer32bit2State exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust the code, or use smaller sub-group size to avoid high register pressure.
*/

SYCL_EXTERNAL 
void kPreconditionOptimizer32bit2State(T* g, T* p,
                float* state1, float* state2, float *unorm,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const float gnorm_scale, const int n,
                const sycl::nd_item<3> &item_ct1,const sycl_la &tacc,const sycl_dacc_float &dacc_state1,const sycl_dacc_float &dacc_state2,const sycl::accessor<T, 1> &dacc_g, const sycl_dacc_float &dacc_unorm)
{

  const int n_full = (BLOCK_SIZE*(n/BLOCK_SIZE)) + (n % BLOCK_SIZE == 0 ? 0 : BLOCK_SIZE);
  const int base_idx = (item_ct1.get_group(2) * item_ct1.get_local_range(2) * NUM_VALS);
  int valid_items = 0;

  T g_vals[NUM_VALS];

  float s1_vals[NUM_VALS];
  float s2_vals[NUM_VALS];

  const float correction1 = 1.0f/(1.0f - pow(beta1, step));
  const float correction2 = 1.0f/(1.0f - pow(beta2, step));
  
  using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
  using group_load_float = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;  

  
  auto *d_g = dacc_g.template get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *d_state1 = dacc_state1.get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *d_state2 = dacc_state2.get_multi_ptr<sycl::access::decorated::yes>().get();
  
  
  
  
  for (unsigned int i = base_idx; i < n_full; i += item_ct1.get_group_range(2)*BLOCK_SIZE)
  {
      valid_items = n - i >= (BLOCK_SIZE) ? (BLOCK_SIZE) : n - i;

      
      item_ct1.barrier(sycl::access::fence_space::local_space);
  
      // 1. load 8 values per thread
      // 2. compute 2-max in registers (64 max per warp)
      // 3. do warp reduction + broadcast back
      // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
      // 5. Repeat (3) 8 times for top 8 values in 256
      // 6. store with byte index
      auto *tmp = tacc.get_multi_ptr<sycl::access::decorated::yes>().get();
      group_load(tmp).load(item_ct1, d_g, g_vals);
      
      
      item_ct1.barrier(sycl::access::fence_space::local_space);
      
      // 1. load 8 values per thread
      // 2. compute 2-max in registers (64 max per warp)
      // 3. do warp reduction + broadcast back
      // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
      // 5. Repeat (3) 8 times for top 8 values in 256
      // 6. store with byte index
      group_load_float(tmp).load(item_ct1, d_state1, s1_vals);
      
      
      
      item_ct1.barrier(sycl::access::fence_space::local_space);
      
      // 1. load 8 values per thread
      // 2. compute 2-max in registers (64 max per warp)
      // 3. do warp reduction + broadcast back
      // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
      // 5. Repeat (3) 8 times for top 8 values in 256
      // 6. store with byte index
      group_load_float(tmp).load(item_ct1, d_state2, s2_vals);
      
      # pragma unroll NUM_VALS
      for(unsigned int j = 0; j < NUM_VALS; j++)
        g_vals[j] = gnorm_scale*((float)g_vals[j]);

      # pragma unroll NUM_VALS
      for(unsigned int j = 0; j < NUM_VALS; j++)
      {
          switch(OPTIMIZER)
          {
              case ADAM:
                  s1_vals[j] = s1_vals[j]*beta1 + ((1.0f -beta1)*((float)g_vals[j]));
                  s2_vals[j] = s2_vals[j]*beta2 + ((1.0f -beta2)*(((float)g_vals[j])*((float)g_vals[j])));
                  s1_vals[j] *= correction1;
                  s2_vals[j] *= correction2;
                  s1_vals[j] = s1_vals[j]/(sycl::sqrt(s2_vals[j])+eps); // update
                  s1_vals[j] *= s1_vals[j]; // update l2 norm (update*update)
                  break;
          }
      }

      # pragma unroll NUM_VALS-1
      for(unsigned int j = 1; j < NUM_VALS; j++)
          s1_vals[0] += s1_vals[j];

    
      item_ct1.barrier(sycl::access::fence_space::local_space);
      s1_vals[0] = sycl::reduce_over_group(item_ct1.get_group(), s1_vals[0], sycl::plus<>());

      if(item_ct1.get_local_id(2) == 0)
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&dacc_unorm[0], s1_vals[0]);

      sycl::group_barrier(item_ct1.get_sub_group());
  }
}

#define NUM_PER_THREAD 4

template<typename T, int OPTIMIZER>
SYCL_EXTERNAL 
void kOptimizer32bit2State(T* g, T* p,
                float* state1, float* state2, float *unorm, const float max_unorm, const float param_norm,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const float gnorm_scale, const bool skip_zeros, const int n,
                const sycl::nd_item<3> &item_ct1,const sycl_la &tacc,const sycl::accessor<T, 1> &dacc_g,const sycl::accessor<T, 1> &dacc_p,const sycl_dacc_float &dacc_state1,const  sycl_dacc_float &dacc_state2, const sycl_dacc_float &dacc_unorm)
{

  const int n_full = ((TH*NUM_PER_THREAD)*(n/(TH*NUM_PER_THREAD))) + (n % (TH*NUM_PER_THREAD) == 0 ? 0 : (TH*NUM_PER_THREAD));
  const int base_idx = (item_ct1.get_group(2) * item_ct1.get_local_range(2) * NUM_PER_THREAD);
  int valid_items = 0;
  float update_scale = 0.0f;
  T g_vals[NUM_PER_THREAD];
  T p_vals[NUM_PER_THREAD];

  float s1_vals[NUM_PER_THREAD];
  float s2_vals[NUM_PER_THREAD];

  const float correction1 = 1.0f - pow(beta1, step);
  const float correction2 = sycl::sqrt(1.0f - pow(beta2, step));
  const float step_size = -lr*correction2/correction1;

  if(max_unorm > 0.0f)
  {
    update_scale = max_unorm > 0.0f ? sycl::sqrt(dacc_unorm[0]) : 1.0f;
    if(update_scale > max_unorm*param_norm){ update_scale = (max_unorm*param_norm)/update_scale; }
    else{ update_scale = 1.0f; }
  }
  else{ update_scale = 1.0f; }
  
  using group_load = dpct::group::workgroup_load<NUM_PER_THREAD, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
  using group_load_float = dpct::group::workgroup_load<NUM_PER_THREAD, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
  
  using group_store = dpct::group::workgroup_store<NUM_PER_THREAD, dpct::group::store_algorithm::BLOCK_STORE_DIRECT, T,  T *, sycl::nd_item<3>>;
  using group_store_float = dpct::group::workgroup_store<NUM_PER_THREAD, dpct::group::store_algorithm::BLOCK_STORE_DIRECT, float,  float *, sycl::nd_item<3>>;
  
  
  auto *d_g = dacc_g.template get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *d_p = dacc_p.template get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *d_state1 = dacc_state1.get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *d_state2 = dacc_state2.get_multi_ptr<sycl::access::decorated::yes>().get();
  
  
  for (unsigned int i = base_idx; i < n_full; i += item_ct1.get_group_range(2)*TH*NUM_PER_THREAD)
  {
      valid_items = n - i >= (TH*NUM_PER_THREAD) ? (TH*NUM_PER_THREAD) : n - i;

      
      item_ct1.barrier(sycl::access::fence_space::local_space);
      
      // 1. load 8 values per thread
      // 2. compute 2-max in registers (64 max per warp)
      // 3. do warp reduction + broadcast back
      // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
      // 5. Repeat (3) 8 times for top 8 values in 256
      // 6. store with byte index
      auto *tmp = tacc.get_multi_ptr<sycl::access::decorated::yes>().get();
      group_load(tmp).load(item_ct1, d_g , g_vals);
  
      
     
      item_ct1.barrier(sycl::access::fence_space::local_space);
      
      // 1. load 8 values per thread
      // 2. compute 2-max in registers (64 max per warp)
      // 3. do warp reduction + broadcast back
      // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
      // 5. Repeat (3) 8 times for top 8 values in 256
      // 6. store with byte index
      group_load_float(tmp).load(item_ct1, d_state1, s1_vals);
      
      
      item_ct1.barrier(sycl::access::fence_space::local_space);
      
      // 1. load 8 values per thread
      // 2. compute 2-max in registers (64 max per warp)
      // 3. do warp reduction + broadcast back
      // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
      // 5. Repeat (3) 8 times for top 8 values in 256
      // 6. store with byte index
      
      group_load_float(tmp).load(item_ct1, d_state2, s2_vals);
      
      
      item_ct1.barrier(sycl::access::fence_space::local_space);
      
     
      // 1. load 8 values per thread
      // 2. compute 2-max in registers (64 max per warp)
      // 3. do warp reduction + broadcast back
      // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
      // 5. Repeat (3) 8 times for top 8 values in 256
      // 6. store with byte index
      
      group_load(tmp).load(item_ct1, d_p, p_vals);
      


      # pragma unroll 4
      for(unsigned int j = 0; j < NUM_PER_THREAD; j++)
        g_vals[j] = gnorm_scale*((float)g_vals[j]);

      # pragma unroll 4
      for(unsigned int j = 0; j < NUM_PER_THREAD; j++)
      {
          switch(OPTIMIZER)
          {
              case ADAM:
									if(!skip_zeros || (skip_zeros && ((float)g_vals[j] != 0.0f)))
									{
										s1_vals[j] = s1_vals[j]*beta1 + ((1.0f -beta1)*((float)g_vals[j]));
										s2_vals[j] = s2_vals[j]*beta2 + ((1.0f -beta2)*(((float)g_vals[j])*((float)g_vals[j])));
										p_vals[j] = ((float)p_vals[j]) + (update_scale*step_size*(s1_vals[j]/(sycl::sqrt(s2_vals[j])+(eps*correction2))));

                    if(weight_decay > 0.0f)
                        p_vals[j] = ((float)p_vals[j])*(1.0f-(lr*weight_decay));
									}
                  break;
          }
      }

      
      item_ct1.barrier(sycl::access::fence_space::local_space);
      
      
      // 1. load 8 values per thread
      // 2. compute 2-max in registers (64 max per warp)
      // 3. do warp reduction + broadcast back
      // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
      // 5. Repeat (3) 8 times for top 8 values in 256
      // 6. store with byte index
      group_store(tmp).store(item_ct1, d_p , p_vals);

    
      
      item_ct1.barrier(sycl::access::fence_space::local_space);
      
      // 1. load 8 values per thread
      // 2. compute 2-max in registers (64 max per warp)
      // 3. do warp reduction + broadcast back
      // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
      // 5. Repeat (3) 8 times for top 8 values in 256
      // 6. store with byte index
      group_store_float(tmp).store(item_ct1, d_state1, s1_vals);

      
      item_ct1.barrier(sycl::access::fence_space::local_space);
      
      // 1. load 8 values per thread
      // 2. compute 2-max in registers (64 max per warp)
      // 3. do warp reduction + broadcast back
      // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
      // 5. Repeat (3) 8 times for top 8 values in 256
      // 6. store with byte index
      group_store_float(tmp).store(item_ct1, d_state2, s2_vals);

    
  }
}

template<typename T, int OPTIMIZER, int BLOCK_SIZE, int NUM_VALS>
SYCL_EXTERNAL 
void kPreconditionOptimizer32bit1State(T* g, T* p,
                float* buff_state1, float *unorm,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const float gnorm_scale, const int n,
                const sycl::nd_item<3> &item_ct1,const sycl_la &tacc,const sycl::accessor<T, 1> &dacc_g,const sycl_dacc_float &dacc_state1,
                const sycl_dacc_float &dacc_unorm)
{

  const int n_full = (BLOCK_SIZE*(n/BLOCK_SIZE)) + (n % BLOCK_SIZE == 0 ? 0 : BLOCK_SIZE);
  const int base_idx = (item_ct1.get_group(2) * item_ct1.get_local_range(2) * NUM_VALS);
  int valid_items = 0;

  T g_vals[NUM_VALS];

  float s1_vals[NUM_VALS];
  
  
  
  
  using group_load = dpct::group::workgroup_load<NUM_VALS, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
  using group_load_float = dpct::group::workgroup_load<NUM_VALS, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
  
  
  auto *d_g = dacc_g.template get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *d_state1 = dacc_state1.get_multi_ptr<sycl::access::decorated::yes>().get();
  
  
  for (unsigned int i = base_idx; i < n_full; i += item_ct1.get_group_range(2)*BLOCK_SIZE)
  {
      valid_items = n - i >= (BLOCK_SIZE) ? (BLOCK_SIZE) : n - i;

      item_ct1.barrier();
      
      // 1. load 8 values per thread
      // 2. compute 2-max in registers (64 max per warp)
      // 3. do warp reduction + broadcast back
      // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
      // 5. Repeat (3) 8 times for top 8 values in 256
      // 6. store with byte index
      auto *tmp = tacc.get_multi_ptr<sycl::access::decorated::yes>().get();
      group_load(tmp).load(item_ct1, d_g, g_vals);
      
      
      
      item_ct1.barrier(sycl::access::fence_space::local_space);
      
      //LoadFloat(temp_storage.loadf).Load(&(state1[i]), s1_vals, valid_items, 0.0f);
      
      // 1. load 8 values per thread
      // 2. compute 2-max in registers (64 max per warp)
      // 3. do warp reduction + broadcast back
      // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
      // 5. Repeat (3) 8 times for top 8 values in 256
      // 6. store with byte index
      group_load_float(tmp).load(item_ct1, d_state1, s1_vals);
      

      # pragma unroll NUM_VALS
      for(unsigned int j = 0; j < NUM_VALS; j++)
        g_vals[j] = gnorm_scale*((float)g_vals[j]);

      # pragma unroll NUM_VALS
      for(unsigned int j = 0; j < NUM_VALS; j++)
      {
          switch(OPTIMIZER)
          {
              case MOMENTUM:
                  if(step == 1)
                    s1_vals[j] = (float)g_vals[j]; // state update
                  else
                    s1_vals[j] = s1_vals[j]*beta1 + ((float)g_vals[j]); // state update
                  s1_vals[j] = s1_vals[j]*s1_vals[j]; // update norm
                  break;
              case LION:
                  s1_vals[j] = s1_vals[j]*beta2 + ((1.0f-beta2)*(float)g_vals[j]); // state update
                  break;
              case RMSPROP:
                  s1_vals[j] = s1_vals[j]*beta1 + ((1.0f-beta1)*((float)g_vals[j])*((float)g_vals[j])); // state update
                  s1_vals[j] = (float)g_vals[j] / (sycl::sqrt(s1_vals[j])+eps); // update value
                  s1_vals[j] = s1_vals[j]*s1_vals[j]; // update norm
                  break;
              case ADAGRAD:
                  s1_vals[j] = s1_vals[j] + ((float)g_vals[j])*((float)g_vals[j]); // state update
                  s1_vals[j] = (float)g_vals[j] / (sycl::sqrt(s1_vals[j])+eps); // update value
                  s1_vals[j] = s1_vals[j]*s1_vals[j]; // update norm
                  break;
          }
      }

      # pragma unroll
      for(unsigned int j = 1; j < NUM_VALS; j++)
        s1_vals[0] += s1_vals[j];

     
      item_ct1.barrier(sycl::access::fence_space::local_space);
     
      //s1_vals[0] = BlockReduce(temp_storage.reduce).Sum(s1_vals[0], valid_items);
      s1_vals[0] = sycl::reduce_over_group(item_ct1.get_group(), s1_vals[0], sycl::plus<>());
      
      if(item_ct1.get_local_id(2) == 0)
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&dacc_unorm[0], s1_vals[0]);

      sycl::group_barrier(item_ct1.get_sub_group());
  }
}


template<typename T, int OPTIMIZER>
SYCL_EXTERNAL 
void kOptimizer32bit1State(T *g, T *p,
                float *state1, float *unorm, const float max_unorm, const float param_norm,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const float gnorm_scale, const bool skip_zeros, const int n,
                const sycl::nd_item<3> &item_ct1,const sycl_la &tacc,const sycl::accessor<T, 1> &dacc_g,const sycl::accessor<T, 1> &dacc_p,const
                sycl_dacc_float &dacc_state1, const sycl_dacc_float &dacc_unorm)
{

  const int n_full = ((TH*NUM_PER_THREAD)*(n/(TH*NUM_PER_THREAD))) + (n % (TH*NUM_PER_THREAD) == 0 ? 0 : (TH*NUM_PER_THREAD));
  const int base_idx = (item_ct1.get_group(2) * item_ct1.get_local_range(2) * NUM_PER_THREAD);
  int valid_items = 0;
  float update_scale = 0.0f;

  if(max_unorm > 0.0f)
  {
    update_scale = max_unorm > 0.0f ? sycl::sqrt(dacc_unorm[0]) : 1.0f;
    if(update_scale > max_unorm*param_norm+eps){ update_scale = (max_unorm*param_norm+eps)/update_scale; }
    else{ update_scale = 1.0f; }
  }
  else{ update_scale = 1.0f; }

  T g_vals[NUM_PER_THREAD];
  T p_vals[NUM_PER_THREAD];

  float s1_vals[NUM_PER_THREAD];
  
  using group_load = dpct::group::workgroup_load<NUM_PER_THREAD, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
  using group_load_float = dpct::group::workgroup_load<NUM_PER_THREAD, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
  
  using group_store = dpct::group::workgroup_store<NUM_PER_THREAD, dpct::group::store_algorithm::BLOCK_STORE_DIRECT, T,  T *, sycl::nd_item<3>>;
  using group_store_float = dpct::group::workgroup_store<NUM_PER_THREAD, dpct::group::store_algorithm::BLOCK_STORE_DIRECT, float,  float *, sycl::nd_item<3>>;
  
  
  auto *d_g = dacc_g.template get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *d_p = dacc_p.template get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *d_state1 = dacc_state1.get_multi_ptr<sycl::access::decorated::yes>().get();
  
  for (unsigned int i = base_idx; i < n_full; i += item_ct1.get_group_range(2)*TH*NUM_PER_THREAD)
  {
      valid_items = n - i >= (TH*NUM_PER_THREAD) ? (TH*NUM_PER_THREAD) : n - i;

      
      item_ct1.barrier();
      
      // 1. load 8 values per thread
      // 2. compute 2-max in registers (64 max per warp)
      // 3. do warp reduction + broadcast back
      // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
      // 5. Repeat (3) 8 times for top 8 values in 256
      // 6. store with byte index
      auto *tmp = tacc.get_multi_ptr<sycl::access::decorated::yes>().get();
      group_load(tmp).load(item_ct1, d_g, g_vals); 
      
      
     
      item_ct1.barrier();
      
      // 1. load 8 values per thread
      // 2. compute 2-max in registers (64 max per warp)
      // 3. do warp reduction + broadcast back
      // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
      // 5. Repeat (3) 8 times for top 8 values in 256
      // 6. store with byte index
      group_load_float(tmp).load(item_ct1, d_state1, s1_vals);
  
      
     
      item_ct1.barrier();
      
      // 1. load 8 values per thread
      // 2. compute 2-max in registers (64 max per warp)
      // 3. do warp reduction + broadcast back
      // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
      // 5. Repeat (3) 8 times for top 8 values in 256
      // 6. store with byte index
      group_load(tmp).load(item_ct1, d_p, p_vals);
      
      # pragma unroll 4
      for(unsigned int j = 0; j < NUM_PER_THREAD; j++)
      {
        g_vals[j] = gnorm_scale*((float)g_vals[j]);
        if(weight_decay > 0.0f)
          g_vals[j] = (float)g_vals[j] + (((float)p_vals[j])*weight_decay);
      }

      # pragma unroll 4
      for(unsigned int j = 0; j < NUM_PER_THREAD; j++)
      {
					if(!skip_zeros || (skip_zeros && ((float)g_vals[j] != 0.0f)))
					{
						switch(OPTIMIZER)
						{
								case MOMENTUM:
										if(step == 1)
											s1_vals[j] = (float)g_vals[j];
										else
											s1_vals[j] = s1_vals[j]*beta1 + ((float)g_vals[j]);

										p_vals[j] = ((float)p_vals[j]) + update_scale*(-lr*(s1_vals[j]));
										break;
								case LION:
										p_vals[j] = ((float)p_vals[j]) - update_scale*(lr*sgn(((float)s1_vals[j])*beta1 + ((1.0f-beta1)*((float)g_vals[j]))));
										s1_vals[j] = s1_vals[j]*beta2 + ((1.0f-beta2)*((float)g_vals[j]));
										break;
								case RMSPROP:
										s1_vals[j] = s1_vals[j]*beta1 + ((1.0f-beta1)*((float)g_vals[j])*((float)g_vals[j]));
										p_vals[j] = ((float)p_vals[j]) - update_scale*(lr*(float)g_vals[j] / (sycl::sqrt((float)s1_vals[j])+eps));
										break;
								case ADAGRAD:
										s1_vals[j] = s1_vals[j] + ((float)g_vals[j])*((float)g_vals[j]);
										p_vals[j] = ((float)p_vals[j]) - lr*(float)g_vals[j] / (sycl::sqrt((float)s1_vals[j])+eps);
										break;
						}
					}
      }

      
      item_ct1.barrier(sycl::access::fence_space::local_space);
     
      // 1. load 8 values per thread
      // 2. compute 2-max in registers (64 max per warp)
      // 3. do warp reduction + broadcast back
      // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
      // 5. Repeat (3) 8 times for top 8 values in 256
      // 6. store with byte index
      group_store(tmp).store(item_ct1, d_p, p_vals);
      
      
      item_ct1.barrier();
      
      // 1. load 8 values per thread
      // 2. compute 2-max in registers (64 max per warp)
      // 3. do warp reduction + broadcast back
      // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
      // 5. Repeat (3) 8 times for top 8 values in 256
      // 6. store with byte index
      group_store_float(tmp).store(item_ct1, d_state1, s1_vals);
      
  
  }
}


//===================8 bit optimizer========================


#define NUM8BIT 16
#define NUM_THREADS 256
#define NUM_PER_BLOCK 4096

template<typename T, int OPTIMIZER>

SYCL_EXTERNAL void
kPreconditionOptimizerStatic8bit2State(T* p, T* __restrict__ const g, unsigned char*__restrict__  const state1, 
                unsigned char* __restrict__ const state2,
                float *unorm,
                const float beta1, const float beta2,
                const float eps, const int step,
                float* __restrict__ const quantiles1, float* __restrict__ const quantiles2,
                float* max1, float* max2, float* new_max1, float* new_max2,
                const float gnorm_scale, const int n,
                const sycl::nd_item<3> &item_ct1, 
                float* smem_quantiles1, float* smem_quantiles2,
                const sycl_la &tacc, const sycl::accessor<T, 1> &dacc_g, const sycl_dacc_uc &dacc_state1, const sycl_dacc_uc &dacc_state2)
{
    const int n_full = item_ct1.get_group_range(2) * NUM_PER_BLOCK;
    const int base_idx = (item_ct1.get_group(2) * item_ct1.get_local_range(2) * NUM_PER_THREAD);
    int valid_items = n - (item_ct1.get_group(2)*NUM_PER_BLOCK) > NUM_PER_BLOCK ? NUM_PER_BLOCK : n - (item_ct1.get_group(2)*NUM_PER_BLOCK);
    float g_val = 0.0f;
    float local_max_s1 = -FLT_MAX;
    float local_max_s2 = -FLT_MAX;
    float local_unorm = 0.0f;

    float s2_vals[NUM8BIT];
    float s1_vals[NUM8BIT];
    T g_vals[NUM8BIT];
    unsigned char m_c1[NUM8BIT];
    unsigned char r_c2[NUM8BIT];

    using group_load = dpct::group::workgroup_load<NUM8BIT, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
  using group_load_uc = dpct::group::workgroup_load<NUM8BIT, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
  
  auto *d_g = dacc_g.template get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *d_state1 = dacc_state1.get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *d_state2 = dacc_state2.get_multi_ptr<sycl::access::decorated::yes>().get();
  
    if(item_ct1.get_local_id(2) < 256)
    {
        smem_quantiles1[item_ct1.get_local_id(2)] = quantiles1[item_ct1.get_local_id(2)];
        smem_quantiles2[item_ct1.get_local_id(2)] = quantiles2[item_ct1.get_local_id(2)];
    }

   
    item_ct1.barrier(sycl::access::fence_space::local_space);

    for (unsigned int i = base_idx; i < n_full; i += NUM_THREADS*item_ct1.get_group_range(2)*NUM8BIT)
    {
        valid_items = n - i >= (TH*NUM_PER_THREAD) ? (TH*NUM_PER_THREAD) : n - i;

        // 1. load 8 values per thread
        // 2. compute 2-max in registers (64 max per warp)
        // 3. do warp reduction + broadcast back
        // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
        // 5. Repeat (3) 8 times for top 8 values in 256
        // 6. store with byte index
        auto *tmp = tacc.get_multi_ptr<sycl::access::decorated::yes>().get();
        group_load(tmp).load(item_ct1, d_g, g_vals);
        
      
        item_ct1.barrier(sycl::access::fence_space::local_space);
         
        // 1. load 8 values per thread
        // 2. compute 2-max in registers (64 max per warp)
        // 3. do warp reduction + broadcast back
        // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
        // 5. Repeat (3) 8 times for top 8 values in 256
        // 6. store with byte index
        group_load_uc(tmp).load(item_ct1, d_state1, m_c1);    
        
        
        item_ct1.barrier(sycl::access::fence_space::local_space);
        
        // 1. load 8 values per thread
        // 2. compute 2-max in registers (64 max per warp)
        // 3. do warp reduction + broadcast back
        // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
        // 5. Repeat (3) 8 times for top 8 values in 256
        // 6. store with byte index
        group_load_uc(tmp).load(item_ct1, d_state2, r_c2);
        
        
        item_ct1.barrier(sycl::access::fence_space::local_space);

        #pragma unroll 16
        for(int j = 0; j < NUM8BIT; j++)
        {
            g_val = g_vals[j];
            g_val *= gnorm_scale;
            s1_vals[j] = smem_quantiles1[m_c1[j]]*max1[0]*beta1;
            s1_vals[j] += (1.0f-beta1)*g_val;
            local_max_s1 = sycl::fmax(local_max_s1, sycl::fabs(s1_vals[j]));
        }

        #pragma unroll 16
        for(int j = 0; j < NUM8BIT; j++)
        {
            g_val = g_vals[j];
            g_val *= gnorm_scale;
            s2_vals[j] = smem_quantiles2[r_c2[j]]*max2[0]*beta2;
            s2_vals[j] += (1.0f-beta2)*g_val*g_val;
            local_max_s2 = sycl::fmax(local_max_s2, sycl::fabs(s2_vals[j]));
        }

        if(unorm != NULL)
        {
          #pragma unroll 16
          for(int j = 0; j < NUM8BIT; j++)
          {
            float correction1 = 1.0f / (1.0f - pow(beta1, step));
            float correction2 = 1.0f / (1.0f - pow(beta2, step));
            s1_vals[j] *= correction1;
            s2_vals[j] *= correction2;
            float update_val = s1_vals[j]/(sycl::sqrt(s2_vals[j])+eps); // update
            local_unorm += update_val*update_val;
          }
        }
    }

    
    item_ct1.barrier(sycl::access::fence_space::local_space);
    
    local_max_s1 = sycl::reduce_over_group(item_ct1.get_group(), local_max_s1, sycl::maximum<>());
   
    item_ct1.barrier(sycl::access::fence_space::local_space);
    
    local_max_s2 = sycl::reduce_over_group(item_ct1.get_group(), local_max_s2, sycl::maximum<>());
    if(unorm != NULL)
    {
      
      item_ct1.barrier(sycl::access::fence_space::local_space);
      
      local_unorm = sycl::reduce_over_group(item_ct1.get_group(), local_unorm, sycl::plus<>()); 
    }

    if(item_ct1.get_local_id(2) == 0)
    {
        atomicMax(&new_max1[0], local_max_s1);
        atomicMax(&new_max2[0], local_max_s2);
        if(unorm != NULL){ dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&unorm[0], local_unorm); }
    }
}

#define NUM_PER_THREAD2 4
#define NUM_THREADS2 1024
#define NUM_PER_BLOCK2 4096

template<typename T, int OPTIMIZER>
SYCL_EXTERNAL void

kOptimizerStatic8bit2State(T* p, T* const g, unsigned char* state1, unsigned char* state2,
                const float *unorm, const float max_unorm, const float param_norm, \
                const float beta1, const float beta2,
                const float eps, const int step, const float lr,
                float* __restrict__ const quantiles1, float* __restrict__ const quantiles2,
                float* max1, float* max2, float* new_max1, float* new_max2,
                float weight_decay,
                const float gnorm_scale, const int n,
                const sycl::nd_item<3> &item_ct1, float* smem_quantiles1, float* smem_quantiles2,
                const sycl_la &tacc, const sycl::accessor<T, 1> &dacc_g, const sycl::accessor<T, 1> &dacc_p,
                const sycl_dacc_uc &dacc_state1, const sycl_dacc_uc &dacc_state2
                )
{

    const int n_full = (item_ct1.get_local_range(2) * item_ct1.get_group_range(2))*NUM_PER_THREAD2;
    const int base_idx = (item_ct1.get_group(2) * item_ct1.get_local_range(2) * NUM_PER_THREAD2);
    int valid_items = 0;
    float g_val = 0.0f;
    float s1_vals[NUM_PER_THREAD2];
    float s2_vals[NUM_PER_THREAD2];
    const float correction1 = 1.0f - pow(beta1, step);
    const float correction2 = sycl::sqrt(1.0f - pow(beta2, step));
    const float step_size = -lr*correction2/correction1;
    //const float step_size = -lr*correction2/correction1;
    float new_max_val1 = 1.0f/new_max1[0];
    float new_max_val2 = 1.0f/new_max2[0];
    float update_scale = 1.0f;

    if(max_unorm > 0.0f)
    {
      update_scale = max_unorm > 0.0f ? sycl::sqrt((float)(unorm[0])) : 1.0f;
      if(update_scale > max_unorm*param_norm){ update_scale = (max_unorm*param_norm)/update_scale; }
      else{ update_scale = 1.0f; }
    }
    else{ update_scale = 1.0f; }

    unsigned char c1s[NUM_PER_THREAD2];
    unsigned char c2s[NUM_PER_THREAD2];
    T p_vals[NUM_PER_THREAD2];
    T g_vals[NUM_PER_THREAD2];
    
    using group_load = dpct::group::workgroup_load<NUM_PER_THREAD2, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
  using group_load_uc = dpct::group::workgroup_load<NUM_PER_THREAD2, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
  
  using group_store = dpct::group::workgroup_store<NUM_PER_THREAD2, dpct::group::store_algorithm::BLOCK_STORE_DIRECT, T,  T *, sycl::nd_item<3>>;
  using group_store_uc = dpct::group::workgroup_store<NUM_PER_THREAD2, dpct::group::store_algorithm::BLOCK_STORE_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
  
  
  auto *d_g = dacc_g.template get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *d_p = dacc_p.template get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *d_state1 = dacc_state1.get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *d_state2 = dacc_state2.get_multi_ptr<sycl::access::decorated::yes>().get();
    
    
    if(item_ct1.get_local_id(2) < 512)
    {
        if(item_ct1.get_local_id(2) < 256)
            smem_quantiles1[item_ct1.get_local_id(2)] = quantiles1[item_ct1.get_local_id(2)];
        else
            smem_quantiles2[item_ct1.get_local_id(2)-256] = quantiles2[item_ct1.get_local_id(2)-256];
    }

   
    item_ct1.barrier(sycl::access::fence_space::local_space);

    for (unsigned int i = base_idx; i < n_full; i += item_ct1.get_group_range(2)*NUM_THREADS2*NUM_PER_THREAD2)
    {
        valid_items = n - i >= (TH*NUM_PER_THREAD) ? (TH*NUM_PER_THREAD) : n - i;
        
        // 1. load 8 values per thread
        // 2. compute 2-max in registers (64 max per warp)
        // 3. do warp reduction + broadcast back
        // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
        // 5. Repeat (3) 8 times for top 8 values in 256
        // 6. store with byte index
        auto *tmp = tacc.get_multi_ptr<sycl::access::decorated::yes>().get();
        group_load(tmp).load(item_ct1, d_g, g_vals);
        
       
        item_ct1.barrier(sycl::access::fence_space::local_space);
        
        // 1. load 8 values per thread
        // 2. compute 2-max in registers (64 max per warp)
        // 3. do warp reduction + broadcast back
        // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
        // 5. Repeat (3) 8 times for top 8 values in 256
        // 6. store with byte index
        group_load_uc(tmp).load(item_ct1, d_state1, c1s);
        
        
        
        item_ct1.barrier(sycl::access::fence_space::local_space);
        
        // 1. load 8 values per thread
        // 2. compute 2-max in registers (64 max per warp)
        // 3. do warp reduction + broadcast back
        // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
        // 5. Repeat (3) 8 times for top 8 values in 256
        // 6. store with byte index
        group_load_uc(tmp).load(item_ct1, d_state2, c2s);
        
        
        
        item_ct1.barrier(sycl::access::fence_space::local_space);
        
        // 1. load 8 values per thread
        // 2. compute 2-max in registers (64 max per warp)
        // 3. do warp reduction + broadcast back
        // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
        // 5. Repeat (3) 8 times for top 8 values in 256
        // 6. store with byte index
        group_load(tmp).load(item_ct1, d_p, p_vals);
        
        if((i + (item_ct1.get_local_id(2)*NUM_PER_THREAD2) + NUM_PER_THREAD2) > n){ continue; }

        # pragma unroll 4
        for(unsigned int j = 0; j < NUM_PER_THREAD2; j++)
        {
            g_val = float(g_vals[j]);
            g_val *= gnorm_scale;
            s1_vals[j] = smem_quantiles1[c1s[j]];
            s1_vals[j] = s1_vals[j]*max1[0];

            s1_vals[j] = (s1_vals[j]*beta1) + (((1.0f-beta1)*g_val));

            c1s[j] = dQuantize<0>(smem_quantiles1, 0.0f, s1_vals[j]*new_max_val1);

            // make sure state1 term has still the same sign after quantization
            // (not needed for state2 term which has only positive values)
            if(sycl::signbit(smem_quantiles1[c1s[j]]) != sycl::signbit(s1_vals[j]))
            {
              if(s1_vals[j] > 0.0f)
                  c1s[j] += 1;
              else
                  c1s[j] -= 1;
            }

            s2_vals[j] = smem_quantiles2[c2s[j]];
            s2_vals[j] = s2_vals[j]*max2[0];
            s2_vals[j] = (s2_vals[j]*beta2) + (((1.0f-beta2)*g_val*g_val));
            c2s[j] = dQuantize<0>(smem_quantiles2, 0.0f, s2_vals[j]*new_max_val2);
        }

        # pragma unroll 4
        for(unsigned int j = 0; j < NUM_PER_THREAD2; j++)
        {
            p_vals[j] = (T)(((float)p_vals[j]) + ((update_scale*step_size*(s1_vals[j]/(sycl::sqrt(s2_vals[j])+(correction2*eps))))));
            if(weight_decay > 0.0f)
                p_vals[j] = update_scale*((float)p_vals[j])*(1.0f-(lr*weight_decay));
        }

        
        // 1. load 8 values per thread
        // 2. compute 2-max in registers (64 max per warp)
        // 3. do warp reduction + broadcast back
        // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
        // 5. Repeat (3) 8 times for top 8 values in 256
        // 6. store with byte index
        group_store(tmp).store(item_ct1, d_p, p_vals);
        
      
        
        item_ct1.barrier(sycl::access::fence_space::local_space);
        
        // 1. load 8 values per thread
        // 2. compute 2-max in registers (64 max per warp)
        // 3. do warp reduction + broadcast back
        // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
        // 5. Repeat (3) 8 times for top 8 values in 256
        // 6. store with byte index
        group_store_uc(tmp).store(item_ct1, d_state1, c1s);
        
        item_ct1.barrier(sycl::access::fence_space::local_space);
        
        // 1. load 8 values per thread
        // 2. compute 2-max in registers (64 max per warp)
        // 3. do warp reduction + broadcast back
        // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
        // 5. Repeat (3) 8 times for top 8 values in 256
        // 6. store with byte index
        group_store_uc(tmp).store(item_ct1, d_state2, c2s);
        
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
}

template<typename T, int OPTIMIZER>
SYCL_EXTERNAL void
kPreconditionOptimizerStatic8bit1State(T* p, T* __restrict__ const g, unsigned char*__restrict__  const state1,
                float *unorm,
                const float beta1, const float beta2,
                const float eps, const int step,
                float* __restrict__ const quantiles1,
                float* max1, float* new_max1,
                const float weight_decay,
                const float gnorm_scale, const int n,
                const sycl::nd_item<3> &item_ct1,
                float* smem_quantiles1,
                const sycl_la &tacc, const sycl::accessor<T, 1> &dacc_g, const sycl_dacc_uc &dacc_state1)
{
    const int n_full = item_ct1.get_group_range(2) * NUM_PER_BLOCK;
    const int base_idx = (item_ct1.get_group(2) * item_ct1.get_local_range(2) * NUM_PER_THREAD);
    int valid_items = n - (item_ct1.get_group(2)*NUM_PER_BLOCK) > NUM_PER_BLOCK ? NUM_PER_BLOCK : n - (item_ct1.get_group(2)*NUM_PER_BLOCK);
    float g_val = 0.0f;
    float local_max_s1 = -FLT_MAX;
    float local_unorm = 0.0f;

    float s1_vals[NUM8BIT];
    T g_vals[NUM8BIT];
    unsigned char m_c1[NUM8BIT];
    
     using group_load = dpct::group::workgroup_load<NUM8BIT, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
  using group_load_uc = dpct::group::workgroup_load<NUM8BIT, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
  
  auto *d_g = dacc_g.template get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *d_state1 = dacc_state1.get_multi_ptr<sycl::access::decorated::yes>().get();
  
  if(item_ct1.get_local_id(2) < 256)
      smem_quantiles1[item_ct1.get_local_id(2)] = quantiles1[item_ct1.get_local_id(2)];

    
    item_ct1.barrier(sycl::access::fence_space::local_space);

    for (unsigned int i = base_idx; i < n_full; i += item_ct1.get_group_range(2)*NUM_THREADS*NUM8BIT)
    {
        valid_items = n - i >= (TH*NUM_PER_THREAD) ? (TH*NUM_PER_THREAD) : n - i;

       
        item_ct1.barrier(sycl::access::fence_space::local_space);
        
        // 1. load 8 values per thread
        // 2. compute 2-max in registers (64 max per warp)
        // 3. do warp reduction + broadcast back
        // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
        // 5. Repeat (3) 8 times for top 8 values in 256
        // 6. store with byte index
        auto *tmp = tacc.get_multi_ptr<sycl::access::decorated::yes>().get();
        group_load(tmp).load(item_ct1, d_g, g_vals);
        
        item_ct1.barrier(sycl::access::fence_space::local_space);
        
        // 1. load 8 values per thread
        // 2. compute 2-max in registers (64 max per warp)
        // 3. do warp reduction + broadcast back
        // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
        // 5. Repeat (3) 8 times for top 8 values in 256
        // 6. store with byte index
        group_load_uc(tmp).load(item_ct1, d_state1, m_c1);
        
        #pragma unroll 16
        for(int j = 0; j < NUM8BIT; j++)
        {
            g_val = g_vals[j];
            g_val *= gnorm_scale;
            s1_vals[j] = smem_quantiles1[m_c1[j]]*max1[0];
            switch(OPTIMIZER)
            {
                case MOMENTUM:
                    if(step == 1)
                      s1_vals[j] = (float)g_vals[j];
                    else
                      s1_vals[j] = s1_vals[j]*beta1 + ((float)g_vals[j]);
                    if(unorm != NULL)
                      local_unorm += s1_vals[j]*s1_vals[j];
                    break;
              case LION:
                  s1_vals[j] = s1_vals[j]*beta2 + ((1.0f-beta2)*g_val);
                  break;
              case RMSPROP:
                    s1_vals[j] = s1_vals[j]*beta1 + ((1.0f-beta1)*(g_val*g_val));
                  break;
            }

            local_max_s1 = sycl::fmax(local_max_s1, sycl::fabs(s1_vals[j]));
        }
    }

    
    item_ct1.barrier(sycl::access::fence_space::local_space);
    
    local_max_s1 = sycl::reduce_over_group(item_ct1.get_group(), local_max_s1, sycl::maximum<>());
    if(item_ct1.get_local_id(2) == 0){ atomicMax(&new_max1[0], local_max_s1); }
    if(unorm != NULL)
    {
      
      item_ct1.barrier(sycl::access::fence_space::local_space);
      
      local_unorm = sycl::reduce_over_group(item_ct1.get_group(), local_unorm, sycl::plus<>());
      if(item_ct1.get_local_id(2) == 0){ dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&unorm[0], local_unorm); }
    }

}

template<typename T, int OPTIMIZER>
SYCL_EXTERNAL void
kOptimizerStatic8bit1State(T* p, T* const g, unsigned char* state1,
                const float *unorm, const float max_unorm, const float param_norm,
                const float beta1, const float beta2,
                const float eps, const int step, const float lr,
                float* __restrict__ const quantiles1,
                float* max1, float* new_max1,
                float weight_decay,
                const float gnorm_scale, const int n,
                const sycl::nd_item<3> &item_ct1,float *smem_quantiles1, const sycl_la &tacc,
                const sycl::accessor<T, 1> &dacc_g, const sycl::accessor<T, 1> &dacc_p,
                const sycl_dacc_uc &dacc_state1)
{

    const int n_full = (item_ct1.get_local_range(2) * item_ct1.get_group_range(2))*NUM_PER_THREAD2;
    const int base_idx = (item_ct1.get_group(2) * item_ct1.get_local_range(2) * NUM_PER_THREAD2);
    int valid_items = 0;
    float g_val = 0.0f;
    float s1_vals[NUM_PER_THREAD2];
    float new_max_val1 = 1.0f/new_max1[0];
    float update_scale = 1.0f;

    if(max_unorm > 0.0f)
    {
      update_scale = max_unorm > 0.0f ? sycl::sqrt((float)(unorm[0])) : 1.0f;
      if(update_scale > max_unorm*param_norm){ update_scale = (max_unorm*param_norm)/update_scale; }
      else{ update_scale = 1.0f; }
    }
    else{ update_scale = 1.0f; }

    unsigned char c1s[NUM_PER_THREAD2];
    T p_vals[NUM_PER_THREAD2];
    T g_vals[NUM_PER_THREAD2];
    
    
    
    using group_load = dpct::group::workgroup_load<NUM_PER_THREAD2, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
  using group_load_uc = dpct::group::workgroup_load<NUM_PER_THREAD2, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
  
  using group_store = dpct::group::workgroup_store<NUM_PER_THREAD2, dpct::group::store_algorithm::BLOCK_STORE_DIRECT, T,  T *, sycl::nd_item<3>>;
  using group_store_uc = dpct::group::workgroup_store<NUM_PER_THREAD2, dpct::group::store_algorithm::BLOCK_STORE_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
  
  
  auto *d_g = dacc_g.template get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *d_p = dacc_p.template get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *d_state1 = dacc_state1.get_multi_ptr<sycl::access::decorated::yes>().get();
    
    
    
    if(item_ct1.get_local_id(2) < 256)
        smem_quantiles1[item_ct1.get_local_id(2)] = quantiles1[item_ct1.get_local_id(2)];

    
    item_ct1.barrier(sycl::access::fence_space::local_space);

    for (unsigned int i = base_idx; i < n_full; i += item_ct1.get_group_range(2)*NUM_THREADS2*NUM_PER_THREAD2)
    {
        valid_items = n - i >= (TH*NUM_PER_THREAD) ? (TH*NUM_PER_THREAD) : n - i;
        
        // 1. load 8 values per thread
        // 2. compute 2-max in registers (64 max per warp)
        // 3. do warp reduction + broadcast back
        // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
        // 5. Repeat (3) 8 times for top 8 values in 256
        // 6. store with byte index
        auto *tmp = tacc.get_multi_ptr<sycl::access::decorated::yes>().get();
        group_load(tmp).load(item_ct1, d_g, g_vals);
        
        
        
        item_ct1.barrier(sycl::access::fence_space::local_space);
        
        // 1. load 8 values per thread
        // 2. compute 2-max in registers (64 max per warp)
        // 3. do warp reduction + broadcast back
        // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
        // 5. Repeat (3) 8 times for top 8 values in 256
        // 6. store with byte index
        group_load_uc(tmp).load(item_ct1, d_state1, c1s);
        
      
       
        item_ct1.barrier(sycl::access::fence_space::local_space);
        
        // 1. load 8 values per thread
        // 2. compute 2-max in registers (64 max per warp)
        // 3. do warp reduction + broadcast back
        // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
        // 5. Repeat (3) 8 times for top 8 values in 256
        // 6. store with byte index
        group_load(tmp).load(item_ct1, d_p, p_vals);
        
        if((i + (item_ct1.get_local_id(2)*NUM_PER_THREAD2) + NUM_PER_THREAD2) > n){ continue; }

        # pragma unroll 4
        for(unsigned int j = 0; j < NUM_PER_THREAD2; j++)
        {
            g_val = float(g_vals[j]);
            g_val *= gnorm_scale;

            if(weight_decay > 0.0f) {
              switch(OPTIMIZER) {
                case MOMENTUM:
                case RMSPROP:
                  g_val += ((float)p_vals[j])*weight_decay;
                  break;
                case LION:
                  p_vals[j] = ((float)p_vals[j])*(1.0f-lr*weight_decay);
                  break;
              }
            }

            s1_vals[j] = smem_quantiles1[c1s[j]]*max1[0];

            switch(OPTIMIZER)
            {
                case MOMENTUM:
                  if(step == 1)
                    s1_vals[j] = g_vals[j];
                  else
                    s1_vals[j] = s1_vals[j]*beta1 + ((float)g_vals[j]);

                  p_vals[j] = ((float)p_vals[j]) + (-lr*update_scale*(s1_vals[j]));
                  break;
              case LION:
                  p_vals[j] = ((float)p_vals[j]) - (lr*sgn(((float)s1_vals[j])*beta1 + ((1.0f-beta1)*((float)g_val))));
                  s1_vals[j] = s1_vals[j]*beta2 + ((1.0f-beta2)*g_val);
                  break;
              case RMSPROP:
                  s1_vals[j] = s1_vals[j]*beta1 + ((1.0f-beta1)*(g_val*g_val));
                  p_vals[j] = ((float)p_vals[j]) - (lr*g_val / (sycl::sqrt(s1_vals[j])+eps));
                  break;
            }

            c1s[j] = dQuantize<0>(smem_quantiles1, 0.0f, s1_vals[j]*new_max_val1);

            // make sure state1 term has still the same sign after quantization
            if(sycl::signbit(smem_quantiles1[c1s[j]]) != sycl::signbit(s1_vals[j]))
            {
              if(s1_vals[j] > 0.0f)
                  c1s[j] += 1;
              else
                  c1s[j] -= 1;
            }
        }

        
        // 1. load 8 values per thread
        // 2. compute 2-max in registers (64 max per warp)
        // 3. do warp reduction + broadcast back
        // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
        // 5. Repeat (3) 8 times for top 8 values in 256
        // 6. store with byte index
        group_store(tmp).store(item_ct1, d_p, p_vals);
        
        
        item_ct1.barrier(sycl::access::fence_space::local_space);
        
        // 1. load 8 values per thread
        // 2. compute 2-max in registers (64 max per warp)
        // 3. do warp reduction + broadcast back
        // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
        // 5. Repeat (3) 8 times for top 8 values in 256
        // 6. store with byte index
        group_store_uc(tmp).store(item_ct1, d_state1, c1s);
        
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
}


//===============================k percentile clipping============================================



template<typename T, int BLOCK_SIZE, int NUM_VALS>
SYCL_EXTERNAL void kPercentileClipping(T * __restrict__ g, float *gnorm_vec, int step, const int n,
                         const sycl::nd_item<3> &item_ct1,const sycl_la &tacc, const sycl::accessor<T, 1> &dacc_g)
{
  const int n_full = (BLOCK_SIZE*(n/BLOCK_SIZE)) + (n % BLOCK_SIZE == 0 ? 0 : BLOCK_SIZE);
  int valid_items = 0;

   using group_load = dpct::group::workgroup_load<NUM_VALS, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
   auto *d_g = dacc_g.template get_multi_ptr<sycl::access::decorated::yes>().get();
  
  T vals[NUM_VALS];
  float local_sum = 0.0f;

  for (unsigned int i = (item_ct1.get_group(2) * BLOCK_SIZE); i < n_full; i += item_ct1.get_group_range(2)*BLOCK_SIZE)
  {
      valid_items = n - i > BLOCK_SIZE ? BLOCK_SIZE : n - i;
      local_sum = 0.0f;

     
      item_ct1.barrier(sycl::access::fence_space::local_space);
     
      // 1. load 8 values per thread
      // 2. compute 2-max in registers (64 max per warp)
      // 3. do warp reduction + broadcast back
      // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
      // 5. Repeat (3) 8 times for top 8 values in 256
      // 6. store with byte index
      auto *tmp = tacc.get_multi_ptr<sycl::access::decorated::yes>().get();
      group_load(tmp).load(item_ct1, d_g, vals);
      
     #pragma unroll NUM_VALS
     for(int j = 0; j < NUM_VALS; j++)
       local_sum += ((float)vals[j])*((float)vals[j]);

    
    local_sum = sycl::reduce_over_group(item_ct1.get_group(), local_sum, sycl::plus<>());
    
    if(item_ct1.get_local_id(2) == 0)
    {
      if(step == 1)
      {
        // initialize with the same norm for all positions
        //#pragma unroll 10
        for(int j = 0; j < 100; j++)
          dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&gnorm_vec[j], local_sum);
      }
      else
          dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&gnorm_vec[step % 100], local_sum);
    }

  }
}


//=========================8 bit blockwise====================================


#define LANES 2
#define QUAD 3
template<typename T, int OPTIMIZER, int BLOCK_SIZE, int N_PER_TH>

SYCL_EXTERNAL 
void
kOptimizerStatic8bit2StateBlockwise(T* p, T* __restrict__ const g, unsigned char* state1, unsigned char* state2,
                const float beta1, const float beta2,
                const float eps, const int step, const float lr,
                float* __restrict__ const quantiles1, float* __restrict__ const quantiles2,
                float* absmax1, float* absmax2,
                float weight_decay,
                const float gnorm_scale, const bool skip_zeros, const int n,
                const sycl::nd_item<3> &item_ct1,
                sycl::local_accessor<float, 2> smem_quantiles1,
                sycl::local_accessor<float, 2> smem_quantiles2,
                float *smem_exchange1, float *smem_exchange2,
                const sycl_la &tacc, const sycl::accessor<T, 1> &dacc_g,
                const sycl::accessor<T, 1> &dacc_p,
                const sycl_dacc_uc &dacc_state1, const sycl_dacc_uc &dacc_state2,
                const sycl_dacc_float &dacc_quantiles1, const sycl_dacc_float &dacc_quantiles2,
                const sycl_dacc_float &dacc_absmax1, const sycl_dacc_float &dacc_absmax2)
{

    //const int n_full = n + (n%BLOCK_SIZE);
    const int n_full = item_ct1.get_group_range(2) * BLOCK_SIZE;
    const int base_idx = (item_ct1.get_group(2) * BLOCK_SIZE);
    int valid_items = 0;
    float g_val = 0.0f;
    float s1_vals[N_PER_TH];
    float s2_vals[N_PER_TH];
    // 2-5%
    const float correction1 = 1.0f - pow(beta1, step);
    const float correction2 = sycl::sqrt(1.0f -pow(beta2, step));
    const float step_size = (-lr*correction2) / correction1;
    const int lane_id = item_ct1.get_local_id(2) % LANES;
    float new_local_abs_max1 = -FLT_MAX;
    float new_local_abs_max2 = -FLT_MAX;
    float quadrants1[QUAD];
    float quadrants2[QUAD];

    unsigned char c1s[N_PER_TH];
    unsigned char c2s[N_PER_TH];
    T g_vals[N_PER_TH];
    T p_vals[N_PER_TH];
    
    
    
     using group_load = dpct::group::workgroup_load<N_PER_TH, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
  using group_load_uc = dpct::group::workgroup_load<N_PER_TH, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
  
  using group_store = dpct::group::workgroup_store<N_PER_TH, dpct::group::store_algorithm::BLOCK_STORE_DIRECT, T,  T *, sycl::nd_item<3>>;
  using group_store_uc = dpct::group::workgroup_store<N_PER_TH, dpct::group::store_algorithm::BLOCK_STORE_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
  
  
  auto *d_g = dacc_g.template get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *d_p = dacc_p.template get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *d_state1 = dacc_state1.get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *d_state2 = dacc_state2.get_multi_ptr<sycl::access::decorated::yes>().get();
    
    //quantiles1 //quantiles2 //absmax1 //absmax2
    
    // init: 0.2 -> 0.23

    // 0.23 -> 0.23
      smem_quantiles1[0][item_ct1.get_local_id(2)] = dacc_quantiles1[item_ct1.get_local_id(2)];
      smem_quantiles2[0][item_ct1.get_local_id(2)] = dacc_quantiles2[item_ct1.get_local_id(2)];
      # pragma unroll
      for(unsigned int j = 1; j < LANES; j++)
      {
        smem_quantiles1[j][item_ct1.get_local_id(2)] = smem_quantiles1[0][item_ct1.get_local_id(2)];
        smem_quantiles2[j][item_ct1.get_local_id(2)] = smem_quantiles2[0][item_ct1.get_local_id(2)];
      }

   
    item_ct1.barrier(sycl::access::fence_space::local_space);

    #pragma unroll
    for(int k = 0; k < QUAD; k++)
    {
      quadrants1[k] = smem_quantiles1[lane_id][(k*256/(QUAD+1)) + (256/(QUAD+1)-1)];
      quadrants2[k] = smem_quantiles2[lane_id][(k*256/(QUAD+1)) + (256/(QUAD+1)-1)];
    }


    for (unsigned int i = base_idx; i < n_full; i += item_ct1.get_group_range(2)*BLOCK_SIZE)
    {
        // loads: 0.23 -> 0.85/1.44
        valid_items = n - i >= BLOCK_SIZE ? BLOCK_SIZE : n - i;
        
        item_ct1.barrier(sycl::access::fence_space::local_space);
        
        // 1. load 8 values per thread
        // 2. compute 2-max in registers (64 max per warp)
        // 3. do warp reduction + broadcast back
        // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
        // 5. Repeat (3) 8 times for top 8 values in 256
        // 6. store with byte index
        auto *tmp = tacc.get_multi_ptr<sycl::access::decorated::yes>().get();
        group_load(tmp).load(item_ct1, d_g, g_vals);
        
        
        item_ct1.barrier(sycl::access::fence_space::local_space);
        
        // 1. load 8 values per thread
        // 2. compute 2-max in registers (64 max per warp)
        // 3. do warp reduction + broadcast back
        // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
        // 5. Repeat (3) 8 times for top 8 values in 256
        // 6. store with byte index
        group_load_uc(tmp).load(item_ct1, d_state1, c1s);
        
        
        item_ct1.barrier(sycl::access::fence_space::local_space);
        
        // 1. load 8 values per thread
        // 2. compute 2-max in registers (64 max per warp)
        // 3. do warp reduction + broadcast back
        // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
        // 5. Repeat (3) 8 times for top 8 values in 256
        // 6. store with byte index
        group_load_uc(tmp).load(item_ct1, d_state2, c2s);
        
     
        new_local_abs_max1 = -FLT_MAX;
        new_local_abs_max2 = -FLT_MAX;

        //  update: 2.48/1.57 -> 2.51/1.60
        # pragma unroll N_PER_TH
        for(unsigned int j = 0; j < N_PER_TH; j++)
        {
            if(!sycl::isnan((float)g_vals[j]) && !sycl::isinf((float)g_vals[j]))
						{
							s2_vals[j] = smem_quantiles2[lane_id][c2s[j]]*dacc_absmax2[i/BLOCK_SIZE];
              g_val = g_vals[j];
              //float ratio = (g_val*g_val)/fmaxf(s2_vals[j], eps*eps);
              //g_val = ratio > 2.0f ? 2.0f*g_val/ratio : g_val;
              g_val *= gnorm_scale;

							s2_vals[j] = (s2_vals[j]*beta2) + (((1.0f-beta2)*g_val*g_val));

							s1_vals[j] = smem_quantiles1[lane_id][c1s[j]]*dacc_absmax1[i/BLOCK_SIZE];
							s1_vals[j] = (s1_vals[j]*beta1) + (((1.0f-beta1)*g_val));
						}
            else
            {
              s1_vals[j] = 0.0f;
              s2_vals[j] = 0.0f;
            }

            new_local_abs_max1 = sycl::fmax(new_local_abs_max1, sycl::fabs(s1_vals[j]));
            new_local_abs_max2 = sycl::fmax(new_local_abs_max2, sycl::fabs(s2_vals[j]));
        }


        //  reduce: 2.51/1.60 -> 2.67/1.69
        new_local_abs_max1 = sycl::reduce_over_group(item_ct1.get_group(), new_local_abs_max1, sycl::maximum<>());
        new_local_abs_max2 = sycl::reduce_over_group(item_ct1.get_group(), new_local_abs_max2, sycl::maximum<>());

        if(item_ct1.get_local_id(2) == 0)
        {
          smem_exchange1[0] = new_local_abs_max1;
          smem_exchange2[0] = new_local_abs_max2;
        }

        
        item_ct1.barrier(sycl::access::fence_space::local_space);

        if(item_ct1.get_local_id(2) == 0)
        {
          dacc_absmax1[i/BLOCK_SIZE] = new_local_abs_max1;
          dacc_absmax2[i/BLOCK_SIZE] = new_local_abs_max2;
        }
        else
        {
          new_local_abs_max1 = smem_exchange1[0];
          new_local_abs_max2 = smem_exchange2[0];
        }

        
        item_ct1.barrier(sycl::access::fence_space::local_space);
        
          
        // 1. load 8 values per thread
        // 2. compute 2-max in registers (64 max per warp)
        // 3. do warp reduction + broadcast back
        // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
        // 5. Repeat (3) 8 times for top 8 values in 256
        // 6. store with byte index
        group_load(tmp).load(item_ct1, d_p, p_vals);
        
        //  reduce: 2.67/1.69 -> 2.67/1.70
        # pragma unroll N_PER_TH
        for(unsigned int j = 0; j < N_PER_TH; j++)
        {
						//if(!skip_zeros || (skip_zeros && ((float)g_vals[j] != 0.0f)))
            if(!sycl::isnan((float)g_vals[j]) && !sycl::isinf((float)g_vals[j]))
						{
							p_vals[j] = (T)(((float)p_vals[j]) + ((step_size*(s1_vals[j] / (sycl::sqrt(s2_vals[j])+(correction2*eps))))));
							if(weight_decay > 0.0f)
									p_vals[j] = ((float)p_vals[j])*(1.0f-(lr*weight_decay));
						}
        }

        //  store: 0.85/1.44 -> 2.48/1.57
        
        item_ct1.barrier(sycl::access::fence_space::local_space);
        
        
        // 1. load 8 values per thread
        // 2. compute 2-max in registers (64 max per warp)
        // 3. do warp reduction + broadcast back
        // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
        // 5. Repeat (3) 8 times for top 8 values in 256
        // 6. store with byte index
        group_store(tmp).store(item_ct1, d_p, p_vals);
        
        //  quantizaztion: 2.67/1.70  -> 3.4/3.3
        # pragma unroll N_PER_TH
        for(unsigned int j = 0; j < N_PER_TH; j++)
        {
            c1s[j] = quantize_2D<1>(quadrants1, s1_vals[j] / new_local_abs_max1);
            c2s[j] = quantize_2D<0>(quadrants2, s2_vals[j] / new_local_abs_max2);

            // make sure state1 term has still the same sign after quantization
            // (not needed for state2 term which has only positive values)
            if(sycl::signbit(smem_quantiles1[lane_id][c1s[j]]) != sycl::signbit(s1_vals[j]))
            {
              if(s1_vals[j] > 0.0f)
                  c1s[j] += 1;
              else
                  c1s[j] -= 1;
            }
        }

        
        item_ct1.barrier(sycl::access::fence_space::local_space);
        
        // 1. load 8 values per thread
        // 2. compute 2-max in registers (64 max per warp)
        // 3. do warp reduction + broadcast back
        // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
        // 5. Repeat (3) 8 times for top 8 values in 256
        // 6. store with byte index
        group_store_uc(tmp).store(item_ct1, d_state1, c1s);
        
       
        item_ct1.barrier(sycl::access::fence_space::local_space);
          
        // 1. load 8 values per thread
        // 2. compute 2-max in registers (64 max per warp)
        // 3. do warp reduction + broadcast back
        // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
        // 5. Repeat (3) 8 times for top 8 values in 256
        // 6. store with byte index
        group_store_uc(tmp).store(item_ct1, d_state2, c2s);
        
        item_ct1.barrier(sycl::access::fence_space::local_space);
        
    }
}


#define LANES 2
#define QUAD 3
template<typename T, int OPTIMIZER, int BLOCK_SIZE, int N_PER_TH>
SYCL_EXTERNAL 
void
kOptimizerStatic8bit1StateBlockwise(T* p, T* __restrict__ const g, unsigned char* state1,
                const float beta1, const float beta2,
                const float eps, const int step, const float lr,
                float* __restrict__ const quantiles1,
                float* absmax1,
                float weight_decay,
                const float gnorm_scale, const bool skip_zeros, const int n,
                const sycl::nd_item<3> &item_ct1,
                sycl::local_accessor<float, 2> smem_quantiles1,
                float *smem_exchange1,
                const sycl_la &tacc,
                const sycl::accessor<T, 1> &dacc_g,
                const sycl::accessor<T, 1> &dacc_p,
                const sycl_dacc_uc &dacc_state1,
                const sycl_dacc_float &dacc_quantiles1,
                const sycl_dacc_float &dacc_absmax1
                )
{

    //const int n_full = n + (n%BLOCK_SIZE);
    const int n_full = item_ct1.get_group_range(2) * BLOCK_SIZE;
    const int base_idx = (item_ct1.get_group(2) * BLOCK_SIZE);
    int valid_items = 0;
    float g_val = 0.0f;
    float s1_vals[N_PER_TH];
    // 2-5%
    const int lane_id = item_ct1.get_local_id(2) % LANES;
    float new_local_abs_max1 = -FLT_MAX;
    float quadrants1[QUAD];

    unsigned char c1s[N_PER_TH];
    T g_vals[N_PER_TH];
		T p_vals[N_PER_TH];

     using group_load = dpct::group::workgroup_load<N_PER_TH, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
  using group_load_uc = dpct::group::workgroup_load<N_PER_TH, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
  
  using group_store = dpct::group::workgroup_store<N_PER_TH, dpct::group::store_algorithm::BLOCK_STORE_DIRECT, T,  T *, sycl::nd_item<3>>;
  using group_store_uc = dpct::group::workgroup_store<N_PER_TH, dpct::group::store_algorithm::BLOCK_STORE_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
  
  
  auto *d_g = dacc_g.template get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *d_p = dacc_p.template get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *d_state1 = dacc_state1.get_multi_ptr<sycl::access::decorated::yes>().get();
    
   
    // init: 0.2 -> 0.23

    // 0.23 -> 0.23
		smem_quantiles1[0][item_ct1.get_local_id(2)] = dacc_quantiles1[item_ct1.get_local_id(2)];
		# pragma unroll
		for(unsigned int j = 1; j < LANES; j++)
			smem_quantiles1[j][item_ct1.get_local_id(2)] = smem_quantiles1[0][item_ct1.get_local_id(2)];

    
    item_ct1.barrier(sycl::access::fence_space::local_space);

    #pragma unroll
    for(int k = 0; k < QUAD; k++)
      quadrants1[k] = smem_quantiles1[lane_id][(k*256/(QUAD+1)) + (256/(QUAD+1)-1)];

    for (unsigned int i = base_idx; i < n_full; i += item_ct1.get_group_range(2)*BLOCK_SIZE)
    {
        // loads: 0.23 -> 0.85/1.44
        valid_items = n - i >= BLOCK_SIZE ? BLOCK_SIZE : n - i;
        
        item_ct1.barrier(sycl::access::fence_space::local_space);
        
        // 1. load 8 values per thread
        // 2. compute 2-max in registers (64 max per warp)
        // 3. do warp reduction + broadcast back
        // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
        // 5. Repeat (3) 8 times for top 8 values in 256
        // 6. store with byte index
        auto *tmp = tacc.get_multi_ptr<sycl::access::decorated::yes>().get();
        group_load(tmp).load(item_ct1, d_g, g_vals);
        
        /*
        DPCT1065:192: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
        
        // 1. load 8 values per thread
        // 2. compute 2-max in registers (64 max per warp)
        // 3. do warp reduction + broadcast back
        // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
        // 5. Repeat (3) 8 times for top 8 values in 256
        // 6. store with byte index
        group_load_uc(tmp).load(item_ct1, d_state1, c1s);
        
        
        item_ct1.barrier(sycl::access::fence_space::local_space);
        
        // 1. load 8 values per thread
        // 2. compute 2-max in registers (64 max per warp)
        // 3. do warp reduction + broadcast back
        // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
        // 5. Repeat (3) 8 times for top 8 values in 256
        // 6. store with byte index
        group_load(tmp).load(item_ct1, d_p, p_vals);
        
        new_local_abs_max1 = -FLT_MAX;

        //  update: 2.48/1.57 -> 2.51/1.60
        # pragma unroll N_PER_TH
        for(unsigned int j = 0; j < N_PER_TH; j++)
        {
            g_val = float(g_vals[j]);
            g_val *= gnorm_scale;
            if(!skip_zeros || (skip_zeros && ((float)g_vals[j] != 0.0f)))
            {
              if(weight_decay > 0.0f) {
                switch(OPTIMIZER) {
                  case 1:
                  case 3:
                  case 2:
                    g_val += ((float)p_vals[j])*weight_decay;
                    break;
                  case 4:
                    p_vals[j] = ((float)p_vals[j])*(1.0f-lr*weight_decay);
                    break;
                }
              }

							s1_vals[j] = smem_quantiles1[lane_id][c1s[j]]*dacc_absmax1[i/BLOCK_SIZE];

							switch(OPTIMIZER)
							{
									case 1:
										if(step == 1)
											s1_vals[j] = g_val;
										else
											s1_vals[j] = (s1_vals[j]*beta1) + g_val;
										break;
									case 4:
										// here, using gvals[j] to store the gradient smoothed by beta1 for the following parameter update, before the momentum is updated by beta2
										g_vals[j] = lr*sgn(((float)s1_vals[j])*beta1 + ((1.0f-beta1)*g_val));
										s1_vals[j] = s1_vals[j]*beta2 + ((1.0f-beta2)*g_val);
										break;
									case 2:
										s1_vals[j] = s1_vals[j]*beta1 + ((1.0f-beta1)*(g_val*g_val));
										break;
									case 3:
										s1_vals[j] = s1_vals[j] + (g_val*g_val);
										break;
							}
						}

            new_local_abs_max1 = sycl::fmax(new_local_abs_max1, sycl::fabs(s1_vals[j]));
        }


        //  reduce: 2.51/1.60 -> 2.67/1.69
        new_local_abs_max1 = sycl::reduce_over_group(item_ct1.get_group(), new_local_abs_max1, sycl::maximum<>());

        if(item_ct1.get_local_id(2) == 0)
          smem_exchange1[0] = new_local_abs_max1;

       
        item_ct1.barrier(sycl::access::fence_space::local_space);

        if(item_ct1.get_local_id(2) == 0)
          dacc_absmax1[i/BLOCK_SIZE] = new_local_abs_max1;
        else
          new_local_abs_max1 = smem_exchange1[0];

        //  reduce: 2.67/1.69 -> 2.67/1.70
        # pragma unroll N_PER_TH
        for(unsigned int j = 0; j < N_PER_TH; j++)
				{
						if(!skip_zeros || (skip_zeros && ((float)g_vals[j] != 0.0f)))
						{
							switch(OPTIMIZER)
							{
									case 1:
										p_vals[j] = ((float)p_vals[j]) - lr*(s1_vals[j]);
										break;
									case 4:
										p_vals[j] = ((float)p_vals[j]) - ((float)g_vals[j]);
										break;
									case 2:
										g_val = g_vals[j];
										p_vals[j] = ((float)p_vals[j]) - lr*(g_val / (sycl::sqrt(s1_vals[j])+eps));
										break;
									case 3:
										g_val = g_vals[j];
										p_vals[j] = ((float)p_vals[j]) - lr*(g_val / (sycl::sqrt(s1_vals[j])+eps));
										break;
							}
						}
				}

        //  store: 0.85/1.44 -> 2.48/1.57
        
        item_ct1.barrier(sycl::access::fence_space::local_space);
        
        // 1. load 8 values per thread
        // 2. compute 2-max in registers (64 max per warp)
        // 3. do warp reduction + broadcast back
        // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
        // 5. Repeat (3) 8 times for top 8 values in 256
        // 6. store with byte index
        group_store(tmp).store(item_ct1, d_p, p_vals);
        
        //  quantizaztion: 2.67/1.70  -> 3.4/3.3
        # pragma unroll N_PER_TH
        for(unsigned int j = 0; j < N_PER_TH; j++)
        {
            c1s[j] = quantize_2D<1>(quadrants1, s1_vals[j] / new_local_abs_max1);

            // make sure state1 term has still the same sign after quantization
            // (not needed for state2 term which has only positive values)
            if(sycl::signbit(smem_quantiles1[lane_id][c1s[j]]) != sycl::signbit(s1_vals[j]))
            {
              if(s1_vals[j] > 0.0f)
                  c1s[j] += 1;
              else
                  c1s[j] -= 1;
            }
        }

       
        item_ct1.barrier(sycl::access::fence_space::local_space);
        
        // 1. load 8 values per thread
        // 2. compute 2-max in registers (64 max per warp)
        // 3. do warp reduction + broadcast back
        // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
        // 5. Repeat (3) 8 times for top 8 values in 256
        // 6. store with byte index
        group_store_uc(tmp).store(item_ct1, d_state1, c1s);
        
    }
}

//==========================k get row col stats==========================================

template<typename T, int THREADS, int ITEMS_PER_THREAD, int TILE_ROWS, int TILE_COLS, int SPARSE_DECOMP> void kgetColRowStats(T * __restrict__ A, float *rowStats, float *colStats, int * nnz_count_row, float nnz_threshold, int rows, int cols, int tiledRows, int tiledCols,  const sycl::nd_item<3> &item_ct1, float *smem_row_absmax_values, int *smem_row_nnz_values, const sycl_la &tacc, const sycl::accessor<T, 1> &dacc_A)
{
  // 0. reset stats to -FLT_MAX
  // 1. load row-by-row ITEMS_PER_THREAD (TILE_SIZE==THREADS*ITEMS_PER_THREAD)
  // 2. compute col max (per thread); store in smem due to register pressure
  // 3. compute row max (per block); store in smem to accumulate full global mem transation
  // 4. store data via atomicMax

  // each block loads TILE_COLs columns and TILE_ROW rows
  // after reading a tile the row counter increase by TILE_ROWS
  // the col counter reset after reading TILE_COL elements
  const int base_row = ((item_ct1.get_group(2)*TILE_COLS)/tiledCols)*TILE_ROWS;
  // col increases by TILE_SIZE for each block and wraps back to 0 after tiledCols is reached
  const int base_col = (item_ct1.get_group(2)*TILE_COLS) % tiledCols;
  const int base_idx = (base_row*cols) + base_col;
  const int items_per_load = ITEMS_PER_THREAD*THREADS;

  
  using group_load = dpct::group::workgroup_load<ITEMS_PER_THREAD, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
  using group_exchange = exchange<float, ITEMS_PER_THREAD>;
  
  
  auto *d_A = dacc_A.template get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *tmp = tacc.get_multi_ptr<sycl::access::decorated::yes>().get();
     
  
  sycl::half local_data[ITEMS_PER_THREAD];
  float local_data_fp32[ITEMS_PER_THREAD];
  float local_col_absmax_values[ITEMS_PER_THREAD];
  int local_row_nnz_count = 0;
  float row_absmax = -FLT_MAX;

  // 0. reset stats to -FLT_MAX
  for(int j = 0; j < ITEMS_PER_THREAD; j++)
  {
    //smem_col_absmax_values[threadIdx.x + (j*THREADS)] = -FLT_MAX;
    smem_row_absmax_values[item_ct1.get_local_id(2) + (j*THREADS)] = -FLT_MAX;
    // smem_row_nnz_values[threadIdx.x + (j*THREADS)] = 0;
  }

  #pragma unroll TILE_ROWS
  for (int j = 0; j < TILE_ROWS; j++) {
    smem_row_nnz_values[j] = 0;
  }

  #pragma unroll ITEMS_PER_THREAD
  for(int j = 0; j < ITEMS_PER_THREAD; j++)
    local_col_absmax_values[j] = -FLT_MAX;

  item_ct1.barrier(sycl::access::fence_space::local_space);

  int valid_items = cols - base_col > items_per_load ? items_per_load : cols - base_col;
  int i = base_idx;
  // we load row after row from the base_position
  // 1. load row-by-row ITEMS_PER_THREAD (TILE_SIZE==THREADS*ITEMS_PER_THREAD)
  for(int row = 0; row < TILE_ROWS; row++)
  {
    if(base_row+row >= rows){ break; }
    local_row_nnz_count = 0;
    i = base_idx + ((row)*cols);
    // each thread gets data from the same column
    
    item_ct1.barrier(sycl::access::fence_space::local_space);
    
    
    // 1. load 8 values per thread
    // 2. compute 2-max in registers (64 max per warp)
    // 3. do warp reduction + broadcast back
    // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
    // 5. Repeat (3) 8 times for top 8 values in 256
    // 6. store with byte index
    group_load(tmp).load(item_ct1, d_A, local_data);
    
    #pragma unroll ITEMS_PER_THREAD
    for(int j = 0; j < ITEMS_PER_THREAD; j++)
      local_data[j] = sycl::fabs(local_data[j]);


    if(SPARSE_DECOMP)
      #pragma unroll ITEMS_PER_THREAD
      for(int j = 0; j < ITEMS_PER_THREAD; j++)
      {
        if((float)local_data[j] >= nnz_threshold)
        {
          local_row_nnz_count += 1;
          local_data[j] = 0.0f;
        }
      }

    // 2. compute col max (per thread); store in smem due to register pressure
    #pragma unroll ITEMS_PER_THREAD
    for(int j = 0; j < ITEMS_PER_THREAD; j++)
      // take the col max for this row
      // we use shared memory because register pressure is too high if we do this locally
      //smem_col_absmax_values[threadIdx.x + (j*THREADS)] = fmaxf(smem_col_absmax_values[threadIdx.x + (j*THREADS)], __half2float(local_data[j]));
      local_col_absmax_values[j] = sycl::fmax(local_col_absmax_values[j], sycl::vec<sycl::half, 1>(local_data[j]).convert<float, sycl::rounding_mode::automatic>()[0]);

    // 3. compute row max (per block); store in smem to accumulate full global mem transation

    // this is slow as it uses extra registers, but we need this to be compatible with Kepler and Maxwell (no fp16 units)
    #pragma unroll ITEMS_PER_THREAD
    for(int j = 0; j < ITEMS_PER_THREAD; j++)
      local_data_fp32[j] = local_data[j];

   
    item_ct1.barrier(sycl::access::fence_space::local_space);

    row_absmax = (float)sycl::reduce_over_group(item_ct1.get_group(), local_data_fp32[0], sycl::maximum<>());
    
    if(SPARSE_DECOMP)
    {
      
      item_ct1.barrier(sycl::access::fence_space::local_space);
      local_row_nnz_count =  sycl::reduce_over_group(item_ct1.get_group(), local_row_nnz_count, sycl::plus<>());
    }
    // we store the data temporarily in shared memory so we
    // can execute a full atomic block transaction into global memory later
    // we use a striped arrangement [0, 8, 16, 24, ..] for t0 for faster stores
    if(item_ct1.get_local_id(2) == 0)
    {
      smem_row_absmax_values[(row % ITEMS_PER_THREAD) + ((row/ITEMS_PER_THREAD)*ITEMS_PER_THREAD)] = row_absmax;
      // each blockIdx.x process 16 rows and 64*4=256 columns -> we sum nnz over 256 columns and have 16 values per block
      smem_row_nnz_values[row] = local_row_nnz_count;
    }

   
    item_ct1.barrier(sycl::access::fence_space::local_space);

  }

  // 4. store data via atomicMax
  // to store col data efficiently we need to rewrite the smem blocked data [0, 1, 2, 3...] for t0
  // into a striped arrangement: [0, 8, 16, 24, ..] for t0
  
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // 1. load 8 values per thread
  // 2. compute 2-max in registers (64 max per warp)
  // 3. do warp reduction + broadcast back
  // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
  // 5. Repeat (3) 8 times for top 8 values in 256
  // 6. store with byte index
  group_exchange(tmp).blocked_to_striped(item_ct1, local_col_absmax_values);
  
  #pragma unroll ITEMS_PER_THREAD
  for(int j = 0; j < ITEMS_PER_THREAD; j++)
    if(base_col+item_ct1.get_local_id(2)+(j*THREADS) < cols)
    {
      float val = colStats[base_col+(item_ct1.get_local_id(2)+(j*THREADS))];
      if(val < local_col_absmax_values[j])
        atomicMax(&colStats[base_col+(item_ct1.get_local_id(2)+(j*THREADS))], local_col_absmax_values[j]);
    }

  for(int j = 0; j < ITEMS_PER_THREAD; j++)
    if(base_row+item_ct1.get_local_id(2)+(j*THREADS) < rows)
    {
      float val = rowStats[base_row+(item_ct1.get_local_id(2)+(j*THREADS))];
      if(val < smem_row_absmax_values[item_ct1.get_local_id(2)+(j*THREADS)])
        atomicMax(&rowStats[base_row+(item_ct1.get_local_id(2)+(j*THREADS))], smem_row_absmax_values[item_ct1.get_local_id(2)+(j*THREADS)]);
    }

    if(SPARSE_DECOMP)
      if(item_ct1.get_local_id(2) < TILE_ROWS)
        nnz_count_row[item_ct1.get_group(2)*TILE_ROWS+item_ct1.get_local_id(2)+1] = smem_row_nnz_values[item_ct1.get_local_id(2)];

}


//=====================================k double row col quant============================

template <int THREADS, int ITEMS_PER_THREAD, int TILE_ROWS, int TILE_COLS, int SPARSE_DECOMP> void kDoubleRowColQuant(sycl::half *__restrict__ const A, float *__restrict__ const rowStats, float * __restrict__ const colStats, char *out_col_normed, char *out_row_normed, int *rowidx, int *colidx, sycl::half *val, int * __restrict__ nnz_block_ptr, float threshold, int rows, int cols, int tiledCols, const sycl::nd_item<3> &item_ct1, float *smem_row_stats, unsigned int *smem_nnz_row_idx, const sycl_la &tacc, const sycl::accessor<sycl::half, 1> &dacc_A, const sycl_dacc_char &dacc_out_col_normed, const sycl_dacc_char &dacc_out_row_normed, const sycl_dacc_float &dacc_rowStats, const sycl_dacc_float &dacc_colStats, const sycl_dacc &dacc_rowidx, const sycl_dacc &dacc_colidx, const sycl::accessor<sycl::half, 1> &dacc_val, const sycl_dacc &dacc_nnz_block_ptr)
{
  // assumes TILE_SIZE == THREADS*ITEMS_PER_THREAD
  // Each thread reads the same column but multiple rows
  // Rows are loaded in shared memory and access is shared across the threadblock (broadcast)

  // 0. Load row stats data into shared memory; load col stat (1 fixed per thread)
  // 1. Load data row by row (should be at least with TILE_SIZE = 512)
  // 2. quantize data with row/col stats
  // 3. Store data (TILE_SIZE = 512 is a bit slow, but should still be close enough to good performance)

  // each block loads TILE_COLs columns and TILE_ROW rows
  // after reading a tile the row counter increase by TILE_ROWS
  // the col counter reset after reading TILE_COL elements
  const int base_row = ((item_ct1.get_group(2)*TILE_COLS)/tiledCols)*TILE_ROWS;
  // col increases by TILE_SIZE for each block and wraps back to 0 after tiledCols is reached
  const int base_col = (item_ct1.get_group(2)*TILE_COLS) % tiledCols;
  const int base_idx = (base_row*cols) + base_col;
  const int items_per_load = ITEMS_PER_THREAD*THREADS;
  //colStats ,rowStats rowidx,  colidx,val ,nnz
  
  using group_load_half = dpct::group::workgroup_load<ITEMS_PER_THREAD, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, sycl::half,  sycl::half *, sycl::nd_item<3>>;
  using group_store_char = dpct::group::workgroup_store<ITEMS_PER_THREAD, dpct::group::store_algorithm::BLOCK_STORE_DIRECT, char,  char *, sycl::nd_item<3>>;
  
  auto *d_A = dacc_A.get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *d_out_col_normed = dacc_out_col_normed.get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *d_out_row_normed = dacc_out_row_normed.get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *tmp = tacc.get_multi_ptr<sycl::access::decorated::yes>().get();
  
  sycl::half local_data[ITEMS_PER_THREAD];
  float local_col_stats[ITEMS_PER_THREAD];
  char local_quantized_data[ITEMS_PER_THREAD];

  // 0. Load row stats data into shared memory; load col stat (1 fixed per thread)
  #pragma unroll ITEMS_PER_THREAD
  for(int j = 0; j < ITEMS_PER_THREAD; j++)
    if(base_col+(item_ct1.get_local_id(2)*ITEMS_PER_THREAD) + j < cols)
      /*
       To-do: __fdividef call is used in a macro/template definition and may not be valid for all macro/template uses.
      */
      local_col_stats[j] = 127.0f / dacc_colStats[base_col+(item_ct1.get_local_id(2)*ITEMS_PER_THREAD)+j];

  for(int i = item_ct1.get_local_id(2); i < TILE_ROWS; i+=item_ct1.get_local_range(2))
  {
    if(base_row + i < rows)
      smem_row_stats[i] = dacc_rowStats[base_row+i];

    if(SPARSE_DECOMP)
      smem_nnz_row_idx[i] = dacc_nnz_block_ptr[(TILE_ROWS*item_ct1.get_group(2)) + i];
  }
  
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // we load row after row from the base_position
  // 1. Load data row by row (should be at least with TILE_SIZE = 512)
  for(int row = 0; row < TILE_ROWS; row++)
  {
    if(base_row + row >= rows){ break; }
    int i = base_idx + (row*cols);
    int valid_items = cols - base_col > items_per_load ? items_per_load : cols - base_col;

    // 1. load 8 values per thread
    // 2. compute 2-max in registers (64 max per warp)
    // 3. do warp reduction + broadcast back
    // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
    // 5. Repeat (3) 8 times for top 8 values in 256
    // 6. store with byte index
    group_load_half(tmp).load(item_ct1, d_A, local_data);

    float row_stat = 127.0f / smem_row_stats[row];

    // 2. quantize data with row/col stats
    #pragma unroll ITEMS_PER_THREAD
    for(int j = 0; j < ITEMS_PER_THREAD; j++)
    {
      // we already pre-normalized the col/row stat:
      // what this does is float/absmax*127 = int8
      if(SPARSE_DECOMP)
      {
        if(sycl::fabs((float)local_data[j]) >= threshold)
        {
          local_quantized_data[j] = 0;

					int old_idx = dpct::atomic_fetch_compare_inc<sycl::access::address_space::generic_space>(&smem_nnz_row_idx[row], UINT_MAX);

          dacc_rowidx[old_idx] = base_row+row;
          dacc_colidx[old_idx] = base_col+(item_ct1.get_local_id(2)*ITEMS_PER_THREAD)+j;
          dacc_val[old_idx] = local_data[j];
        }
				else
				{
					local_quantized_data[j] = (char)(sycl::rint(sycl::vec<sycl::half, 1>(local_data[j]).convert<float, sycl::rounding_mode::automatic>()[0]*row_stat));
				}
      }
      else
        local_quantized_data[j] = (char)(sycl::rint(sycl::vec<sycl::half, 1>(local_data[j]).convert<float, sycl::rounding_mode::automatic>()[0]*row_stat));
    }

    // 1. load 8 values per thread
    // 2. compute 2-max in registers (64 max per warp)
    // 3. do warp reduction + broadcast back
    // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
    // 5. Repeat (3) 8 times for top 8 values in 256
    // 6. store with byte index
    group_store_char(tmp).store(item_ct1, d_out_row_normed, local_quantized_data);
    
    // 2. quantize data with row/col stats
    #pragma unroll ITEMS_PER_THREAD
    for(int j = 0; j < ITEMS_PER_THREAD; j++)
    {
      // we already pre-normalized the col/row stat:
      // what this does is float/absmax*127 = int8
			local_quantized_data[j] = (char)(sycl::rint(sycl::vec<sycl::half, 1>(local_data[j]).convert<float, sycl::rounding_mode::automatic>()[0]*local_col_stats[j]));
    }

    
    item_ct1.barrier(sycl::access::fence_space::local_space);
   
    //StoreInt8(storeint8).Store(&(out_col_normed[i]), local_quantized_data, valid_items);
    // 1. load 8 values per thread
    // 2. compute 2-max in registers (64 max per warp)
    // 3. do warp reduction + broadcast back
    // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
    // 5. Repeat (3) 8 times for top 8 values in 256
    // 6. store with byte index
    group_store_char(tmp).store(item_ct1, d_out_col_normed, local_quantized_data);
      
  }
}

//============================================k transform row format=====================================================

template <int THREADS, int ITEMS_PER_THREAD, int TILE_ROWS, int TILE_COLS, int TRANSPOSE, int FORMAT> SYCL_EXTERNAL void kTransformRowToFormat(char *__restrict__ const A, char *out, int rows, int cols, int tiledCols, int outRows, int outCols,  const sycl::nd_item<3> &item_ct1, char *smem_data,
const sycl_dacc_char &dacc_A, const sycl_dacc_char &dacc_out)
{

  // 0. Load data into 32*32 shared memory tiles
  // 1. transpose / reorder in shared memory
  // 2. store

  // COL32 FORMAT:
  // rows*32 tiles

  // TURING FORMAT:
  // 8*32 tiles with 4*4 subtiles
  // the 8*32 subtile has first all 4*4 subtiles of even rows (max 4*4*4 = 64 elements)
  // the subsequent 4*4 subtiles are for all odd rows if some rows columns are empty the values are zero
  // the tile repeats again after the 8*32 tile in a major column order, meaning: (next 8 rows are A[8:16, 0:32])
  // the next tile is the next 8 rows for the same 32 columns. Once all rows are finished, the column
  // index increases by 32

  // AMPERE FORMAT:
  // 32*32 tiles with 8*32 subtiles. The rows are interleaved in pairs of two rows with offset of 8 between pairs of two rows:
	// row idx (each number stands for 32 values): [0 1 8 9 16 17 24 25] [2 3 10 11 18 19 26 27]...
  // the tiles are column-major ordered, so after 1024*1024 values we process: A[32:64, 0:32]


  // To have efficient loads and stores if we transpose we need 128 consequitive bytes which at 1 byte are 128 values
  // As such we need:
  // at least 32*4 shared memory tiles for col32; preferably 32*32
  // at least 32*6 shared memory tiles for col32_ampere: preferably 32*32
  // at least 32*8 shared memory tiles for col4_turing: preferably 32*32
  // for efficient loading of row major we need to load 128 elements and repeat this 32 items
  // this would imply a 32x128 shared memory tile -> 4kb
  // It is more efficient to have more than 1 warp, so with 64 threads we need 32x128 -> 8 kb
  // we have 64k sharded mem per SM in Turing which is 8 blocks per SM which is 2*8 = 32 warps = 100% occupancy
  // for turing and 50% for A100 and 75% for RTX 30s / A40 which is probably good enough
  // register pressure should be low with: 8 registers from local memoryh per block and 64 registers per SM
  //
  // to make the shared memory work with that occupancy we might need to union the block loads/stores

  // each block loads TILE_COLs columns and TILE_ROW rows
  // after reading a tile the row counter increase by TILE_ROWS
  // the col counter reset after reading TILE_COL elements
  const int base_row = ((item_ct1.get_group(2)*TILE_COLS)/tiledCols)*TILE_ROWS;
  // col increases by TILE_SIZE for each block and wraps back to 0 after tiledCols is reached
  const int base_col = (item_ct1.get_group(2)*TILE_COLS) % tiledCols;
  const int base_idx = (base_row*cols) + base_col;

  // we load 128 bytes per warp with
  // 32 rows for transposes that fill col32 types
  // so that we can have contiguous stores
  
  char local_data[ITEMS_PER_THREAD];
  
  // we load row after row from the base_position
  // Load data row by row
  int warps = item_ct1.get_local_range(2)/32;
  int warp_id = item_ct1.get_local_id(2)/32;
  int warp_lane = item_ct1.get_local_id(2) % 32;
  int offset = 0;

  int smem_row = 0;
  // each warp loads one row of 128 bytes
  for(int row = warp_id; row < TILE_ROWS; row+=warps)
  {
    int i = base_idx + (row*cols);
    // we load up to 128 bytes/items per load
    int valid_items = cols - base_col > 32*ITEMS_PER_THREAD ? 32*ITEMS_PER_THREAD : cols - base_col;

    // 0. Load data into 32*32 shared memory tiles
    if(base_row + row < rows)
    {
      #pragma unroll ITEMS_PER_THREAD
      for(int j = 0; j < ITEMS_PER_THREAD; j++)
      {
        int col_idx = warp_lane+(j*32);
        if(col_idx < valid_items)
          local_data[j] = dacc_A[i+col_idx];
        else
          local_data[j] = 0;
      }
    }
    else
    {
      #pragma unroll ITEMS_PER_THREAD
      for(int j = 0; j < ITEMS_PER_THREAD; j++)
        local_data[j] = 0;
    }

    if(TRANSPOSE)
    {
      #pragma unroll ITEMS_PER_THREAD
      for(int j = 0; j < ITEMS_PER_THREAD; j++)
      {
        int local_col = (32*j)+warp_lane;
        //int local_row = row;
        // store as 256x32
        smem_data[(local_col*33) + row] = local_data[j];
      }
    }
    else
    {
      // treat smem as 32x256, that is 32 rows and 256 columns
      #pragma unroll ITEMS_PER_THREAD
      for(int j = 0; j < ITEMS_PER_THREAD; j++)
        smem_data[row*32*ITEMS_PER_THREAD + (warp_lane) + (j*32)] = local_data[j];
    }



    smem_row += warps;

    // 1. transpose / reorder in shared memory
    if(smem_row % 32 == 0)
    {
      smem_row = 0;
      item_ct1.barrier(sycl::access::fence_space::local_space);

      for(int subrow = warp_id; subrow < 32; subrow+=warps)
      {
        for(int j = 0; j < ITEMS_PER_THREAD; j++)
        {

          switch(FORMAT)
          {
              case COL32:
                if(TRANSPOSE)
                {
                  // data lies in shared memory in the following way:
                  // row0 [col0 col1 ... col31]
                  // row1 [col0 col1 ... col31]
                  // ...
                  //
                  // As such we read consecutive entries with 256 threads (8rows x 32 columns)
                  // as j increase, the row increase by a factor of 8
                  // We load 8 rows per subrow loop, and subrow increase by 8 per loop
                  // so we have an offset of 8 rows every loop or (subrow/warps)*8 = (subrow/8)*8
                  const int jrow = j*ITEMS_PER_THREAD; // 8 rows per j
                  const int subrow_loop_row = (subrow/warps)*ITEMS_PER_THREAD*ITEMS_PER_THREAD; // 8 rows per j; 8j per subrow loop (subrow/warps)
                  //const int local_row =  warp_id; // each warp_id is one row
                  //const int block_row = base_col; // block offset for row
                  //const int local_col = warp_lane
                  //const int global_col = base_row; // block offset for col
                  if((base_col + subrow_loop_row + jrow + warp_id < outRows) && (base_row+warp_lane < rows))
                  {
                    // each row hae 32 columns and is offset by 1 to prevent bank conflict during storage into smem
                    char data = smem_data[(subrow_loop_row + jrow + warp_id)*33 + warp_lane];

                    // each 32 columns we have new tile
                    // each tile has size outRows*32 and base_row is done in increments of 32
                    offset = base_row*outRows;
                    dacc_out[offset + (base_col + jrow + subrow_loop_row)*32 + item_ct1.get_local_id(2)] = data;
                  }
                }
                else
                {
                  if(((base_row+subrow) < rows) && (base_col+(j*32)+warp_lane < outCols))
                  {
                    offset = (base_col/32)*(32*rows);
                    char data = smem_data[(subrow*32*ITEMS_PER_THREAD) + (j*32) + warp_lane];
                    dacc_out[offset+(base_row+subrow)*32 + ((j)*rows*32)+warp_lane] = data;
                  }
                }
                break;
              case COL_TURING:
                // TURING FORMAT:
                // 8*32 tiles with 4*4 subtiles
                // the 8*32 subtile has first all 4*4 subtiles of even rows (max 4*4*4 = 64 elements)
                // the subsequent 4*4 subtiles are for all odd rows if some rows columns are empty the values are zero
                // the tile repeats again after the 8*32 tile in a major column order, meaning: (next 8 rows are A[8:16, 0:32])
                // the next tile is the next 8 rows for the same 32 columns. Once all rows are finished, the column
                // index increases by 32
                //
                // [0 0 0 0, 2 2 2 2, 4 4 4 4, 6 6 6 6, 0 0 0 0 ...]
                if(TRANSPOSE)
                {
                  const int jrow = j*ITEMS_PER_THREAD; // 8 rows per j
                  const int subrow_loop_row = (subrow/warps)*ITEMS_PER_THREAD*ITEMS_PER_THREAD; // 8 rows per j; 8j per subrow loop (subrow/warps)
                  //const int local_row =  warp_id; // each warp_id is one row
                  //const int block_row = base_col; // block offset for row
                  //const int local_col = warp_lane
                  //const int global_col = base_row; // block offset for col
                  if((base_col + subrow_loop_row + jrow + warp_id < outRows) && (base_row+warp_lane < rows))
                  {
                    // each row hae 32 columns and is offset by 1 to prevent bank conflict during storage into smem
                    char data = smem_data[(subrow_loop_row + jrow + warp_id)*33 + warp_lane];

                    // each 32 columns we have new tile
                    // each tile has size 8*32 = 256 elements offset
                    // for each row offset of 8 we increaes the tile first
                    // after all rows are exhausted, we increase the col
                    int row_offset = ((base_col+jrow+subrow_loop_row+warp_id)/8)*256; // global_row+jrow+subrow_loop_row+local_row, increase tile(=256) every 8 rows

                    // we increase by row_tile_column every 32 columns
                    // base_row increase in increments of 32
                    //int row_tile_column = 256*outRows/8; // there are outRows/8 row tiles, and each tile is 256 elements
                    //int col_offset = (base_row/32)*row_tile_column;
                    // -> we can remove the divisions to speed up compute since outRows is always a multiple of 8
                    // 256*outRows/8*base_row/32 = outRows*base_row
                    int col_offset = outRows*base_row;

                    offset = row_offset+col_offset;

                    // since we process even number of rows with each j (8) and with each subrow (8j) we can determine
                    // odd or even rows with the warp_id (each warp processes one row)
                    // the col is warp_lane (max 32 columns per row) and the row warp_id
                    if(warp_id % 2 == 1)
                      // odd
                      offset += 128 + (warp_lane/4)*16 + (warp_lane%4) + (((warp_id%8)-1)*2);
                    else
                      // even
                      offset += 0   + (warp_lane/4)*16 + (warp_lane%4) + ((warp_id%8)*2);

                    dacc_out[offset] = data;
                  }
                }
                else
                {
                  if(((base_row+subrow) < rows) && (base_col+(j*32)+warp_lane < outCols))
                  {
                    char data = smem_data[(subrow*32*ITEMS_PER_THREAD) + (j*32) + warp_lane];
                    // set offset designates the tile offset among the 8*32 tiles
                    // we first increase rows and then columns. Since we load 128 columns at once
                    // we increase the offset by outRows*32 every 32 columns
                    // additionally, we increase the offset by 8*32=256 every 8 rows
                    offset = ((base_col+(j*32))/32)*outRows*32 + (((base_row+subrow)/8)*256); // global offset (8x32 tile)
                    // first 4 rows are reserved for even rows, [0, 2, 4, 6], the next 4 for odd
                    // each of these has 32 values in total for 32*4 = 128 as offset if odd
                    // every set of 4 columns increases the total offset by 16
                    // each even row increase the offset by 4, for example row 2 is offset by 4, 4 by 6 etc so: subrow/2*4 = subrow*2
                    // this happens every 8 rows anew (subrow % 8)
                    // one writes 4 columns at once that is (col % 4) for the particular index in the subtile
                    int subcol = warp_lane;

                    // add local offset (4x4 sub-tile)
                    if(subrow % 2 == 1)
                      // odd
                      offset += 128 + (subcol/4)*16 + (subcol%4) + (((subrow%8)-1)*2);
                    else
                      // even
                      offset += 0   + (subcol/4)*16 + (subcol%4) + ((subrow%8)*2);

                    dacc_out[offset] = data;
                  }
                }
                break;
								case COL_AMPERE:
									// AMPERE FORMAT:
									// 32*32 tiles with 8*32 subtiles. The rows are interleaved in pairs of two rows with offset of 8 between pairs of two rows:
									// row idx (each number stands for 32 values): [0 1 8 9 16 17 24 25] [2 3 10 11 18 19 26 27]...
									// the tiles are column-major ordered, so after 1024*1024 values we process: A[32:64, 0:32]
									if(TRANSPOSE)
									{
										const int jrow = j*ITEMS_PER_THREAD; // 8 rows per j
										const int subrow_loop_row = (subrow/warps)*ITEMS_PER_THREAD*ITEMS_PER_THREAD; // 8 rows per j; 8j per subrow loop (subrow/warps)
										//const int local_row =  warp_id; // each warp_id is one row
										//const int block_row = base_col; // block offset for row
										//const int local_col = warp_lane
										//const int global_col = base_row; // block offset for col
										if((base_col + subrow_loop_row + jrow + warp_id < outRows) && (base_row+warp_lane < rows))
										{
											// each row hae 32 columns and is offset by 1 to prevent bank conflict during storage into smem
											char data = smem_data[(subrow_loop_row + jrow + warp_id)*33 + warp_lane];

											// each 32 columns we have new tile
											// each tile has size 32*32 = 1024 elements offset
											// for each row offset of 32 we increaes the tile first
											// after all rows are exhausted, we increase the col
											int row_offset = ((base_col+jrow+subrow_loop_row+warp_id)/32)*1024; // global_row+jrow+subrow_loop_row+local_row, increase tile(=256) every 8 rows

											// we increase by row_tile_column every 32 columns
											// base_row increase in increments of 32
											//int row_tile_column = 1024*outRows/32; // there are outRows/32 row tiles, and each tile is 1024 elements
											//int col_offset = (base_row/32)*row_tile_column;
											// -> we can remove the divisions to speed up compute since outRows is always a multiple of 8
											// 1024*outRows/32*base_row/32 = outRows*base_row
											int col_offset = outRows*base_row;

											offset = row_offset+col_offset;


											// same as in the non-transpose case (see below)
											// the difference is that now rows = cols
											// in this case warp_id = subrow

											// [0 1 8 9 16 17 24 25] [2 3 10 11 18 19 26 27]...
											// subrow % 8 -> [0,1] in tile0, [2, 3] in tile 1 etc
											// subrow % 2 -> 0 for 1st row in the pair, 1 for the 2nd row
											// every 2 rows, the offset increases by two [0, 1, 8, 9...]
											// every 2 rows, the row index increase by 8 [0, 1, 8, 9...]
											int local_row = (jrow + warp_id) % 32; // offset for row > 32 is already calculated into row_offset
											int ampere_row = ((local_row % 8)/2)*8 + (local_row/8)*2 + (local_row % 2);

											// global offset + row with 32 cols each + 32 cols per j + col_idx=warp_lane
											dacc_out[offset + (ampere_row*32) + warp_lane] = data;
										}
									}
									else
									{
										if(((base_row+subrow) < rows) && (base_col+(j*32)+warp_lane < outCols))
										{
											char data = smem_data[(subrow*32*ITEMS_PER_THREAD) + (j*32) + warp_lane];

											// set offset designates the tile offset among the 32*32 tiles
											// we first increase rows and then columns. Since we load 128 columns at once
											// we increase the offset by outRows*32 every 32 columns
											// additionally, we increase the offset by 32*32=1024 every 32 rows
											offset = ((base_col+(j*32))/32)*outRows*32 + (((base_row+subrow)/32)*1024); // global offset (32x32 tile)

											// [0 1 8 9 16 17 24 25] [2 3 10 11 18 19 26 27]...
											// subrow % 8 -> [0,1] in tile0, [2, 3] in tile 1 etc
											// subrow % 2 -> 0 for 1st row in the pair, 1 for the 2nd row
											// every 2 rows, the offset increases by two [0, 1, 8, 9...]
											// every 2 rows, the row index increase by 8 [0, 1, 8, 9...]
											int local_row = ((subrow % 8)/2)*8 + (subrow/8)*2 + (subrow % 2);

											// global offset + row with 32 cols each + 32 cols per j + col_idx
											dacc_out[offset + (local_row*32) + warp_lane] = data;
										}
									}
								break;
          }
        }
      }
    }
  }
}
 
//========================================k dequant mm int32fp16===================


#define MM_DEQUANT_CONST 6.200012e-05f //1.0f/(127.0f*127.0f)

template <int ITEMS_PER_THREAD, int SUBTILE_ROWS, int THREADS>void kdequant_mm_int32_fp16(int *__restrict__ const A, float *__restrict__ const rowStats, float *__restrict__ const colStats, sycl::half *out, float* newRowStats, float* newcolStats, sycl::half *__restrict__ const bias, const int numRows, const int numCols, const int tileCols, const int n,  const sycl::nd_item<3> &item_ct1, float *smem_rowStats, const sycl_la &tacc, const sycl_dacc &dacc_A)
{

  // Strategy: To dequantize we need to load col/row statistics. This can be very expensive
  // since different row/col stats need to be loaded with each thread.
  // (1, bad algorithm) Loading 32 items per thread would only occur 1 row load, but this increases register pressure
  // and would lead to low global load utilization.
  // (2, bad algorithm) If each thread loads some columns and multiple rows one needs to do lot of row loads
  // for each thread and this is duplicated by a factor of 32/num-cols-per-thread.
  // (3, good algorithm) Combining (1) and (2) we use sub-tiles of size 32xk in shared memory per threadblock.
  // This allows for efficient row/col loading from shared memory within the tile.
  // We can run for example 32x128 sub-tiles and warp-strided loads of 4 elements so that each thread has
  // the same col statistic but needs to load 4 row stats from shared memory. To prevent bank conflicts
  // we use a block-striped shared memory config [1, 31, 63, 95] so no bank conflicts happen during the
  // shared memory loads.

  // data is in 32 column-tile major with tile width 32 columns and numRows rows
  // L1. Load sub-tile row/col statistics. Each thread only holds 1 col, load rows into shared memory.
  // L2. Load data in warp-striped arrangement (t0 holds colidx [0, 0, 0, 0], rowidx [0, 1, 2, 3])
  // C1. Compute val(row_stat*col_stat)/(127*127) (load 1/(127*127 into register))
  // C2. Compute normalization values and store col values in register
  // S1. Store C1 into 16-bit output
  // S2. Store col/row statistics of new buffer in shared memory

  // We allow for sub-tiles to span multiple col32 tiles. This is okay
  // since the items per thread only rely on a single column statistic.


  const int n_out = numRows*numCols;

  int num_row_tiles = (numRows/SUBTILE_ROWS) + (numRows % SUBTILE_ROWS == 0 ? 0 : 1);
  // we have tiles of size numRows*32, thus col only increases every numRows
  // num_row_tiles is the tiles after which the column increases by 32
  // blockIdx.x is the index of the current tile
  int col = ((item_ct1.get_local_id(2) % 32) + ((item_ct1.get_group(2)/num_row_tiles)*32));
  // base_row increases by SUBTILE_ROWS every block. It wraps back to zero once num_row_tiles is reached
  int base_row = (item_ct1.get_group(2)*SUBTILE_ROWS) % (num_row_tiles*SUBTILE_ROWS);

  // SUBTILE_ROWS is independent from ITEMS_PER_THREAD is independent from THREADS
  // subtiles have 32*SUBTILE_ROWS elements <= THREADS*ITEMS_PER_THREAD
  // Total subtiles should be n/(32*SUBTILE_ROWS) where each subtile has SUBTILE_ROW*32/4 threads.
  // For example for a 1024x1024 matrix with 128 SUBTILE_ROWS and 4 ITEMS_PER_THREAD we have
  // 1024*1024/(128*32) = 256 tiles
  // 256 tiles are 256*128*32/4 = 256*1024 threads

  // 1. Figure out how index relates to the start of the sub-tile
  // 2. Each thread < SUBTILE_ROWS calculates row index
  // 3. Load striped and store in shared memory

  int local_values[ITEMS_PER_THREAD];
  sycl::half local_output[ITEMS_PER_THREAD];
  float local_rowStats[ITEMS_PER_THREAD];
  
  using group_load_int = dpct::group::workgroup_load<ITEMS_PER_THREAD, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, int,  int *, sycl::nd_item<3>>;
  using group_exchange = exchange<int, ITEMS_PER_THREAD>;
  
  auto *d_A = dacc_A.get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *tmp = tacc.get_multi_ptr<sycl::access::decorated::yes>().get();
    

  // L1. Load sub-tile row/col statistics. Each thread only holds 1 col, load rows into shared memory.
  float colStat = col >= numCols ? 0.0f : colStats[col];
  float local_biasValue = ((bias == NULL) || (col >= numCols)) ? 0.0f : sycl::vec<sycl::half, 1>(bias[col]).convert<float, sycl::rounding_mode::automatic>()[0];
  // no block loads for rows for now -- keep it simple
  for(int j = item_ct1.get_local_id(2); j < SUBTILE_ROWS; j+=item_ct1.get_local_range(2))
  {
    // todo: is this global mem access slow due to overlaps or does the L1 cache work well here?
    int row = (base_row+j) % numRows; // wrap around
    // each warp accesses the same element, for four consequitive elements
    // todo: update description about striped shared memory, it is not needed
    // rowidx: [0, 1, 2, 3...] and each warp reads ITEMS_PER_THREAD consequitive elements
    smem_rowStats[j] = rowStats[row];
  }
  
  item_ct1.barrier(sycl::access::fence_space::local_space);


  // each block processes SUBTILE_ROWS*32 elements
  const int items_per_load = THREADS*ITEMS_PER_THREAD;
  const int rows_per_load = items_per_load/32;

  int subtile_base_row = (item_ct1.get_local_id(2) / 32)*ITEMS_PER_THREAD; // row within the tile
  int row_offset = 0;
  // subtile_idx starts at the base_row*32 + the total offset for a full numRow*32 tile is passed
  int subtile_start = (item_ct1.get_group(2)/num_row_tiles)*(numRows*32) + (base_row*32);
  for(int subtile_idx = subtile_start; subtile_idx < subtile_start + (SUBTILE_ROWS*32); subtile_idx+=items_per_load)
  {
    int valid_rows = numRows - (base_row+row_offset) > rows_per_load ? rows_per_load : numRows - (base_row+row_offset);
    int valid_items = valid_rows*32;
    if(valid_items <= 0) // the sub-tile might have more elements than the tile itself
      break;

    // L2. Load data in warp-striped arrangement (t0 holds colidx [0, 0, 0, 0], rowidx [0, 1, 2, 3])
    
    //LoadInt32(loadint32).Load(&(A[subtile_idx]), local_values, valid_items, 0);
    
    // 1. load 8 values per thread
    // 2. compute 2-max in registers (64 max per warp)
    // 3. do warp reduction + broadcast back
    // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
    // 5. Repeat (3) 8 times for top 8 values in 256
    // 6. store with byte index
    auto *tmp = tacc.get_multi_ptr<sycl::access::decorated::yes>().get();
    group_load_int(tmp).load(item_ct1, d_A, local_values);
    
    //ExchangeInt32(exchangeint32).BlockedToWarpStriped(local_values, local_values);
    
    // 1. load 8 values per thread
    // 2. compute 2-max in registers (64 max per warp)
    // 3. do warp reduction + broadcast back
    // 4. Up-shift maxed value, write index into shared memory, replace with 2nd largest
    // 5. Repeat (3) 8 times for top 8 values in 256
    // 6. store with byte index
    group_exchange(tmp).blocked_to_striped(item_ct1, local_values);
    
    #pragma unroll ITEMS_PER_THREAD
    for(int j = 0; j < ITEMS_PER_THREAD; j++)
      local_rowStats[j] = smem_rowStats[subtile_base_row+row_offset+j];

    #pragma unroll ITEMS_PER_THREAD
    for(int j = 0; j < ITEMS_PER_THREAD; j++)
      local_output[j] = sycl::vec<float, 1>((local_values[j]*MM_DEQUANT_CONST*local_rowStats[j]*colStat) + local_biasValue).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
      //absmax_col = fmax(fabsf(local_output[j]), absmax_col);

    // we store data in row major
    // to store data efficiently, we want to use block exchange: [0, 32, 64, 92] -> [0, 1, 2, 3]
    // so that each thread holds ITEMS_PER_THREAD consecutive items for each row
    // this way throughput into storage is increased by a factor of ~2x
    // for now we use a simple store
    #pragma unroll ITEMS_PER_THREAD
    for(int j = 0; j < ITEMS_PER_THREAD; j++)
    {
      int outIdx = col + ((base_row+subtile_base_row+row_offset+j)*numCols);
      if(outIdx< n_out && col < numCols)
        out[outIdx] = local_output[j];
    }

    row_offset += rows_per_load;
  }
}
 
 
//========================================k extract outliers======================

template <int FORMAT> SYCL_EXTERNAL void kExtractOutliers(char *A, int *idx, char *out, int idx_size, int rowsA, int colsA, int tiledRowsA, int tiledColsA, const sycl::nd_item<3> &item_ct1, const sycl_dacc_char &dacc_A, const sycl_dacc_char &dacc_out)
{
	int local_colidx = idx[item_ct1.get_group(2)];

	if(FORMAT==COL_TURING)
	{
		// TURING FORMAT:
		// 8*32 tiles with 4*4 subtiles
		// the 8*32 subtile has first all 4*4 subtiles of even rows (max 4*4*8 = 128 elements)
		// the subsequent 4*4 subtiles are for all odd rows if some rows columns are empty the values are zero
		// the tile repeats again after the 8*32 tile in a major column order, meaning: (next 8 rows are A[8:16, 0:32])
		// the next tile is the next 8 rows for the same 32 columns. Once all rows are finished, the column
		// index increases by 32
		// columns are grouped in increments of 4, meaning that one has the following rows and columns
		// rows: [0 0 0 0, 2 2 2 2, 4 4 4 4, 6 6 6 6, 0 0 0 0 ...]
		// cols: [0 1 2 3, 0 1 2 4, 0 1 2 3, 0 1 2 3, 4 5 6 7 ...]

		// each thread reads 1 element = 1 row
		for(int row = item_ct1.get_local_id(2); row < rowsA; row+= item_ct1.get_local_range(2))
		{
			int offset_per_col_tile = ((rowsA+7)/8)*32*8;
			int tile_offset_rows = (row/8)*32*8;
			int tile_offset_cols = (local_colidx/32)*offset_per_col_tile;
			int offset = 0;
			int subtile_col_idx = local_colidx%32;
			int subtile_row_idx = row % 8;
			if(row % 2 == 1)
				offset += 128 + (subtile_col_idx/4)*16 + (subtile_col_idx%4) + ((subtile_row_idx-1)*2);
			else
				// even
				offset += 0   + (subtile_col_idx/4)*16 + (subtile_col_idx%4) + (subtile_row_idx*2);

			offset += tile_offset_rows + tile_offset_cols;

			char val = dacc_A[offset];

			int out_idx = (row*idx_size) + item_ct1.get_group(2);
			dacc_out[out_idx] = val;
		}
	}
	else if(FORMAT == COL_AMPERE)
	{

		for(int row = item_ct1.get_local_id(2); row < rowsA; row+= item_ct1.get_local_range(2))
		{
			// we got 32x32 tiles and we use the magic equation from the cublasLt doc to get the element
			// within each tile.
			int offset_per_col_tile = ((rowsA+31)/32)*32*32;
			int tile_offset_rows = (row/32)*32*32;
			int tile_offset_cols = (local_colidx/32)*offset_per_col_tile;
			int subtile_col_idx = local_colidx%32;
			int subtile_row_idx = row % 32;
			// this magic is taken from the cublasLt doc (search for COL32)
			int offset = (((subtile_row_idx%8)/2*4+subtile_row_idx/8)*2+subtile_row_idx%2)*32+subtile_col_idx;
			offset += tile_offset_cols + tile_offset_rows;

			char val = A[offset];
			int out_idx = (row*idx_size) + item_ct1.get_group(2);
			dacc_out[out_idx] = val;
		}
	}
}

//====================kfunc================

template <typename T, int FUNC> SYCL_EXTERNAL void kfunc(T *A, T *B, T value, long n, const sycl::nd_item<3> &item_ct1)
{
  for(long i = (item_ct1.get_local_range(2)*item_ct1.get_group(2)) + item_ct1.get_local_id(2); i < n; i+=(item_ct1.get_local_range(2)*item_ct1.get_group_range(2)))
  {
    switch(FUNC)
    {
      case FILL:
        A[i] = (T)value;
        break;
      case ARANGE:
        A[i] = (T)i;
        break;
      case _MUL:
        A[i] = A[i]*B[i];
        break;
    }
  }
}


//=====================================================================================================


//==============================================================
//                   TEMPLATE DEFINITIONS
//==============================================================

template SYCL_EXTERNAL void kfunc<float, FILL>(float *A, float *B, float value, long n,
                                 const sycl::nd_item<3> &item_ct1);
template SYCL_EXTERNAL void kfunc<unsigned char, FILL>(unsigned char *A, unsigned char *B, unsigned char value, long n,
                                         const sycl::nd_item<3> &item_ct1);
template SYCL_EXTERNAL void kfunc<float, ARANGE>(float *A, float *B, float value, long n,
                                   const sycl::nd_item<3> &item_ct1);
template SYCL_EXTERNAL void kfunc<float, _MUL>(float *A, float *B, float value, long n,
                                 const sycl::nd_item<3> &item_ct1);



template void kdequant_mm_int32_fp16<4, 128, 512>(int *__restrict__ const A, float *__restrict__ const rowStats, float *__restrict__ const colStats, sycl::half *out, float* newRowStats, float* newcolStats, sycl::half * __restrict__ const bias, const int numRows, const int numCols, const int tileCols, const int n, const sycl::nd_item<3> &item_ct1,  float *smem_rowStats, const sycl_la &tacc, const sycl_dacc &dacc_A);



template SYCL_EXTERNAL void kExtractOutliers<COL_TURING>(char *A, int *idx, char *out, int idx_size, int rowsA, int colsA, int tiledRowsA, int tiledColsA,   const sycl::nd_item<3> &item_ct1, const sycl_dacc_char &dacc_A, const sycl_dacc_char &dacc_out);

template SYCL_EXTERNAL void kExtractOutliers<COL_AMPERE>(char *A, int *idx, char *out, int idx_size, int rowsA, int colsA, int tiledRowsA, int tiledColsA,  const sycl::nd_item<3> &item_ct1, const sycl_dacc_char &dacc_A, const sycl_dacc_char &dacc_out);



template SYCL_EXTERNAL void kTransformRowToFormat<256, 8, 32, 32*8, 0, COL32>(char *__restrict__ const A, char *out, int rows, int cols, int tiledCols, int outRows, int outCols,  const sycl::nd_item<3> &item_ct1,   char *smem_data, const sycl_dacc_char &dacc_A, const sycl_dacc_char &dacc_out);

template SYCL_EXTERNAL void kTransformRowToFormat<256, 8, 32, 32*8, 1, COL32>(char *__restrict__ const A, char *out, int rows, int cols, int tiledCols, int outRows, int outCols,  const sycl::nd_item<3> &item_ct1,   char *smem_data, const sycl_dacc_char &dacc_A, const sycl_dacc_char &dacc_out);

template SYCL_EXTERNAL void kTransformRowToFormat<256, 8, 32, 32*8, 0, COL_TURING>(char *__restrict__ const A, char *out, int rows, int cols, int tiledCols, int outRows, int outCols,  const sycl::nd_item<3> &item_ct1,  char *smem_data, const sycl_dacc_char &dacc_A, const sycl_dacc_char &dacc_out);

template SYCL_EXTERNAL void kTransformRowToFormat<256, 8, 32, 32*8, 1, COL_TURING>(char *__restrict__ const A, char *out, int rows, int cols, int tiledCols, int outRows, int outCols,  const sycl::nd_item<3> &item_ct1, char *smem_data, const sycl_dacc_char &dacc_A, const sycl_dacc_char &dacc_out);

template SYCL_EXTERNAL void kTransformRowToFormat<256, 8, 32, 32*8, 0, COL_AMPERE>(char *__restrict__ const A, char *out, int rows, int cols, int tiledCols, int outRows, int outCols, const sycl::nd_item<3> &item_ct1, char *smem_data, const sycl_dacc_char &dacc_A, const sycl_dacc_char &dacc_out);

template SYCL_EXTERNAL void kTransformRowToFormat<256, 8, 32, 32*8, 1, COL_AMPERE>(char *__restrict__ const A, char *out, int rows, int cols, int tiledCols, int outRows, int outCols, const sycl::nd_item<3> &item_ct1, char *smem_data, const sycl_dacc_char &dacc_A, const sycl_dacc_char &dacc_out);




template void kDoubleRowColQuant<64, 4, 16, 64*4, 0>(sycl::half *__restrict__ const A, float *__restrict__ const rowStats, float * __restrict__ const colStats, char *out_col_normed, char *out_row_normed, int *rowidx, int *colidx, sycl::half *val, int * __restrict__ nnz_block_ptr, float threshold, int rows, int cols, int tiledCols, const sycl::nd_item<3> &item_ct1, float *smem_row_stats, unsigned int *smem_nnz_row_idx, const sycl_la &tacc, const sycl::accessor<sycl::half, 1> &dacc_A, const sycl_dacc_char &dacc_out_col_normed, const sycl_dacc_char &dacc_out_row_normed, const sycl_dacc_float &dacc_rowStats, const sycl_dacc_float &dacc_colStats, const sycl_dacc &dacc_rowidx, const sycl_dacc &dacc_colidx, const sycl::accessor<sycl::half, 1> &dacc_val, const sycl_dacc &dacc_nnz_block_ptr);

template void kDoubleRowColQuant<64, 4, 16, 64*4, 1>(sycl::half *__restrict__ const A, float *__restrict__ const rowStats, float * __restrict__ const colStats, char *out_col_normed, char *out_row_normed, int *rowidx, int *colidx, sycl::half *val, int * __restrict__ nnz_block_ptr, float threshold, int rows, int cols, int tiledCols, const sycl::nd_item<3> &item_ct1, float *smem_row_stats, unsigned int *smem_nnz_row_idx, const sycl_la &tacc, const sycl::accessor<sycl::half, 1> &dacc_A, const sycl_dacc_char &dacc_out_col_normed, const sycl_dacc_char &dacc_out_row_normed, const sycl_dacc_float &dacc_rowStats, const sycl_dacc_float &dacc_colStats, const sycl_dacc &dacc_rowidx, const sycl_dacc &dacc_colidx, const sycl::accessor<sycl::half, 1> &dacc_val, const sycl_dacc &dacc_nnz_block_ptr);



template void kgetColRowStats<sycl::half, 64, 4, 16, 64*4, 0>(sycl::half * __restrict__ A, float *rowStats, float *colStats, int * nnz_count_row, float nnz_threshold, int rows, int cols, int tiledRows, int tiledCols, const sycl::nd_item<3> &item_ct1, float *smem_row_absmax_values, int *smem_row_nnz_values, const sycl_la &tacc, const sycl::accessor<sycl::half, 1> &dacc_A);
template void kgetColRowStats<sycl::half, 64, 4, 16, 64*4, 1>(sycl::half * __restrict__ A, float *rowStats, float *colStats, int * nnz_count_row, float nnz_threshold, int rows, int cols, int tiledRows, int tiledCols, const sycl::nd_item<3> &item_ct1, float *smem_row_absmax_values, int *smem_row_nnz_values, const sycl_la &tacc, const sycl::accessor<sycl::half, 1> &dacc_A);


template unsigned char dQuantize<0>(float* smem_code, const float rand, float x);
template unsigned char dQuantize<1>(float* smem_code, const float rand, float x);

template<typename T> SYCL_EXTERNAL void kEstimateQuantiles(const T A, float *code, const float offset, const float max_val, const int n, const sycl::nd_item<3> &item_ct1, const sycl_la &tacc, const sycl::accessor<T, 1> &dacc_A,  const sycl_dacc_float &dacc_code);
template<typename T> SYCL_EXTERNAL void kEstimateQuantiles(const T A, float *code, const float offset, const sycl::half max_val, const int n, const sycl::nd_item<3> &item_ct1, const sycl_la &tacc, const sycl::accessor<T, 1> &dacc_A, const sycl_dacc_float &dacc_code);

#define MAKE_PreconditionOptimizer32bit1State(oname, gtype) \
template SYCL_EXTERNAL void kPreconditionOptimizer32bit1State<gtype, oname, 4096, 8>(gtype* g, gtype* p, \
                float* state1, float *unorm, \
                const float beta1, const float beta2, const float eps, const float weight_decay, \
                const int step, const float lr, const float gnorm_scale, const int n, const sycl::nd_item<3> &item_ct1,const sycl_la &tacc,const sycl::accessor<gtype, 1> &dacc_g,const sycl_dacc_float &dacc_state1, const sycl_dacc_float &dacc_unorm); \

MAKE_PreconditionOptimizer32bit1State(MOMENTUM, sycl::half)
MAKE_PreconditionOptimizer32bit1State(MOMENTUM, float)
MAKE_PreconditionOptimizer32bit1State(RMSPROP, sycl::half)
MAKE_PreconditionOptimizer32bit1State(RMSPROP, float)
MAKE_PreconditionOptimizer32bit1State(LION, sycl::half)
MAKE_PreconditionOptimizer32bit1State(LION, float)
MAKE_PreconditionOptimizer32bit1State(LION, sycl::ext::oneapi::bfloat16)
MAKE_PreconditionOptimizer32bit1State(ADAGRAD, sycl::half)
MAKE_PreconditionOptimizer32bit1State(ADAGRAD, float)

#define MAKE_Optimizer32bit1State(oname, gtype) \
template SYCL_EXTERNAL void kOptimizer32bit1State<gtype, oname>(gtype* g, gtype* p, float* state1, float *unorm, const float max_unorm, const float param_norm, \
    const float beta1, const float beta2, const float eps, const float weight_decay,const int step, const float lr, const float gnorm_scale, const bool skip_zeros, const int n, const sycl::nd_item<3> &item_ct1,const sycl_la &tacc,const sycl::accessor<gtype, 1> &dacc_g,const sycl::accessor<gtype, 1> &dacc_p,const sycl_dacc_float &dacc_state1, const sycl_dacc_float &dacc_unorm); \

MAKE_Optimizer32bit1State(MOMENTUM, sycl::half)
MAKE_Optimizer32bit1State(MOMENTUM, float)
MAKE_Optimizer32bit1State(RMSPROP, sycl::half)
MAKE_Optimizer32bit1State(RMSPROP, float)
MAKE_Optimizer32bit1State(LION, sycl::half)
MAKE_Optimizer32bit1State(LION, float)
MAKE_Optimizer32bit1State(LION, sycl::ext::oneapi::bfloat16)
MAKE_Optimizer32bit1State(ADAGRAD, sycl::half)
MAKE_Optimizer32bit1State(ADAGRAD, float)

#define MAKE_PreconditionOptimizer32bit2State(oname, gtype) \
template SYCL_EXTERNAL void kPreconditionOptimizer32bit2State<gtype, oname, 4096, 8>(gtype* g, gtype* p,  \
                float* state1, float* state2, float *unorm, \
                const float beta1, const float beta2, const float eps, const float weight_decay, \
                const int step, const float lr, const float gnorm_scale, const int n, const sycl::nd_item<3> &item_ct1,const sycl_la &tacc,const sycl_dacc_float &dacc_state1,const sycl_dacc_float &dacc_state2,const sycl::accessor<gtype, 1> &dacc_g, const sycl_dacc_float &dacc_unorm); \

MAKE_PreconditionOptimizer32bit2State(ADAM, float)
MAKE_PreconditionOptimizer32bit2State(ADAM, sycl::half)
MAKE_PreconditionOptimizer32bit2State(ADAM, sycl::ext::oneapi::bfloat16)


template SYCL_EXTERNAL void kOptimizer32bit2State<float, ADAM>(float* g, float* p, float* state1, float* state2, float *unorm, const float max_unorm, const float param_norm,
    const float beta1, const float beta2, const float eps, const float weight_decay,const int step, const float lr, const float gnorm_scale, const bool skip_zeros, const int n,
    const sycl::nd_item<3> &item_ct1, const sycl_la &tacc, const sycl::accessor<float, 1> &dacc_g, const sycl::accessor<float, 1> &dacc_p, const sycl_dacc_float &dacc_state1, const  sycl_dacc_float &dacc_state2, const sycl_dacc_float &dacc_unorm);

template SYCL_EXTERNAL void kOptimizer32bit2State<sycl::half, ADAM>(sycl::half* g, sycl::half* p, float* state1, float* state2, float *unorm, const float max_unorm, const float param_norm,
    const float beta1, const float beta2, const float eps, const float weight_decay,const int step, const float lr, const float gnorm_scale, const bool skip_zeros, const int n,
    const sycl::nd_item<3> &item_ct1,const sycl_la &tacc,const sycl::accessor<sycl::half, 1> &dacc_g,const sycl::accessor<sycl::half, 1> &dacc_p,const sycl_dacc_float &dacc_state1,const  sycl_dacc_float &dacc_state2, const sycl_dacc_float &dacc_unorm);
    
template SYCL_EXTERNAL void kOptimizer32bit2State<sycl::ext::oneapi::bfloat16, ADAM>(sycl::ext::oneapi::bfloat16* g, sycl::ext::oneapi::bfloat16* p, float* state1, float* state2, float *unorm, const float max_unorm, const float param_norm,
    const float beta1, const float beta2, const float eps, const float weight_decay,const int step, const float lr, const float gnorm_scale, const bool skip_zeros, const int n,
    const sycl::nd_item<3> &item_ct1,const sycl_la &tacc,const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_g,const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_p,const sycl_dacc_float &dacc_state1,const  sycl_dacc_float &dacc_state2, const sycl_dacc_float &dacc_unorm);

#define MAKE_PreconditionStatic8bit1State(oname, gtype) \
template SYCL_EXTERNAL void kPreconditionOptimizerStatic8bit1State<gtype, oname>(gtype* p, gtype* __restrict__ const g, unsigned char*__restrict__  const state1,  \
                float *unorm,  \
                const float beta1,  \
                const float beta2,  \
                const float eps, const int step,  \
                float* __restrict__ const quantiles1,  \
                float* max1, float* new_max1,  \
                const float weight_decay, \
                const float gnorm_scale,  \
                const int n, const sycl::nd_item<3> &item_ct1, float *smem_quantiles1,const sycl_la &tacc, const sycl::accessor<gtype, 1> &dacc_g, const sycl_dacc_uc &dacc_state1); \

MAKE_PreconditionStatic8bit1State(MOMENTUM, sycl::half)
MAKE_PreconditionStatic8bit1State(MOMENTUM, float)
MAKE_PreconditionStatic8bit1State(RMSPROP, sycl::half)
MAKE_PreconditionStatic8bit1State(RMSPROP, float)
MAKE_PreconditionStatic8bit1State(LION, sycl::half)
MAKE_PreconditionStatic8bit1State(LION, float)

#define MAKE_optimizerStatic8bit1State(oname, gtype) \
template void kOptimizerStatic8bit1State<gtype, oname>(gtype* p, gtype* const g, unsigned char* state1,  \
                const float *unorm, const float max_unorm, const float param_norm, \
                const float beta1,  \
                const float beta2,  \
                const float eps, const int step, const float lr, \
                float* __restrict__ const quantiles1,  \
                float* max1, float* new_max1,  \
                float weight_decay, \
                const float gnorm_scale,  \
                const int n, const sycl::nd_item<3> &item_ct1, float *smem_quantiles1, const sycl_la &tacc, \
                const sycl::accessor<gtype, 1> &dacc_g, const sycl::accessor<gtype, 1> &dacc_p, \
                const sycl_dacc_uc &dacc_state1); \

MAKE_optimizerStatic8bit1State(MOMENTUM, sycl::half)
MAKE_optimizerStatic8bit1State(MOMENTUM, float)
MAKE_optimizerStatic8bit1State(RMSPROP, sycl::half)
MAKE_optimizerStatic8bit1State(RMSPROP, float)
MAKE_optimizerStatic8bit1State(LION, sycl::half)
MAKE_optimizerStatic8bit1State(LION, float)

#define MAKE_PreconditionStatic8bit2State(oname, gtype) \
template void kPreconditionOptimizerStatic8bit2State<gtype, oname>(gtype* p, gtype* __restrict__ const g, unsigned char*__restrict__  const state1, unsigned char* __restrict__ const state2, \
                float *unorm, \
                const float beta1, const float beta2, \
                const float eps, const int step,  \
                float* __restrict__ const quantiles1, float* __restrict__ const quantiles2, \
                float* max1, float* max2, float* new_max1, float* new_max2, \
                const float gnorm_scale,  \
                const int n, const sycl::nd_item<3> &item_ct1,  float *smem_quantiles1, float *smem_quantiles2, const sycl_la &tacc, const sycl::accessor<gtype, 1> &dacc_g, const sycl_dacc_uc &dacc_state1, const sycl_dacc_uc &dacc_state2); \

MAKE_PreconditionStatic8bit2State(ADAM, sycl::half)
MAKE_PreconditionStatic8bit2State(ADAM, float)

#define MAKE_optimizerStatic8bit2State(oname, gtype) \
template void kOptimizerStatic8bit2State<gtype, oname>(gtype* p, gtype* const g, unsigned char* state1, unsigned char* state2, \
                const float *unorm, const float max_unorm, const float param_norm, \
                const float beta1, const float beta2, \
                const float eps, const int step, const float lr, \
                float* __restrict__ const quantiles1, float* __restrict__ const quantiles2, \
                float* max1, float* max2, float* new_max1, float* new_max2, \
                float weight_decay, \
                const float gnorm_scale,  \
                const int n, const sycl::nd_item<3> &item_ct1, float *smem_quantiles1, float *smem_quantiles2, const sycl_la &tacc, const sycl::accessor<gtype, 1> &dacc_g, const sycl::accessor<gtype, 1> &dacc_p, const sycl_dacc_uc &dacc_state1, const sycl_dacc_uc &dacc_state2); \

MAKE_optimizerStatic8bit2State(ADAM, sycl::half)
MAKE_optimizerStatic8bit2State(ADAM, float)

template SYCL_EXTERNAL void kPercentileClipping<float, 2048, 4>(float * __restrict__ g, float *gnorm_vec, int step, const int n,
                                                  const sycl::nd_item<3> &item_ct1, const sycl_la &tacc, const sycl::accessor<float, 1> &dacc_g);
template SYCL_EXTERNAL void kPercentileClipping<sycl::half, 2048, 4>(sycl::half * __restrict__ g, float *gnorm_vec, int step, const int n,
                                                 const sycl::nd_item<3> &item_ct1, const sycl_la &tacc, const sycl::accessor<sycl::half, 1> &dacc_g);


#define MAKE_kQuantizeBlockwise(dtype, blocksize, num_per_thread, stochastic, data_type_name) \
template void kQuantizeBlockwise<dtype, blocksize, num_per_thread, stochastic, data_type_name>(float * code, dtype * __restrict__ const A, float *absmax, unsigned char *out, float * __restrict__ const rand, const int rand_offset, const int n, const sycl::nd_item<3> &item_ct1, float *smem_code, float *smem_absmax_value,const sycl_la &tacc,const sycl::accessor<dtype, 1> &dacc_A, const sycl_dacc_float &dacc_rand, const sycl_dacc_uc &dacc_out, const sycl_dacc_float &dacc_code, const sycl_dacc_float &dacc_absmax); 

MAKE_kQuantizeBlockwise(sycl::half,  4096, 4, 0, General8bit)
MAKE_kQuantizeBlockwise(sycl::half,  4096, 4, 1, General8bit)
MAKE_kQuantizeBlockwise(sycl::half,  2048, 4, 0, General8bit)
MAKE_kQuantizeBlockwise(sycl::half,  1024, 4, 0, General8bit)
MAKE_kQuantizeBlockwise(sycl::half,   512, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(sycl::half,   256, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(sycl::half,   128, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(sycl::half,    64, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(sycl::half,  4096, 4, 0, FP4)
MAKE_kQuantizeBlockwise(sycl::half,  2048, 4, 0, FP4)
MAKE_kQuantizeBlockwise(sycl::half,  1024, 4, 0, FP4)
MAKE_kQuantizeBlockwise(sycl::half,   512, 2, 0, FP4)
MAKE_kQuantizeBlockwise(sycl::half,   256, 2, 0, FP4)
MAKE_kQuantizeBlockwise(sycl::half,   128, 2, 0, FP4)
MAKE_kQuantizeBlockwise(sycl::half,    64, 2, 0, FP4)
MAKE_kQuantizeBlockwise(sycl::half,  4096, 4, 0, NF4)
MAKE_kQuantizeBlockwise(sycl::half,  2048, 4, 0, NF4)
MAKE_kQuantizeBlockwise(sycl::half,  1024, 4, 0, NF4)
MAKE_kQuantizeBlockwise(sycl::half,   512, 2, 0, NF4)
MAKE_kQuantizeBlockwise(sycl::half,   256, 2, 0, NF4)
MAKE_kQuantizeBlockwise(sycl::half,   128, 2, 0, NF4)
MAKE_kQuantizeBlockwise(sycl::half,    64, 2, 0, NF4)
MAKE_kQuantizeBlockwise(float, 4096, 4, 0, General8bit)
MAKE_kQuantizeBlockwise(float, 4096, 4, 1, General8bit)
MAKE_kQuantizeBlockwise(float, 2048, 4, 0, General8bit)
MAKE_kQuantizeBlockwise(float, 1024, 4, 0, General8bit)
MAKE_kQuantizeBlockwise(float,  512, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(float,  256, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(float,  128, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(float,   64, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(float, 4096, 4, 0, FP4)
MAKE_kQuantizeBlockwise(float, 2048, 4, 0, FP4)
MAKE_kQuantizeBlockwise(float, 1024, 4, 0, FP4)
MAKE_kQuantizeBlockwise(float,  512, 2, 0, FP4)
MAKE_kQuantizeBlockwise(float,  256, 2, 0, FP4)
MAKE_kQuantizeBlockwise(float,  128, 2, 0, FP4)
MAKE_kQuantizeBlockwise(float,   64, 2, 0, FP4)
MAKE_kQuantizeBlockwise(float, 4096, 4, 0, NF4)
MAKE_kQuantizeBlockwise(float, 2048, 4, 0, NF4)
MAKE_kQuantizeBlockwise(float, 1024, 4, 0, NF4)
MAKE_kQuantizeBlockwise(float,  512, 2, 0, NF4)
MAKE_kQuantizeBlockwise(float,  256, 2, 0, NF4)
MAKE_kQuantizeBlockwise(float,  128, 2, 0, NF4)
MAKE_kQuantizeBlockwise(float,   64, 2, 0, NF4)

MAKE_kQuantizeBlockwise(sycl::ext::oneapi::bfloat16, 4096, 4, 0, General8bit)
MAKE_kQuantizeBlockwise(sycl::ext::oneapi::bfloat16, 4096, 4, 1, General8bit)
MAKE_kQuantizeBlockwise(sycl::ext::oneapi::bfloat16, 2048, 4, 0, General8bit)
MAKE_kQuantizeBlockwise(sycl::ext::oneapi::bfloat16, 1024, 4, 0, General8bit)
MAKE_kQuantizeBlockwise(sycl::ext::oneapi::bfloat16,  512, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(sycl::ext::oneapi::bfloat16,  256, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(sycl::ext::oneapi::bfloat16,  128, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(sycl::ext::oneapi::bfloat16,   64, 2, 0, General8bit)
MAKE_kQuantizeBlockwise(sycl::ext::oneapi::bfloat16, 4096, 4, 0, FP4)
MAKE_kQuantizeBlockwise(sycl::ext::oneapi::bfloat16, 2048, 4, 0, FP4)
MAKE_kQuantizeBlockwise(sycl::ext::oneapi::bfloat16, 1024, 4, 0, FP4)
MAKE_kQuantizeBlockwise(sycl::ext::oneapi::bfloat16,  512, 2, 0, FP4)
MAKE_kQuantizeBlockwise(sycl::ext::oneapi::bfloat16,  256, 2, 0, FP4)
MAKE_kQuantizeBlockwise(sycl::ext::oneapi::bfloat16,  128, 2, 0, FP4)
MAKE_kQuantizeBlockwise(sycl::ext::oneapi::bfloat16,   64, 2, 0, FP4)
MAKE_kQuantizeBlockwise(sycl::ext::oneapi::bfloat16, 4096, 4, 0, NF4)
MAKE_kQuantizeBlockwise(sycl::ext::oneapi::bfloat16, 2048, 4, 0, NF4)
MAKE_kQuantizeBlockwise(sycl::ext::oneapi::bfloat16, 1024, 4, 0, NF4)
MAKE_kQuantizeBlockwise(sycl::ext::oneapi::bfloat16,  512, 2, 0, NF4)
MAKE_kQuantizeBlockwise(sycl::ext::oneapi::bfloat16,  256, 2, 0, NF4)
MAKE_kQuantizeBlockwise(sycl::ext::oneapi::bfloat16,  128, 2, 0, NF4) 
MAKE_kQuantizeBlockwise(sycl::ext::oneapi::bfloat16,   64, 2, 0, NF4)

template SYCL_EXTERNAL void kDequantizeBlockwise<sycl::half, 512, 64, 8, FP4>(float *code, unsigned char * A, float * absmax, sycl::half *out, const int blocksize, const int n, const sycl::nd_item<3> &item_ct1,const sycl_la &tacc,const sycl_dacc_uc &dacc_A,const sycl::accessor<sycl::half, 1> &dacc_out, const sycl_dacc_float &dacc_code, const sycl_dacc_float &dacc_absmax);

template SYCL_EXTERNAL void kDequantizeBlockwise<sycl::half, 512, 64, 8, General8bit>(float *code, unsigned char * A, float * absmax, sycl::half *out, const int blocksize, const int n, const sycl::nd_item<3> &item_ct1,const sycl_la &tacc, const sycl_dacc_uc &dacc_A, const sycl::accessor<sycl::half, 1> &dacc_out, const sycl_dacc_float &dacc_code, const sycl_dacc_float &dacc_absmax);

template SYCL_EXTERNAL void kDequantizeBlockwise<sycl::half, 512, 64, 8, NF4>(float *code, unsigned char * A, float * absmax, sycl::half *out, const int blocksize, const int n,const sycl::nd_item<3> &item_ct1,const sycl_la &tacc, const sycl_dacc_uc &dacc_A, const sycl::accessor<sycl::half, 1> &dacc_out, const sycl_dacc_float &dacc_code, const sycl_dacc_float &dacc_absmax);

template SYCL_EXTERNAL void kDequantizeBlockwise<float, 512, 64, 8, FP4>(float *code, unsigned char * A, float * absmax, float *out, const int blocksize, const int n, const sycl::nd_item<3> &item_ct1, const sycl_la &tacc, const sycl_dacc_uc &dacc_A, const sycl::accessor<float, 1> &dacc_out, const sycl_dacc_float &dacc_code, const sycl_dacc_float &dacc_absmax);

template SYCL_EXTERNAL void kDequantizeBlockwise<float, 512, 64, 8, General8bit>(float *code, unsigned char * A, float * absmax, float *out, const int blocksize, const int n,const sycl::nd_item<3> &item_ct1, const sycl_la &tacc, const sycl_dacc_uc &dacc_A, const sycl::accessor<float, 1> &dacc_out, const sycl_dacc_float &dacc_code, const sycl_dacc_float &dacc_absmax);

template SYCL_EXTERNAL void kDequantizeBlockwise<float, 512, 64, 8, NF4>(float *code, unsigned char * A, float * absmax, float *out, const int blocksize, const int n, const sycl::nd_item<3> &item_ct1,const sycl_la &tacc, const sycl_dacc_uc &dacc_A, const sycl::accessor<float, 1> &dacc_out, const sycl_dacc_float &dacc_code, const sycl_dacc_float &dacc_absmax);


template SYCL_EXTERNAL void kDequantizeBlockwise<sycl::ext::oneapi::bfloat16, 512, 64, 8, FP4>(float *code, unsigned char * A, float * absmax, sycl::ext::oneapi::bfloat16 *out, const int blocksize, const int n,const sycl::nd_item<3> &item_ct1, const sycl_la &tacc, const sycl_dacc_uc &dacc_A, const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_out, const sycl_dacc_float &dacc_code, const sycl_dacc_float &dacc_absmax);

template SYCL_EXTERNAL void kDequantizeBlockwise<sycl::ext::oneapi::bfloat16, 512, 64, 8, General8bit>(float *code, unsigned char * A, float * absmax, sycl::ext::oneapi::bfloat16 *out, const int blocksize, const int n,const sycl::nd_item<3> &item_ct1, const sycl_la &tacc, const sycl_dacc_uc &dacc_A, const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_out, const sycl_dacc_float &dacc_code, const sycl_dacc_float &dacc_absmax);

template SYCL_EXTERNAL void kDequantizeBlockwise<sycl::ext::oneapi::bfloat16, 512, 64, 8, NF4>(float *code, unsigned char * A, float * absmax, sycl::ext::oneapi::bfloat16 *out, const int blocksize, const int n,const sycl::nd_item<3> &item_ct1,const sycl_la &tacc, const sycl_dacc_uc &dacc_A, const sycl::accessor<sycl::ext::oneapi::bfloat16, 1> &dacc_out, const sycl_dacc_float &dacc_code, const sycl_dacc_float &dacc_absmax);

#define MAKE_OptimizerStatic8bit2StateBlockwise(oname, gtype, block_size, num_per_thread) \
template void kOptimizerStatic8bit2StateBlockwise<gtype, oname, block_size, num_per_thread>(gtype* p, gtype* __restrict__ const g, unsigned char* state1, unsigned char* state2, \
                const float beta1, const float beta2, \
                const float eps, const int step, const float lr, \
                float* __restrict__ const quantiles1, float* __restrict__ const quantiles2, \
                float* absmax1, float* absmax2,  \
                float weight_decay, \
                const float gnorm_scale, const bool skip_zeros, const int n, const sycl::nd_item<3> &item_ct1, sycl::local_accessor<float, 2> smem_quantiles1, sycl::local_accessor<float, 2> smem_quantiles2, float *smem_exchange1, float *smem_exchange2,const sycl_la &tacc, \
                const sycl::accessor<gtype, 1> &dacc_g, \
                const sycl::accessor<gtype, 1> &dacc_p, \
                const sycl_dacc_uc &dacc_state1, const sycl_dacc_uc &dacc_state2, \
                const sycl_dacc_float &dacc_quantiles1, const sycl_dacc_float &dacc_quantiles2, \
                const sycl_dacc_float &dacc_absmax1, const sycl_dacc_float &dacc_absmax2); \
                
MAKE_OptimizerStatic8bit2StateBlockwise(ADAM, float, 2048, 8)
MAKE_OptimizerStatic8bit2StateBlockwise(ADAM, sycl::half, 2048, 8)
MAKE_OptimizerStatic8bit2StateBlockwise(ADAM, sycl::ext::oneapi::bfloat16, 2048, 8)


#define MAKE_OptimizerStatic8bit1StateBlockwise(oname, gtype, block_size, num_per_thread) \
template void kOptimizerStatic8bit1StateBlockwise<gtype, oname, block_size, num_per_thread>( \
		gtype* p, gtype* __restrict__ const g, unsigned char* state1, \
                const float beta1, const float beta2, \
                const float eps, const int step, const float lr, \
                float* __restrict__ const quantiles1, \
                float* absmax1, \
                float weight_decay, \
                const float gnorm_scale, const bool skip_zeros, const int n, const sycl::nd_item<3> &item_ct1, sycl::local_accessor<float, 2> smem_quantiles1, float *smem_exchange1,const sycl_la &tacc, \
                const sycl::accessor<gtype, 1> &dacc_g, \
                const sycl::accessor<gtype, 1> &dacc_p, \
                const sycl_dacc_uc &dacc_state1,        \
                const sycl_dacc_float &dacc_quantiles1, \
                const sycl_dacc_float &dacc_absmax1);   \

MAKE_OptimizerStatic8bit1StateBlockwise(MOMENTUM, float, 2048, 8)
MAKE_OptimizerStatic8bit1StateBlockwise(MOMENTUM, sycl::half, 2048, 8)
MAKE_OptimizerStatic8bit1StateBlockwise(RMSPROP, float, 2048, 8)
MAKE_OptimizerStatic8bit1StateBlockwise(RMSPROP, sycl::half, 2048, 8)
MAKE_OptimizerStatic8bit1StateBlockwise(LION, float, 2048, 8)
MAKE_OptimizerStatic8bit1StateBlockwise(LION, sycl::half, 2048, 8)
MAKE_OptimizerStatic8bit1StateBlockwise(LION, sycl::ext::oneapi::bfloat16, 2048, 8)
MAKE_OptimizerStatic8bit1StateBlockwise(ADAGRAD, float, 2048, 8)
MAKE_OptimizerStatic8bit1StateBlockwise(ADAGRAD, sycl::half, 2048, 8)

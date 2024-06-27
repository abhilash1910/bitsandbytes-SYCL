#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
#include <dpct/dpl_extras/dpcpp_extensions.h>
#include <dpct/lib_common_utils.hpp>

#include "oneapi/dnnl/dnnl.hpp"

#define ERR_NOT_IMPLEMENTED 100

#define HLF_MAX 65504
#define TH 512
#define NUM 4
#define NUM_BLOCK 512

#define THREADS_ESTIMATE 512
#define NUM_ESTIMATE 8
#define BLOCK_ESTIMATE 512

#define NUM_PER_THREAD 4

#define STATS_THREADS 64
#define STATS_ITEMS 4
#define STATS_ROWS 16

using namespace dnnl;

typedef sycl::ext::oneapi::bfloat16 bf16;
typedef sycl::local_accessor<uint8_t ,1> sycl_la;

typedef sycl::accessor<int, 1> sycl_dacc;
typedef sycl::accessor<float, 1> sycl_dacc_float;
typedef sycl::accessor<unsigned char, 1> sycl_dacc_uc;
typedef sycl::accessor<char, 1> sycl_dacc_char;

template <typename T> __dpct_inline__ int sgn(T val)
{
  return (T(0) < val) - (val < T(0));
}

int fill_up_to_nearest_multiple(int value, int multiple)
{
  return value + (value % multiple == 0 ? 0 : (multiple - (value % multiple)));
}

//Cub Helpers provided for reference -> already in dpcpp_extensions.h


/// Load linear segment items into block format across threads
/// Helper for Block Load
namespace dpct_{
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

    if constexpr (ALGORITHM == dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT) {
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

//=====================================================

#define FLT_MAX std::numeric_limits<float>::max()
#define FLT_MIN std::numeric_limits<float>::min()
//================================helpers===========================


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
//========================helper-==============================









#define NUM8BIT 16
#define NUM_THREADS 512
#define NUM_PER_BLOCK 512



//========================================k extract outliers======================

template <int FORMAT> SYCL_EXTERNAL void kExtractOutliers(char *A, int *idx, char *out, int idx_size, int rowsA, int colsA, int tiledRowsA, int tiledColsA, const sycl::nd_item<3> &item_ct1, const sycl_dacc_char &dacc_A, const sycl_dacc_char &dacc_out, const sycl_dacc &dacc_idx)
{
	int local_colidx = dacc_idx[item_ct1.get_group(2)];

	if(FORMAT== 1)//COL_TURING)
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
	else if(FORMAT == 2)//COL_AMPERE)
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

			char val = dacc_A[offset];
			int out_idx = (row*idx_size) + item_ct1.get_group(2);
			dacc_out[out_idx] = val;
		}
	}
}




//===========================extract outliers===========================

template <int FORMAT> void extractOutliers(char * A, int *idx, char *out, int idx_size, int rows, int cols)
{
  int threads = 512;
  // we load 128 column values per warp
  int tiledCols = fill_up_to_nearest_multiple(cols, 32);
  int tiledRows = 0;
  int size = NUM_BLOCK;
	int num_blocks = idx_size;

  if(FORMAT == 1)//COL_TURING)
  {
  
    tiledRows = fill_up_to_nearest_multiple(rows, 8);
  }
  else if(FORMAT == 2)//COL_AMPERE)
  {
      
      tiledRows = fill_up_to_nearest_multiple(rows, 32);
	}

  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
  
  sycl::buffer<char, 1> buff_A(A,sycl::range<1>(size));
  sycl::buffer<char, 1> buff_out(out,sycl::range<1>(size));
  sycl::buffer<int, 1> buff_idx(idx,sycl::range<1>(size));
  
  
  dpct::get_in_order_queue().submit(
    [&](sycl::handler &cgh) {
    
      sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
     sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
     sycl::accessor dacc_idx(buff_idx, cgh, sycl::read_write);
     
    
    cgh.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)), 
      [=](sycl::nd_item<3> item_ct1) {
           kExtractOutliers<FORMAT>(A, idx, out, idx_size, rows, cols, tiledRows, tiledCols, item_ct1, dacc_A, dacc_out, dacc_idx);
    });
   });
 
}


int main(){

             
float threshold  = 1.0f ;
int rows =1;
int cols= rows;


//====test extract outlier=========

char Ad[512];
int idx[512];
char out[512];
for(int i=0;i<512;i++){ Ad[i]=1;idx[i]=i-1;out[i]=1;}
int idx_size=512;

extractOutliers<1>(Ad, idx, out, idx_size, rows, cols);
//SEG FAULT ERROR
extractOutliers<2>(Ad, idx, out, idx_size, rows, cols);


return 0;

}
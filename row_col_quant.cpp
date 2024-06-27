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
  
  using group_load_half = dpct_::group::workgroup_load<ITEMS_PER_THREAD, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, sycl::half,  sycl::half *, sycl::nd_item<3>>;
  using group_store_char = dpct_::group::workgroup_store<ITEMS_PER_THREAD, dpct_::group::store_algorithm::BLOCK_STORE_DIRECT, char,  char *, sycl::nd_item<3>>;
  
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
              case 0: //COL32:
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
              case 1:// COL_TURING:
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
								case 2://COL_AMPERE:
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


//===================================double row col quant======================

void doubleRowColQuant(sycl::half * A, float *rowStats, float *colStats, char *out_col_normed, char *out_row_normed, int *rowidx, int *colidx, sycl::half *val, int *nnz_block_ptr, float threshold, int rows, int cols)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
	int num_blocks = 0;
  int size = NUM_BLOCK;
  
  sycl::buffer<sycl::half, 1> buff_A(A,sycl::range<1>(size));
  sycl::buffer<char, 1> buff_out_col_normed(out_col_normed,sycl::range<1>(size));
  sycl::buffer<char, 1> buff_out_row_normed(out_row_normed,sycl::range<1>(size));
  
  sycl::buffer<float, 1> buff_rowStats(rowStats,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_colStats(colStats,sycl::range<1>(size));
  sycl::buffer<int, 1> buff_rowidx(rowidx,sycl::range<1>(size));
  sycl::buffer<int, 1> buff_colidx(colidx,sycl::range<1>(size));
  sycl::buffer<sycl::half, 1> buff_val(val,sycl::range<1>(size));
  sycl::buffer<int, 1> buff_nnz_block_ptr(nnz_block_ptr,sycl::range<1>(size));
  
  
  
  int threads = 64;
  int items_per_thread = 4;
  int tile_cols = threads*items_per_thread;
  int tile_rows = 16;
  int tiledCols = fill_up_to_nearest_multiple(cols, tile_cols);
  int tiledRows = fill_up_to_nearest_multiple(rows, tile_rows);
	int row_tiles = (tiledRows/tile_rows);
	int col_tiles = (tiledCols/tile_cols);
	row_tiles = row_tiles > 0 ? row_tiles : 1;
	col_tiles = col_tiles > 0 ? col_tiles : 1;
  num_blocks = row_tiles * col_tiles;


  if(threshold > 0.0f)
    {
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
		    [&](sycl::handler &cgh) {
            
            using group_load = dpct_::group::workgroup_load<NUM_ESTIMATE, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, sycl::half,  sycl::half *, sycl::nd_item<3>>;
            size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);
            sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
            sycl::accessor dacc_out_col_normed(buff_out_col_normed, cgh, sycl::read_write);
            sycl::accessor dacc_out_row_normed(buff_out_row_normed, cgh, sycl::read_write);

            sycl::accessor dacc_rowStats(buff_rowStats, cgh, sycl::read_write);
            sycl::accessor dacc_colStats(buff_colStats, cgh, sycl::read_write);
            sycl::accessor dacc_rowidx(buff_rowidx, cgh, sycl::read_write);
            sycl::accessor dacc_colidx(buff_colidx, cgh, sycl::read_write);
            sycl::accessor dacc_val(buff_val, cgh, sycl::read_write);
            sycl::accessor dacc_nnz_block_ptr(buff_nnz_block_ptr, cgh, sycl::read_write);


            //__shared__ vars
            sycl::local_accessor<float, 1> smem_row_stats_acc_ct1(sycl::range<1>(256), cgh);
			      sycl::local_accessor<unsigned int, 1> smem_nnz_row_idx_acc_ct1(sycl::range<1>(256), cgh);
                        
        cgh.parallel_for(      
           sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
           [=](sycl::nd_item<3> item_ct1) {
          
                kDoubleRowColQuant<STATS_THREADS, STATS_ITEMS, STATS_ROWS, STATS_THREADS*STATS_ITEMS, 0>(A, rowStats, colStats, out_col_normed, out_row_normed, rowidx, colidx, val, nnz_block_ptr, threshold, rows, cols, tiledCols, item_ct1, smem_row_stats_acc_ct1.get_pointer(), smem_nnz_row_idx_acc_ct1.get_pointer(), tacc, dacc_A, dacc_out_col_normed, dacc_out_row_normed, dacc_rowStats, dacc_colStats, dacc_rowidx, dacc_colidx, dacc_val, dacc_nnz_block_ptr);
          });
      });
    }
  else
    {
  
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
		    [&](sycl::handler &cgh) {
            
            using group_load = dpct_::group::workgroup_load<NUM_ESTIMATE, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, sycl::half,  sycl::half *, sycl::nd_item<3>>;
            size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);
            sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
            sycl::accessor dacc_out_col_normed(buff_out_col_normed, cgh, sycl::read_write);
            sycl::accessor dacc_out_row_normed(buff_out_row_normed, cgh, sycl::read_write);
            
            sycl::accessor dacc_rowStats(buff_rowStats, cgh, sycl::read_write);
            sycl::accessor dacc_colStats(buff_colStats, cgh, sycl::read_write);
            sycl::accessor dacc_rowidx(buff_rowidx, cgh, sycl::read_write);
            sycl::accessor dacc_colidx(buff_colidx, cgh, sycl::read_write);
            sycl::accessor dacc_val(buff_val, cgh, sycl::read_write);
            sycl::accessor dacc_nnz_block_ptr(buff_nnz_block_ptr, cgh, sycl::read_write);

            
            //__shared__ vars
            sycl::local_accessor<float, 1> smem_row_stats_acc_ct1(sycl::range<1>(256), cgh);
			      sycl::local_accessor<unsigned int, 1> smem_nnz_row_idx_acc_ct1(sycl::range<1>(256), cgh);
            
                        
        cgh.parallel_for(      
           sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
           [=](sycl::nd_item<3> item_ct1) {
          
                kDoubleRowColQuant<STATS_THREADS, STATS_ITEMS, STATS_ROWS, STATS_THREADS*STATS_ITEMS, 0>(A, rowStats, colStats, out_col_normed, out_row_normed, rowidx, colidx, val, nnz_block_ptr, threshold, rows, cols, tiledCols,item_ct1, smem_row_stats_acc_ct1.get_pointer(), smem_nnz_row_idx_acc_ct1.get_pointer(),  tacc, dacc_A, dacc_out_col_normed, dacc_out_row_normed, dacc_rowStats, dacc_colStats, dacc_rowidx, dacc_colidx, dacc_val, dacc_nnz_block_ptr);
          });
      });
  
  }
  
}
//========================== transform row to format================================
template <int FORMAT, int TRANSPOSE> void transformRowToFormat(char * A, char *out, int rows, int cols)
{
  
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
	int num_blocks = 0;
  int size = NUM_BLOCK;
  
  sycl::buffer<char, 1> buff_A(A,sycl::range<1>(size));
  sycl::buffer<char, 1> buff_out(out,sycl::range<1>(size));
  
  
  int threads = 64;
  int items_per_thread = 8;
  // we load 128 column values per warp
  int tile_cols = 32*items_per_thread;
  int tile_rows = 32;
  int tiledCols = fill_up_to_nearest_multiple(cols, tile_cols);
  int tiledRows = fill_up_to_nearest_multiple(rows, tile_rows);
	int row_tiles = (tiledRows/tile_rows);
	int col_tiles = (tiledCols/tile_cols);
	row_tiles = row_tiles > 0 ? row_tiles : 1;
	col_tiles = col_tiles > 0 ? col_tiles : 1;
  num_blocks = row_tiles * col_tiles;

  int outCols = fill_up_to_nearest_multiple(cols, 32);
  int outRows = fill_up_to_nearest_multiple(rows, 32);
  if(FORMAT == 1)//COL_TURING)
  {
    if(TRANSPOSE)
      outRows = fill_up_to_nearest_multiple(cols, 8);
    else
      outRows = fill_up_to_nearest_multiple(rows, 8);
  }
  else if(FORMAT == 2)//COL_AMPERE)
  {
    if(TRANSPOSE)
      outRows = fill_up_to_nearest_multiple(cols, 32);
    else
      outRows = fill_up_to_nearest_multiple(rows, 32);
  }
  else
  {
    if(TRANSPOSE)
    {
      outCols = fill_up_to_nearest_multiple(rows, 32);
      outRows = cols;
    }
  }

  
  dpct::get_in_order_queue().submit(
    [&](sycl::handler &cgh) {
     
     sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
     sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
            
      
    //__shared__ vars
      sycl::local_accessor<char, 1> smem_data_acc_ct1(sycl::range<1>(32*33*8), cgh);

      cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)), 
        [=](sycl::nd_item<3> item_ct1) {
          kTransformRowToFormat<256, 8, 32, 32*8, TRANSPOSE, FORMAT>(A, out, rows, cols, tiledCols, outRows, outCols, item_ct1, smem_data_acc_ct1.get_pointer(), dacc_A, dacc_out);
        });
    });
  
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

             
                
sycl::half A[512];
float rowStats[512];
float colStats[512];
char out_col_normed[512];
char out_row_normed[512];
int rowidx[512];
int colidx[512];
sycl::half val[512];
int nnz_block_ptr[512];


for(int i=0;i<512;i++){ A[i]=sycl::half(1);rowStats[i]=0.5f;colStats[i]=0.5f;out_col_normed[i]=1;out_row_normed[i]=1;rowidx[i]=i-1;colidx[i]=i-1;nnz_block_ptr[i]=1;val[i]=sycl::half(0.5f);}

float threshold  = 1.0f ;
int rows =16;
int cols= rows;

//====test double row col quant======
doubleRowColQuant(A, rowStats, colStats, out_col_normed,out_row_normed, rowidx, colidx, val, nnz_block_ptr, threshold, rows, cols);

//====test extract outlier=========

char Ad[512];
int idx[512];
char out[512];
for(int i=0;i<512;i++){ A[i]=1;idx[i]=i-1;out[i]=1;}
int idx_size=512;

extractOutliers<1>(Ad, idx, out, idx_size, rows, cols);
//SEG FAULT ERROR
//extractOutliers<2>(Ad, idx, out, idx_size, rows, cols);

//====test transform================

transformRowToFormat<1, 0>(Ad, out, rows, cols);
//SEG FAULT ERROR
//transformRowToFormat<2, 0>(Ad, out, rows, cols);

return 0;

}
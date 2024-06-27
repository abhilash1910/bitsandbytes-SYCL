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

#define BLOCKSIZE_2STATE 2048
#define NUM_2STATE 8
#define BLOCKSIZE_1STATE 2048
#define NUM_1STATE 8


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

//========================================k dequant mm int32fp16===================


#define MM_DEQUANT_CONST 6.200012e-05f //1.0f/(127.0f*127.0f)

template <int ITEMS_PER_THREAD, int SUBTILE_ROWS, int THREADS>void kdequant_mm_int32_fp16(int *__restrict__ const A, float *__restrict__ const rowStats, float *__restrict__ const colStats, sycl::half *out, float* newRowStats, float* newcolStats, sycl::half *__restrict__ const bias, const int numRows, const int numCols, const int tileCols, const int n,  const sycl::nd_item<3> &item_ct1, float *smem_rowStats, const sycl_la &tacc, const sycl_dacc &dacc_A, const sycl_dacc_float &dacc_rowStats, const sycl_dacc_float &dacc_colStats, const sycl::accessor<sycl::half, 1> &dacc_out, sycl::half *dacc_bias )
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
  
  using group_load_int = dpct_::group::workgroup_load<ITEMS_PER_THREAD, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, int,  int *, sycl::nd_item<3>>;
  using group_exchange = exchange<int, ITEMS_PER_THREAD>;
  
  auto *d_A = dacc_A.get_multi_ptr<sycl::access::decorated::yes>().get();
  auto *tmp = tacc.get_multi_ptr<sycl::access::decorated::yes>().get();
   //dacc_colStats //dacc_bias //dacc_rowStats

  // L1. Load sub-tile row/col statistics. Each thread only holds 1 col, load rows into shared memory.
  float colStat = col >= numCols ? 0.0f : dacc_colStats[col];
  float local_biasValue = ((dacc_bias == NULL) || (col >= numCols)) ? 0.0f : sycl::vec<sycl::half, 1>(dacc_bias[col]).convert<float, sycl::rounding_mode::automatic>()[0];
  // no block loads for rows for now -- keep it simple
  for(int j = item_ct1.get_local_id(2); j < SUBTILE_ROWS; j+=item_ct1.get_local_range(2))
  {
    // todo: is this global mem access slow due to overlaps or does the L1 cache work well here?
    int row = (base_row+j) % numRows; // wrap around
    // each warp accesses the same element, for four consequitive elements
    // todo: update description about striped shared memory, it is not needed
    // rowidx: [0, 1, 2, 3...] and each warp reads ITEMS_PER_THREAD consequitive elements
    smem_rowStats[j] = dacc_rowStats[row];
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
        dacc_out[outIdx] = local_output[j];
    }

    row_offset += rows_per_load;
  }
}

//==========================dequant mm int 32 fp16==========================

void dequant_mm_int32_fp16(int *A, float *rowStats, float *colStats, sycl::half *out, float* newRowStats, float* newcolStats, sycl::half *bias, int numRows, int numCols)
{
  int threads = 512;
  int tileCols = fill_up_to_nearest_multiple(numCols, 32);
  int n = numRows*tileCols;
  int subtile_rows = 128;
  int tilesize = 32*subtile_rows;
  int num_blocks = numRows/subtile_rows;
  num_blocks += (numRows % subtile_rows == 0) ? 0 : 1;
  num_blocks = num_blocks*(tileCols/32);
  assert(threads <= tilesize);
  int size = NUM_BLOCK;
  
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();

  
  sycl::buffer<int, 1> buff_A (A, sycl::range<1>(size));
  sycl::buffer<float, 1> buff_rowStats (rowStats, sycl::range<1>(size));
  sycl::buffer<float, 1> buff_colStats (colStats, sycl::range<1>(size));
  sycl::buffer<sycl::half, 1> buff_out (out, sycl::range<1>(size));
  //sycl::buffer<sycl::half, 1> buff_bias (bias, sycl::range<1>(size));
  
  
  
  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
    q_ct1.submit(
		[&](sycl::handler &cgh) {
            
          using group_load = dpct_::group::workgroup_load<NUM_ESTIMATE, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, int,  int *, sycl::nd_item<3>>;
          size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);
          sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);  
          sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
          
          sycl::accessor dacc_rowStats(buff_rowStats, cgh, sycl::read_write);
          sycl::accessor dacc_colStats(buff_colStats, cgh, sycl::read_write);
          sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
          //sycl::accessor dacc_bias(buff_bias, cgh, sycl::read_write);
          
            
            //__shared__ vars
            sycl::local_accessor<float, 1> smem_rowStats_acc_ct1(sycl::range<1>(256), cgh);
            sycl::local_accessor<sycl::half, 1> dacc_bias(sycl::range<1>(size), cgh);
            
			      cgh.parallel_for(
			        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, BLOCKSIZE_1STATE/NUM_1STATE), sycl::range<3>(1, 1, BLOCKSIZE_1STATE/NUM_1STATE)), 
			        [=](sycl::nd_item<3> item_ct1) {
  kdequant_mm_int32_fp16<4, 128, 512>(A, rowStats, colStats, out, newRowStats, newcolStats, bias, numRows, numCols, tileCols, n, item_ct1,smem_rowStats_acc_ct1.get_pointer(), tacc, dacc_A, dacc_rowStats, dacc_colStats, dacc_out, dacc_bias.get_pointer() );
           });
  
  });
  
}


int main(){

             
                
int A[512];
float rowStats[512];
float colStats[512];
float newRowStats[512];
float newColStats[512];
sycl::half out[512], bias[512];


for(int i=0;i<512;i++){ A[i]=1;rowStats[i]=0.5f;colStats[i]=0.5f;newRowStats[i]=1.0f;newColStats[i]=1.0f;bias[i]=sycl::half(0.5f);out[i]=sycl::half(0.5f);}

float threshold  = 1.0f ;
int numRows =16;
int numCols= numRows;

dequant_mm_int32_fp16(A, rowStats, colStats, out, newRowStats, newColStats, bias, numRows, numCols);

return 0;

}
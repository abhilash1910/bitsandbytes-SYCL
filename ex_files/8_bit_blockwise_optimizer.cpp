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
using namespace dnnl;

typedef sycl::ext::oneapi::bfloat16 bf16;
typedef sycl::local_accessor<uint8_t ,1> sycl_la;

typedef sycl::accessor<int, 1> sycl_dacc;
typedef sycl::accessor<float, 1> sycl_dacc_float;
typedef sycl::accessor<unsigned char, 1> sycl_dacc_uc;


template <typename T> __dpct_inline__ int sgn(T val)
{
  return (T(0) < val) - (val < T(0));
}



//Use algorithm from dpct directly as already mentioned in #include <dpct/dpl_extras/dpcpp_extensions.h> 
//This load & store implementation is placed for reference (in case needed) -> already present in dpct master



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


template <int SIGNED>
__dpct_inline__ unsigned char quantize_2D(float *__restrict__ quadrants, float x )
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
        val = i >= 64 ? quadrants[local_pivot] : 0;
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



//========================helper-==============================









#define NUM8BIT 16
#define NUM_THREADS 512
#define NUM_PER_BLOCK 512



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
    const float correction1 = 1.0f - dpct::pow(beta1, step);
    const float correction2 = sycl::sqrt(1.0f -dpct::pow(beta2, step));
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
    
    
    
     using group_load = dpct_::group::workgroup_load<N_PER_TH, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
  using group_load_uc = dpct_::group::workgroup_load<N_PER_TH, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
  
  using group_store = dpct_::group::workgroup_store<N_PER_TH, dpct_::group::store_algorithm::BLOCK_STORE_DIRECT, T,  T *, sycl::nd_item<3>>;
  using group_store_uc = dpct_::group::workgroup_store<N_PER_TH, dpct_::group::store_algorithm::BLOCK_STORE_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
  
  
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

     using group_load = dpct_::group::workgroup_load<N_PER_TH, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
  using group_load_uc = dpct_::group::workgroup_load<N_PER_TH, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
  
  using group_store = dpct_::group::workgroup_store<N_PER_TH, dpct_::group::store_algorithm::BLOCK_STORE_DIRECT, T,  T *, sycl::nd_item<3>>;
  using group_store_uc = dpct_::group::workgroup_store<N_PER_TH, dpct_::group::store_algorithm::BLOCK_STORE_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
  
  
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




#define BLOCKSIZE_2STATE 512
#define NUM_2STATE 8
#define BLOCKSIZE_1STATE 512
#define NUM_1STATE 8

template<typename T, int OPTIMIZER> void optimizerStatic8bitBlockwise(T* p, T* g,
                unsigned char* state1, unsigned char* state2, float beta1, float beta2, float eps, int step, float lr,
                float* quantiles1, float* quantiles2, float* absmax1, float* absmax2, float weight_decay, const float gnorm_scale, bool skip_zeros, int n)
 try {
    
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
	int num_blocks = 0;
  int size = NUM_BLOCK;
  
  sycl::buffer<T, 1> buff_g(g,sycl::range<1>(size));
  sycl::buffer<T, 1> buff_p(p,sycl::range<1>(size));
  sycl::buffer<unsigned char, 1> buff_state1(state1,sycl::range<1>(size));
  sycl::buffer<unsigned char, 1> buff_state2(state2,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_quantiles1(quantiles1,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_quantiles2(quantiles2,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_absmax1(absmax1,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_absmax2(absmax2,sycl::range<1>(size));
  
   
	switch(OPTIMIZER)
	{
		case 0:
			num_blocks = n/BLOCKSIZE_2STATE;
			num_blocks = n % BLOCKSIZE_2STATE == 0 ? num_blocks : num_blocks + 1;
			{
			  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
			     
            using group_load = dpct_::group::workgroup_load<NUM_2STATE, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);
             sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
             sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
             sycl::accessor dacc_p(buff_p, cgh, sycl::read_write);
             sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
             sycl::accessor dacc_state2(buff_state2, cgh, sycl::read_write);
             sycl::accessor dacc_quantiles1(buff_quantiles1, cgh, sycl::read_write);
             sycl::accessor dacc_quantiles2(buff_quantiles2, cgh, sycl::read_write);
             sycl::accessor dacc_absmax1(buff_absmax1, cgh, sycl::read_write);
             sycl::accessor dacc_absmax2(buff_absmax2, cgh, sycl::read_write);
            
            
            
            //__shared__ vars
            sycl::local_accessor<float, 2> smem_quantiles1_acc_ct1(sycl::range<2>(2/*LANES*/, 257), cgh);        
			      sycl::local_accessor<float, 2> smem_quantiles2_acc_ct1(sycl::range<2>(2/*LANES*/, 257), cgh);
			      sycl::local_accessor<float, 1> smem_exchange1_acc_ct1(sycl::range<1>(1), cgh);
			      sycl::local_accessor<float, 1> smem_exchange2_acc_ct1(sycl::range<1>(1), cgh);
			      
     
			      cgh.parallel_for(
			        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, BLOCKSIZE_2STATE/NUM_2STATE), sycl::range<3>(1, 1, BLOCKSIZE_2STATE/NUM_2STATE)), 
			        [=](sycl::nd_item<3> item_ct1) {
			          kOptimizerStatic8bit2StateBlockwise<T, OPTIMIZER, BLOCKSIZE_2STATE, NUM_2STATE>(p, g, state1, state2, beta1, beta2, eps, step, lr, quantiles1, quantiles2, absmax1, absmax2, weight_decay, gnorm_scale, skip_zeros, n,item_ct1,  smem_quantiles1_acc_ct1, smem_quantiles2_acc_ct1,smem_exchange1_acc_ct1.get_pointer(), smem_exchange2_acc_ct1.get_pointer(), tacc, dacc_g, dacc_p, dacc_state1, dacc_state2, dacc_quantiles1, dacc_quantiles2, dacc_absmax1, dacc_absmax2);
			        });
			    });
			}
		
		break;
		case 1:
		case 2:
    case 3:
    case 4:
			num_blocks = n/BLOCKSIZE_1STATE;
			num_blocks = n % BLOCKSIZE_1STATE == 0 ? num_blocks : num_blocks + 1;
			{
			  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
			      
            using group_load = dpct_::group::workgroup_load<NUM_1STATE, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);
             sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
             sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
             sycl::accessor dacc_p(buff_p, cgh, sycl::read_write);
             sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
             sycl::accessor dacc_quantiles1(buff_quantiles1, cgh, sycl::read_write);
             sycl::accessor dacc_absmax1(buff_absmax1, cgh, sycl::read_write);
             
             
            //__shared__ vars
            sycl::local_accessor<float, 2> smem_quantiles1_acc_ct1(sycl::range<2>(2/*LANES*/, 257), cgh);
			      sycl::local_accessor<float, 1> smem_exchange1_acc_ct1(sycl::range<1>(1), cgh);
            
			      cgh.parallel_for(
			        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, BLOCKSIZE_1STATE/NUM_1STATE), sycl::range<3>(1, 1, BLOCKSIZE_1STATE/NUM_1STATE)), 
			        [=](sycl::nd_item<3> item_ct1) {
			          kOptimizerStatic8bit1StateBlockwise<T, OPTIMIZER, BLOCKSIZE_1STATE, NUM_1STATE>(p, g, state1, beta1, beta2, eps, step, lr, quantiles1, absmax1, weight_decay, gnorm_scale, skip_zeros, n, item_ct1, smem_quantiles1_acc_ct1, smem_exchange1_acc_ct1.get_pointer(), tacc, dacc_g, dacc_p, dacc_state1,  dacc_quantiles1, dacc_absmax1);
			        });
			    });
			}
			
		break;
	}

}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}





int main(){

             
                
float g[512];
float p[512];
unsigned char state1[512];
unsigned char state2[512];
float unorm[512];
float quantiles1[512];
float quantiles2[512];
float max1[512];
float max2[512];
float new_max1[512];
float new_max2[512];
float absmax1[512];
float absmax2[512];

for(int i=0;i<512;i++){ g[i]=1.0f;p[i]=1.0f;state1[i]=1;state2[i]=1;unorm[i]=0.75f;quantiles1[i]=0.5f;quantiles2[i]=0.5f;max1[1]=1.0f;max2[i]=0.0f;new_max1[i]=0.0f;new_max2[i]=0.0f;absmax1[i]=0.2f;absmax2[i]=0.0f;}
float max_unorm  = 1.0f ;
float param_norm = max_unorm; const float beta1 = max_unorm; const float beta2= max_unorm;
const float eps  = 0.5f;const float weight_decay = eps;
const int step = 2;
const float lr =  0.9f;const float gnorm_scale = lr; 
const int n =512;
bool skip_zeros = false;




optimizerStatic8bitBlockwise<float, 0>(p, g,
                state1, state2, beta1, beta2, eps, step, lr,
                quantiles1, quantiles2, absmax1,  absmax2,
                weight_decay, gnorm_scale, skip_zeros, n);




return 0;

}
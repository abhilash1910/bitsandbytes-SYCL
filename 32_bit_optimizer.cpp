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
using namespace dnnl;

typedef sycl::ext::oneapi::bfloat16 bf16;
typedef sycl::local_accessor<uint8_t ,1> sycl_la;

typedef sycl::accessor<int, 1> sycl_dacc;
typedef sycl::accessor<float, 1> sycl_dacc_float;



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








//#define dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, int,  int *, sycl::nd_item<3>> wg_load;

template<typename T, int OPTIMIZER, int BLOCK_SIZE, int NUM_VALS>
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

  const float correction1 = 1.0f/(1.0f - dpct::pow(beta1, step));
  const float correction2 = 1.0f/(1.0f - dpct::pow(beta2, step));
  
  using group_load = dpct_::group::workgroup_load<NUM_ESTIMATE, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
  using group_load_float = dpct_::group::workgroup_load<NUM_ESTIMATE, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;  

  
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
              case 1:
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

  const float correction1 = 1.0f - dpct::pow(beta1, step);
  const float correction2 = sycl::sqrt(1.0f - dpct::pow(beta2, step));
  const float step_size = -lr*correction2/correction1;

  if(max_unorm > 0.0f)
  {
    update_scale = max_unorm > 0.0f ? sycl::sqrt(dacc_unorm[0]) : 1.0f;
    if(update_scale > max_unorm*param_norm){ update_scale = (max_unorm*param_norm)/update_scale; }
    else{ update_scale = 1.0f; }
  }
  else{ update_scale = 1.0f; }
  
  using group_load = dpct_::group::workgroup_load<NUM_PER_THREAD, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
  using group_load_float = dpct_::group::workgroup_load<NUM_PER_THREAD, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
  
  using group_store = dpct_::group::workgroup_store<NUM_PER_THREAD, dpct_::group::store_algorithm::BLOCK_STORE_DIRECT, T,  T *, sycl::nd_item<3>>;
  using group_store_float = dpct_::group::workgroup_store<NUM_PER_THREAD, dpct_::group::store_algorithm::BLOCK_STORE_DIRECT, float,  float *, sycl::nd_item<3>>;
  
  
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
              case 1:
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
  
  
  
  
  using group_load = dpct_::group::workgroup_load<NUM_VALS, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
  using group_load_float = dpct_::group::workgroup_load<NUM_VALS, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
  
  
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
              case 0:
                  if(step == 1)
                    s1_vals[j] = (float)g_vals[j]; // state update
                  else
                    s1_vals[j] = s1_vals[j]*beta1 + ((float)g_vals[j]); // state update
                  s1_vals[j] = s1_vals[j]*s1_vals[j]; // update norm
                  break;
              case 1:
                  s1_vals[j] = s1_vals[j]*beta2 + ((1.0f-beta2)*(float)g_vals[j]); // state update
                  break;
              case 2:
                  s1_vals[j] = s1_vals[j]*beta1 + ((1.0f-beta1)*((float)g_vals[j])*((float)g_vals[j])); // state update
                  s1_vals[j] = (float)g_vals[j] / (sycl::sqrt(s1_vals[j])+eps); // update value
                  s1_vals[j] = s1_vals[j]*s1_vals[j]; // update norm
                  break;
              case 3:
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
  
  using group_load = dpct_::group::workgroup_load<NUM_PER_THREAD, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
  using group_load_float = dpct_::group::workgroup_load<NUM_PER_THREAD, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
  
  using group_store = dpct_::group::workgroup_store<NUM_PER_THREAD, dpct_::group::store_algorithm::BLOCK_STORE_DIRECT, T,  T *, sycl::nd_item<3>>;
  using group_store_float = dpct_::group::workgroup_store<NUM_PER_THREAD, dpct_::group::store_algorithm::BLOCK_STORE_DIRECT, float,  float *, sycl::nd_item<3>>;
  
  
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
								case 0:
										if(step == 1)
											s1_vals[j] = (float)g_vals[j];
										else
											s1_vals[j] = s1_vals[j]*beta1 + ((float)g_vals[j]);

										p_vals[j] = ((float)p_vals[j]) + update_scale*(-lr*(s1_vals[j]));
										break;
								case 1:
										p_vals[j] = ((float)p_vals[j]) - update_scale*(lr*sgn(((float)s1_vals[j])*beta1 + ((1.0f-beta1)*((float)g_vals[j]))));
										s1_vals[j] = s1_vals[j]*beta2 + ((1.0f-beta2)*((float)g_vals[j]));
										break;
								case 2:
										s1_vals[j] = s1_vals[j]*beta1 + ((1.0f-beta1)*((float)g_vals[j])*((float)g_vals[j]));
										p_vals[j] = ((float)p_vals[j]) - update_scale*(lr*(float)g_vals[j] / (sycl::sqrt((float)s1_vals[j])+eps));
										break;
								case 3:
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

/*
template<typename T, int OPTIMIZER> void optimizer32bit(T* g, T* p,
                float* state1, float* state2, float *unorm, float max_unorm, float param_norm,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const float gnorm_scale, bool skip_zeros, const int n)
 try {
  dpct_::device_ext &dev_ct1 = dpct_::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
  int num_blocks = n/4096;
  num_blocks = n % 4096 == 0 ? num_blocks : num_blocks + 1;
  int size= NUM_BLOCK;  
   
  
  sycl::buffer<T, 1> buff_g(g,sycl::range<1>(size));
  sycl::buffer<T, 1> buff_p(p,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_state1(state1,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_state2(state2,sycl::range<1>(size));
  
  
	switch(OPTIMIZER)
	{
		case 1:
      if(max_unorm > 0.0f)
			{
				
        {
          //dpct_::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
          q_ct1.submit(
            [&](sycl::handler &cgh) {
              
               using group_load = dpct_::group::workgroup_load<NUM_ESTIMATE, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
               using group_load_float = dpct_::group::workgroup_load<NUM_ESTIMATE, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
               size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
       
              sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
              sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
              sycl::accessor dacc_p(buff_p, cgh, sycl::read_write);
              sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
              sycl::accessor dacc_state2(buff_state2, cgh, sycl::read_write);
         
              
             
              cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
                [=](sycl::nd_item<3> item_ct1) {
                  kPreconditionOptimizer32bit2State<T, OPTIMIZER, 4096, 8>(g, p, state1, state2, unorm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, n, item_ct1, tacc, dacc_state1, dacc_state2, dacc_g);
                });
            });
        }
        
      }
      
      
			 { //dpct_::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
			      
            
            using group_load = dpct_::group::workgroup_load<NUM_PER_THREAD, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
            using group_load_float = dpct_::group::workgroup_load<NUM_PER_THREAD, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
            using group_store = dpct_::group::workgroup_store<NUM_PER_THREAD, dpct_::group::store_algorithm::BLOCK_STORE_DIRECT, T,  T *, sycl::nd_item<3>>;
            using group_store_float = dpct_::group::workgroup_store<NUM_PER_THREAD, dpct_::group::store_algorithm::BLOCK_STORE_DIRECT, float,  float *, sycl::nd_item<3>>; 
             
             
            size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
            sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
            sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
            sycl::accessor dacc_p(buff_p, cgh, sycl::read_write);
            sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
            sycl::accessor dacc_state2(buff_state2, cgh, sycl::read_write);     
       
              
			      //sycl::local_accessor<uint8_t[sizeof(type_ct3)], 0> temp_storage_ct1_acc_ct1(cgh);

			      cgh.parallel_for(
			        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
			        [=](sycl::nd_item<3> item_ct1) {
			          kOptimizer32bit2State<T, OPTIMIZER>(g, p, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, skip_zeros, n, item_ct1, tacc, dacc_g, dacc_p, dacc_state1, dacc_state2);
			        });
			    });	
		}
   break;
 case 2:
      if(max_unorm > 0.0f)
			{
				//DPCT_CHECK_ERROR(q_ct1.memset(unorm, 0, 1*sizeof(float)).wait());
				
				{
				  //dpct_::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
				  q_ct1.submit(
				    [&](sycl::handler &cgh) {
				      
                                  
             using group_load = dpct_::group::workgroup_load<NUM_PER_THREAD, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             using group_load_float = dpct_::group::workgroup_load<NUM_PER_THREAD, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
              size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
            sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
              sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
             sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
            
                                
				      //sycl::local_accessor<uint8_t[sizeof(type_ct4)], 0> temp_storage_ct1_acc_ct1(cgh);

				      cgh.parallel_for(
				        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
				        [=](sycl::nd_item<3> item_ct1) {
				          kPreconditionOptimizer32bit1State<T, OPTIMIZER, 4096, 4>(g, p, state1, unorm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, n, item_ct1, tacc, dacc_g, dacc_state1);
				        });
				    });
				}
      }
      
      q_ct1.submit(
			    [&](sycl::handler &cgh) {
             using group_load = dpct_::group::workgroup_load<NUM_PER_THREAD, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             using group_load_float = dpct_::group::workgroup_load<NUM_PER_THREAD, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
              size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
            sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
             sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
             sycl::accessor dacc_p(buff_p, cgh, sycl::read_write);
       
             sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
         
			       cgh.parallel_for(
			        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
			        [=](sycl::nd_item<3> item_ct1) {
			          kOptimizer32bit1State<T, OPTIMIZER>(g, p, state1, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, skip_zeros, n, item_ct1, tacc, dacc_g, dacc_p, dacc_state1);
			        });
			    });
      
   break;
 case 3:
 
     {
        //dpct_::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
        q_ct1.submit(
          [&](sycl::handler &cgh) {
            
              using group_load = dpct_::group::workgroup_load<NUM_PER_THREAD, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             using group_load_float = dpct_::group::workgroup_load<NUM_PER_THREAD, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
              size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
            sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
             sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
             sycl::accessor dacc_p(buff_p, cgh, sycl::read_write);
       
             sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
              
            cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
              [=](sycl::nd_item<3> item_ct1) {
                kOptimizer32bit1State<T, OPTIMIZER>(g, p, state1, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, skip_zeros, n, item_ct1, tacc, dacc_g, dacc_p, dacc_state1);
              });
          });
      }
     

      if(max_unorm > 0.0f)
      {
        //DPCT_CHECK_ERROR(q_ct1.memset(unorm, 0, 1*sizeof(float)).wait());
        
        {
          //dpct_::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
          q_ct1.submit(
            [&](sycl::handler &cgh) {
              using group_load = dpct_::group::workgroup_load<NUM_PER_THREAD, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             using group_load_float = dpct_::group::workgroup_load<NUM_PER_THREAD, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
              size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
            sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
             sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
             sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
                         
              cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
                [=](sycl::nd_item<3> item_ct1) {
                  kPreconditionOptimizer32bit1State<T, OPTIMIZER, 4096, 4>(g, p, state1, unorm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, n, item_ct1, tacc, dacc_g, dacc_state1);
                });
            });
        }
        
      }
      break;   
        
  }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
*/




template<typename T, int OPTIMIZER> void optimizer32bit(T* g, T* p,
                float* state1, float* state2, float *unorm, float max_unorm, float param_norm,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const float gnorm_scale, bool skip_zeros, const int n)
 try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
  int num_blocks = n/4096;
  num_blocks = n % 4096 == 0 ? num_blocks : num_blocks + 1;
  int size= NUM_BLOCK;  
   
  sycl::buffer<T, 1> buff_g(g,sycl::range<1>(size));
  sycl::buffer<T, 1> buff_p(p,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_state1(state1,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_state2(state2,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_unorm(unorm, sycl::range<1>(size));
  
	switch(OPTIMIZER)
	{
		case 0:
      if(max_unorm > 0.0f)
			{
				
        //DPCT_CHECK_ERROR(q_ct1.memset(unorm, 0, 1*sizeof(float)).wait());
       
        {
          //dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
          q_ct1.submit(
            [&](sycl::handler &cgh) {
              
              using group_load = dpct_::group::workgroup_load<NUM_ESTIMATE, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
              using group_load_float = dpct_::group::workgroup_load<NUM_ESTIMATE, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
              
              
              size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
              sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
              
              sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
              sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
              sycl::accessor dacc_state2(buff_state2, cgh, sycl::read_write);
              sycl::accessor dacc_unorm(buff_unorm, cgh, sycl::read_write);
              
              
              cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
                [=](sycl::nd_item<3> item_ct1) {
                  kPreconditionOptimizer32bit2State<T, OPTIMIZER, 4096, 8>(g, p, state1, state2, unorm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, n, item_ct1, tacc, dacc_state1, dacc_state2, dacc_g, dacc_unorm);
                });
            });
        }
        
      }
			
			{
			  //dpct_::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
			      
            
            using group_load = dpct_::group::workgroup_load<NUM_PER_THREAD, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
            using group_load_float = dpct_::group::workgroup_load<NUM_PER_THREAD, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
             
            size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
            sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
            sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
            sycl::accessor dacc_p(buff_p, cgh, sycl::read_write);
            sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
            sycl::accessor dacc_state2(buff_state2, cgh, sycl::read_write);   
            sycl::accessor dacc_unorm(buff_unorm, cgh, sycl::read_write);
                
           
			      cgh.parallel_for(
			        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
			        [=](sycl::nd_item<3> item_ct1) {
			          kOptimizer32bit2State<T, OPTIMIZER>(g, p, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, skip_zeros, n, item_ct1, tacc, dacc_g, dacc_p, dacc_state1, dacc_state2, dacc_unorm);
			        });
			    });
			}
      
			break;
		case 1:
    case 2:
    case 3:
      if(max_unorm > 0.0f)
			{
				//DPCT_CHECK_ERROR(q_ct1.memset(unorm, 0, 1*sizeof(float)).wait());
				
				{
				  //dpct_::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
				  q_ct1.submit(
				    [&](sycl::handler &cgh) {
                                  
             using group_load = dpct_::group::workgroup_load<NUM_ESTIMATE, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             using group_load_float = dpct_::group::workgroup_load<NUM_ESTIMATE, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
              size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
            sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
             sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
             sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
             sycl::accessor dacc_unorm(buff_unorm, cgh, sycl::read_write);
              
                                
				      
		         cgh.parallel_for(
				        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
				        [=](sycl::nd_item<3> item_ct1) {
				          kPreconditionOptimizer32bit1State<T, OPTIMIZER, 4096, 8>(g, p, state1, unorm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, n, item_ct1, tacc, dacc_g, dacc_state1, dacc_unorm);
				        });
				    });
				}
      }  

			
			{
			  //dpct_::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
             using group_load = dpct_::group::workgroup_load<NUM_PER_THREAD, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             using group_load_float = dpct_::group::workgroup_load<NUM_PER_THREAD, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
              size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
            sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
             sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
             sycl::accessor dacc_p(buff_p, cgh, sycl::read_write);
       
             sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
             sycl::accessor dacc_unorm(buff_unorm, cgh, sycl::read_write);
              
         
			       cgh.parallel_for(
			        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
			        [=](sycl::nd_item<3> item_ct1) {
			          kOptimizer32bit1State<T, OPTIMIZER>(g, p, state1, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, skip_zeros, n, item_ct1, tacc, dacc_g, dacc_p, dacc_state1, dacc_unorm);
			        });
			    });
			}
      
			break;
    case 4:
      // in lion, the momentum update after the parameter update
      
      {
        //dpct_::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
        q_ct1.submit(
          [&](sycl::handler &cgh) {
            
              using group_load = dpct_::group::workgroup_load<NUM_PER_THREAD, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             using group_load_float = dpct_::group::workgroup_load<NUM_PER_THREAD, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
              size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
            sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
             sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
             sycl::accessor dacc_p(buff_p, cgh, sycl::read_write);
       
             sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
             sycl::accessor dacc_unorm(buff_unorm, cgh, sycl::read_write);
              
              
            cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
              [=](sycl::nd_item<3> item_ct1) {
                kOptimizer32bit1State<T, OPTIMIZER>(g, p, state1, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, skip_zeros, n, item_ct1, tacc, dacc_g, dacc_p, dacc_state1, dacc_unorm);
              });
          });
      }
     

      if(max_unorm > 0.0f)
      {
        //DPCT_CHECK_ERROR(q_ct1.memset(unorm, 0, 1*sizeof(float)).wait());
        
        {
          //dpct_::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
          q_ct1.submit(
            [&](sycl::handler &cgh) {
              using group_load = dpct_::group::workgroup_load<NUM_ESTIMATE, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             using group_load_float = dpct_::group::workgroup_load<NUM_ESTIMATE, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
              size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
            sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
             sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
             sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
             sycl::accessor dacc_unorm(buff_unorm, cgh, sycl::read_write);
              
                         
              cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
                [=](sycl::nd_item<3> item_ct1) {
                  kPreconditionOptimizer32bit1State<T, OPTIMIZER, 4096, 8>(g, p, state1, unorm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, n, item_ct1, tacc, dacc_g, dacc_state1, dacc_unorm);
                });
            });
        }
        
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
float state1[512];
float state2[512];
float unorm[512];
for(int i=0;i<512;i++){ g[i]=1.0f;p[i]=1.0f;state1[i]=0.5f;state2[i]=0.5f;unorm[i]=0.75f;}
float max_unorm  = 1.0f ;
float param_norm = max_unorm; const float beta1 = max_unorm; const float beta2= max_unorm;
const float eps  = 0.5f;const float weight_decay = eps;
const int step = 2;
const float lr =  0.9f;const float gnorm_scale = lr; 
const int n =512;
bool skip_zeros = false;

optimizer32bit<float, 0>(g, p,
                state1, state2, unorm, max_unorm, param_norm,
                beta1, beta2, eps, weight_decay,
                 step,  lr, gnorm_scale, skip_zeros, n);

optimizer32bit<float, 1>(g, p,
                state1, state2, unorm, max_unorm, param_norm,
                beta1, beta2, eps, weight_decay,
                 step,  lr, gnorm_scale, skip_zeros, n);

optimizer32bit<float, 2>(g, p,
                state1, state2, unorm, max_unorm, param_norm,
                beta1, beta2, eps, weight_decay,
                 step,  lr, gnorm_scale, skip_zeros, n);

optimizer32bit<float, 3>(g, p,
                state1, state2, unorm, max_unorm, param_norm,
                beta1, beta2, eps, weight_decay,
                 step,  lr, gnorm_scale, skip_zeros, n);

optimizer32bit<float, 4>(g, p,
                state1, state2, unorm, max_unorm, param_norm,
                beta1, beta2, eps, weight_decay,
                 step,  lr, gnorm_scale, skip_zeros, n);


/*
int data[512];
for(int i=0;i<512;i++){data[i]=i;}
float offset =1.0f;
float code[512];
for(int i=0;i<512;i++){code[i]=1.0f;}

estimateQuantiles<int>(data, code, offset, 512);
*/
return 0;

}
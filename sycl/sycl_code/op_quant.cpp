// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_extras/dpcpp_extensions.h>
#include "ops.h"
#include "kernel_quant.h"
#include <limits>
#include <BinSearch.h>
#include <cassert>
#include <common.h>
#include <dpct/lib_common_utils.hpp>

#include "oneapi/dnnl/dnnl.hpp"

#define ERR_NOT_IMPLEMENTED 100

#define HLF_MAX 65504
#define TH 1024
#define NUM 4
#define NUM_BLOCK 4096

#define THREADS_ESTIMATE 512
#define NUM_ESTIMATE 8
#define BLOCK_ESTIMATE 4096
#define NUM_PER_THREAD 4
using namespace dnnl;

typedef sycl::ext::oneapi::bfloat16 bf16;

using namespace BinSearch;
using std::cout;
using std::endl;

int fill_up_to_nearest_multiple(int value, int multiple)
{
  return value + (value % multiple == 0 ? 0 : (multiple - (value % multiple)));
}

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


void histogramScatterAdd2D(float* histogram, int *index1, int *index2, float *src, int maxidx1, int n)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  
  int threads = 512;
  int num_blocks = n/threads;
  num_blocks = n % threads == 0 ? num_blocks : num_blocks + 1;
  int size = NUM_BLOCK;
	
  
  sycl::buffer<float, 1> buff_histogram(histogram,sycl::range<1>(size));
  sycl::buffer<int, 1> buff_index1(index1,sycl::range<1>(size));
  sycl::buffer<int, 1> buff_index2(index2,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_src(src,sycl::range<1>(size));
  
  
  {
  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
    q_ct1.submit(
    [&](sycl::handler &cgh) {
    
     sycl::accessor dacc_histogram(buff_histogram, cgh, sycl::read_write);
     sycl::accessor dacc_index1(buff_index1, cgh, sycl::read_write);
     sycl::accessor dacc_index2(buff_index2, cgh, sycl::read_write);
     sycl::accessor dacc_src(buff_src, cgh, sycl::read_write);
     
    
    cgh.parallel_for(
    sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
    [=](sycl::nd_item<3> item_ct1) {
      kHistogramScatterAdd2D(histogram, index1, index2, src, maxidx1, n, item_ct1, dacc_histogram, dacc_index1, dacc_index2, dacc_src);
      });
    });
  }

}



//============================estimate quantiles===============================
template <typename T> void estimateQuantiles(T *A, float *code, float offset, int n)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  int num_blocks = n/4096;
  num_blocks = n % 4096 == 0 ? num_blocks : num_blocks + 1;
  std::memset(code, 0, 256*sizeof(float));
  //DPCT_CHECK_ERROR(q_ct1.memset(code, 0, 256*sizeof(float)).wait());
  sycl::context ctx = q_ct1.get_context();
  int size = 512;
  
  
  sycl::buffer<T, 1> buff_A(A,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_code(code,sycl::range<1>(size));
  {
    dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
    q_ct1.submit(
      [&](sycl::handler &cgh) {
      
        using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, int,  int *, sycl::nd_item<3>>;
        //using group_radix_sort = dpct::group::radix_sort<int, NUM_ESTIMATE>;
        size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
        sycl::local_accessor<uint8_t, 1> tacc(sycl::range<1>(temp_storage_size), cgh);
        sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
        sycl::accessor dacc_code(buff_code, cgh, sycl::read_write);
          
        
        auto std_numeric_limits_T_max_ct3 = std::numeric_limits<T>::max();

        cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
          [=](sycl::nd_item<3> item_ct1) {
            kEstimateQuantiles<T>(A, code, offset, std_numeric_limits_T_max_ct3, n, item_ct1, tacc, dacc_A, dacc_code);
            
          });
      });
  }
  
}

//============================k quantize ===============================
void quantize(float *code, float *A, unsigned char *out, int n)
{
  int num_blocks = n/1024;
  num_blocks = n % 1024 == 0 ? num_blocks : num_blocks + 1;
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
  int size = NUM_BLOCK;
  
  sycl::buffer<float, 1> buff_A(A,sycl::range<1>(size));
  sycl::buffer<unsigned char, 1> buff_out(out,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_code(code,sycl::range<1>(size));
  
  {
    dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
    q_ct1.submit(
      [&](sycl::handler &cgh) {
      using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
              
      size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
      sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
          
      sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
      sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
      sycl::accessor dacc_code(buff_code, cgh, sycl::read_write);
      
      //__shared__ vars
      sycl::local_accessor<float, 1> smem_code_acc_ct1(sycl::range<1>(256), cgh);
      
      cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
        [=](sycl::nd_item<3> item_ct1) {
          kQuantize(code, A, out, n, item_ct1, smem_code_acc_ct1.get_pointer(), tacc, dacc_A, dacc_out, dacc_code);
        });
    });
  }
  
}
//============================k dequantize===============================
void dequantize(float *code, unsigned char *A, float *out, int n)
{
  int num_blocks = n/1024;
  num_blocks = n % 1024 == 0 ? num_blocks : num_blocks + 1;
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
  int size = NUM_BLOCK;
  
  unsigned char *buff_A;
  float *buff_out;
  *((void **)&buff_A) = sycl::malloc_device(size, dev_ct1, ctx);
  *((void **)&buff_out) = sycl::malloc_device(size, dev_ct1, ctx);
  q_ct1.memcpy((void*)(buff_out), (void*)(out), NUM_BLOCK);
  q_ct1.memcpy((void*)(buff_A), (void*)(A), NUM_BLOCK);

  
  {
    dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
    q_ct1.submit(
      [&](sycl::handler &cgh) {
      
      //__shared__ vars
      sycl::local_accessor<float, 1> smem_code_acc_ct1(sycl::range<1>(256), cgh);
      
      cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
        [=](sycl::nd_item<3> item_ct1) {
          kDequantize(code, buff_A, buff_out, n, item_ct1, smem_code_acc_ct1.get_pointer());
        });
    });
   
   
   } 
  //back memcpy
  q_ct1.memcpy((void *)(out), (void*)(buff_out), NUM_BLOCK);
  q_ct1.memcpy((void*)(A), (void*)(buff_A), NUM_BLOCK); 
  
}

//============================quantize blockwise===============================

template <typename T, int STOCHASTIC, int DATA_TYPE> void quantizeBlockwise(float * code, T *A, float *absmax, unsigned char *out, float *rand, int rand_offset, int blocksize, const int n)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  int num_blocks = n/blocksize;
  num_blocks = n % blocksize == 0 ? num_blocks : num_blocks + 1;
  sycl::context ctx = q_ct1.get_context();
  int size= NUM_BLOCK;
  for(int i=0; i< NUM_BLOCK; i++){ out[i]=out[(DATA_TYPE > 0) ? i/2 : i];};
  
  
  sycl::buffer<T, 1> buff_A(A,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_rand(rand,sycl::range<1>(size));
  sycl::buffer<unsigned char, 1> buff_out(out,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_code(code,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_absmax(absmax,sycl::range<1>(size));
  
  
  
  if(blocksize == 4096)
    
    
    {
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
        [&](sycl::handler &cgh) {
          
          
           using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
              
          size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
          sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
          
          sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
          sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
          sycl::accessor dacc_rand(buff_rand, cgh, sycl::read_write);
          sycl::accessor dacc_code(buff_code, cgh, sycl::read_write);
          sycl::accessor dacc_absmax(buff_absmax, cgh, sycl::read_write);
          
          
          //__shared__ vars for funtions
          sycl::local_accessor<float, 1> smem_code_acc_ct1(sycl::range<1>(256), cgh);
          sycl::local_accessor<float, 1> smem_absmax_value_acc_ct1(sycl::range<1>(1), cgh);
          
          
          cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
            [=](sycl::nd_item<3> item_ct1) {
              kQuantizeBlockwise<T, 4096, 4, STOCHASTIC, 0>(code, A, absmax, out, rand, rand_offset, n, item_ct1,  smem_code_acc_ct1.get_pointer(), smem_absmax_value_acc_ct1.get_pointer(), tacc, dacc_A, dacc_rand, dacc_out, dacc_code, dacc_absmax);
            });
        });
    }
  else if(blocksize == 2048)
    
    {
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
        [&](sycl::handler &cgh) {
          
          using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
              
          size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
          sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
          
          sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
          sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
          sycl::accessor dacc_rand(buff_rand, cgh, sycl::read_write);
          sycl::accessor dacc_code(buff_code, cgh, sycl::read_write);
          sycl::accessor dacc_absmax(buff_absmax, cgh, sycl::read_write);
          
          //__shared__ vars
          sycl::local_accessor<float, 1> smem_code_acc_ct1(sycl::range<1>(256), cgh);
          sycl::local_accessor<float, 1> smem_absmax_value_acc_ct1(sycl::range<1>(1), cgh);
          
          cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
            [=](sycl::nd_item<3> item_ct1) {
              kQuantizeBlockwise<T, 2048, 4, 0, DATA_TYPE>(code, A, absmax, out, rand, rand_offset, n, item_ct1,  smem_code_acc_ct1.get_pointer(), smem_absmax_value_acc_ct1.get_pointer(), tacc, dacc_A, dacc_rand, dacc_out, dacc_code, dacc_absmax);
            });
        });
    }
  else if(blocksize == 1024)
    {
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
        [&](sycl::handler &cgh) {
          
          using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
              
          size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
          sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
          
          sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
          sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
          sycl::accessor dacc_rand(buff_rand, cgh, sycl::read_write);
          sycl::accessor dacc_code(buff_code, cgh, sycl::read_write);
          sycl::accessor dacc_absmax(buff_absmax, cgh, sycl::read_write);
          
          //__shared__vars
          sycl::local_accessor<float, 1> smem_code_acc_ct1(sycl::range<1>(256), cgh);
          sycl::local_accessor<float, 1> smem_absmax_value_acc_ct1(sycl::range<1>(1), cgh);
          
          cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)), 
            [=](sycl::nd_item<3> item_ct1) {
              kQuantizeBlockwise<T, 1024, 4, 0, DATA_TYPE>(code, A, absmax, out, rand, rand_offset, n, item_ct1,  smem_code_acc_ct1.get_pointer(), smem_absmax_value_acc_ct1.get_pointer(), tacc, dacc_A, dacc_rand, dacc_out, dacc_code, dacc_absmax);
            });
        });
    }
  else if(blocksize == 512)
    {
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
        [&](sycl::handler &cgh) {
          
         
          using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
              
          size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
          sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
          
          sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
          sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
          sycl::accessor dacc_rand(buff_rand, cgh, sycl::read_write);
          sycl::accessor dacc_code(buff_code, cgh, sycl::read_write);
          sycl::accessor dacc_absmax(buff_absmax, cgh, sycl::read_write);
          
          //__shared__ vars
          sycl::local_accessor<float, 1> smem_code_acc_ct1(sycl::range<1>(256), cgh);
          sycl::local_accessor<float, 1> smem_absmax_value_acc_ct1(sycl::range<1>(1), cgh);
          
          cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)), 
            [=](sycl::nd_item<3> item_ct1) {
              kQuantizeBlockwise<T, 512, 2, 0, DATA_TYPE>(code, A, absmax, out, rand, rand_offset, n, item_ct1,  smem_code_acc_ct1.get_pointer(), smem_absmax_value_acc_ct1.get_pointer(), tacc, dacc_A, dacc_rand, dacc_out, dacc_code, dacc_absmax);
            });
        });
    }
  else if(blocksize == 256)
    {
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
        [&](sycl::handler &cgh) {
          
          using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
              
          size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
          sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
          
          sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
          sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
          sycl::accessor dacc_rand(buff_rand, cgh, sycl::read_write);
          sycl::accessor dacc_code(buff_code, cgh, sycl::read_write);
          sycl::accessor dacc_absmax(buff_absmax, cgh, sycl::read_write);
          
          //__shared__ vars
          sycl::local_accessor<float, 1> smem_code_acc_ct1(sycl::range<1>(256), cgh);
          sycl::local_accessor<float, 1> smem_absmax_value_acc_ct1(sycl::range<1>(1), cgh);

          
          cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)), 
            [=](sycl::nd_item<3> item_ct1) {
              kQuantizeBlockwise<T, 256, 2, 0, DATA_TYPE>(code, A, absmax, out, rand, rand_offset, n, item_ct1,  smem_code_acc_ct1.get_pointer(), smem_absmax_value_acc_ct1.get_pointer(), tacc, dacc_A, dacc_rand, dacc_out, dacc_code, dacc_absmax);
            });
        });
    }
  else if(blocksize == 128)
    {
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
        [&](sycl::handler &cgh) {
          
          using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
              
          size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
          sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
          
          sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
          sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
          sycl::accessor dacc_rand(buff_rand, cgh, sycl::read_write);
          sycl::accessor dacc_code(buff_code, cgh, sycl::read_write);
          sycl::accessor dacc_absmax(buff_absmax, cgh, sycl::read_write);
          
          //__shared__ vars
          sycl::local_accessor<float, 1> smem_code_acc_ct1(sycl::range<1>(256), cgh);
          sycl::local_accessor<float, 1> smem_absmax_value_acc_ct1(sycl::range<1>(1), cgh);
          cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)), 
            [=](sycl::nd_item<3> item_ct1) {
              kQuantizeBlockwise<T, 128, 2, 0, DATA_TYPE>(code, A, absmax, out, rand, rand_offset, n, item_ct1,  smem_code_acc_ct1.get_pointer(), smem_absmax_value_acc_ct1.get_pointer(), tacc, dacc_A, dacc_rand, dacc_out, dacc_code, dacc_absmax);
            });
        });
    }
  else if(blocksize == 64)
    {
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
        [&](sycl::handler &cgh) {
          
          using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
              
          size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
          sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
          
          sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
          sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
          sycl::accessor dacc_rand(buff_rand, cgh, sycl::read_write);
           sycl::accessor dacc_code(buff_code, cgh, sycl::read_write);
          sycl::accessor dacc_absmax(buff_absmax, cgh, sycl::read_write);
          
          //__shared__ vars
          sycl::local_accessor<float, 1> smem_code_acc_ct1(sycl::range<1>(256), cgh);
          sycl::local_accessor<float, 1> smem_absmax_value_acc_ct1(sycl::range<1>(1), cgh);
          
      
          cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)), 
            [=](sycl::nd_item<3> item_ct1) {
              kQuantizeBlockwise<T, 64, 2, 0, DATA_TYPE>(code, A, absmax, out, rand, rand_offset, n, item_ct1,  smem_code_acc_ct1.get_pointer(), smem_absmax_value_acc_ct1.get_pointer(), tacc, dacc_A, dacc_rand, dacc_out, dacc_code, dacc_absmax);
            });
        });
    }
  
}


//============================k dequantize blockwise===============================
template<typename T, int DATA_TYPE> void dequantizeBlockwise(float *code, unsigned char *A, float *absmax, T *out, int blocksize, const int n)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  int num_blocks = n/blocksize;
  num_blocks = n % blocksize == 0 ? num_blocks : num_blocks + 1;
  int tile_size = (DATA_TYPE > 0) ? 1024 : 512;
  sycl::context ctx = q_ct1.get_context();
  int size = NUM_BLOCK;
  
  sycl::buffer<unsigned char, 1> buff_A(A,sycl::range<1>(size));
  sycl::buffer<T, 1> buff_out(out,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_code(code,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_absmax(absmax,sycl::range<1>(size));
  
  
  
    
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
      [&](sycl::handler &cgh){
      
              using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, unsigned char,  unsigned char *, sycl::nd_item<3>>;
              
              size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);  
              sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
              
              sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
              sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
              sycl::accessor dacc_code(buff_code, cgh, sycl::read_write);
              sycl::accessor dacc_absmax(buff_absmax, cgh, sycl::read_write);
  
      cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, (n+tile_size-1)/tile_size) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)), 
        [=](sycl::nd_item<3> item_ct1) {
          if(DATA_TYPE > 0){
          kDequantizeBlockwise<T, 512, 64, 8, DATA_TYPE>(code, A, absmax, out, blocksize/2, n, item_ct1, tacc, dacc_A, dacc_out, dacc_code, dacc_absmax); }
          else{
          kDequantizeBlockwise<T, 512, 64, 8, DATA_TYPE>(code, A, absmax, out, blocksize, n, item_ct1, tacc, dacc_A, dacc_out, dacc_code, dacc_absmax);
          }
        });
        
      });
 
}

//void matmul4bite(half *A, unsigned char *B, half*out, int lda, int ldb, int rowsA, int colsA, int colsB)
//{
//	int num_blocks = (colsB+32-1)/32;
//	kMatmul_inference_4bit<NF4, half, half, half><<<num_blocks, 256>>>(A, B, out, lda, ldb, rowsA, colsA, colsB);
//  CUDA_CHECK_RETURN(cudaPeekAtLastError());
//}



//============================32 bit optimizer===============================
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
		case ADAM:
      if(max_unorm > 0.0f)
			{
				std::memset(unorm, 0, 1*sizeof(float));
        //DPCT_CHECK_ERROR(q_ct1.memset(unorm, 0, 1*sizeof(float)).wait());
       
        {
          dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
          q_ct1.submit(
            [&](sycl::handler &cgh) {
              
              using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
              using group_load_float = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
              
              
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
			  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
			      
            
            using group_load = dpct::group::workgroup_load<NUM_PER_THREAD, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
            using group_load_float = dpct::group::workgroup_load<NUM_PER_THREAD, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
             
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
		case MOMENTUM:
    case RMSPROP:
    case ADAGRAD:
      if(max_unorm > 0.0f)
			{
        std::memset(unorm, 0, 1*sizeof(float));
				//DPCT_CHECK_ERROR(q_ct1.memset(unorm, 0, 1*sizeof(float)).wait());
				
				{
				  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
				  q_ct1.submit(
				    [&](sycl::handler &cgh) {
                                  
             using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             using group_load_float = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
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
			  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
             using group_load = dpct::group::workgroup_load<NUM_PER_THREAD, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             using group_load_float = dpct::group::workgroup_load<NUM_PER_THREAD, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
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
    case LION:
      // in lion, the momentum update after the parameter update
      
      {
        dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
        q_ct1.submit(
          [&](sycl::handler &cgh) {
            
              using group_load = dpct::group::workgroup_load<NUM_PER_THREAD, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             using group_load_float = dpct::group::workgroup_load<NUM_PER_THREAD, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
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
        std::memset(unorm, 0, 1*sizeof(float));
        //DPCT_CHECK_ERROR(q_ct1.memset(unorm, 0, 1*sizeof(float)).wait());
        
        {
          dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
          q_ct1.submit(
            [&](sycl::handler &cgh) {
              using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             using group_load_float = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, float,  float *, sycl::nd_item<3>>;
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



//============================8 bit optimizer===============================


#define NUM8BIT 16
#define NUM_THREADS 256
#define NUM_PER_BLOCK 4096

template<typename T, int OPTIMIZER> void optimizerStatic8bit(T* p, T* g,
                unsigned char* state1, unsigned char* state2,
                float *unorm, float max_unorm, float param_norm,
                float beta1, float beta2,
                float eps, int step, float lr,
                float* quantiles1, float* quantiles2,
                float* max1, float* max2, float* new_max1, float* new_max2,
                float weight_decay,
                const float gnorm_scale, int n)
 try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  int num_blocks = n/4096;
  num_blocks = n % 4096 == 0 ? num_blocks : num_blocks + 1;
  sycl::context ctx = q_ct1.get_context();
  int size = NUM_BLOCK;
  
  sycl::buffer<T, 1> buff_g(g,sycl::range<1>(size));
  sycl::buffer<T, 1> buff_p(p,sycl::range<1>(size));
  sycl::buffer<unsigned char, 1> buff_state1(state1,sycl::range<1>(size));
  sycl::buffer<unsigned char, 1> buff_state2(state2,sycl::range<1>(size));

  if(max_unorm > 0.0f){ 
    q_ct1.memset(unorm, 0, 1*sizeof(float)).wait(); }

	switch(OPTIMIZER)
	{
		case ADAM:
			
      //DPCT_CHECK_ERROR(q_ct1.memset(new_max1, 0, 1*sizeof(float)).wait());
      //DPCT_CHECK_ERROR(q_ct1.memset(new_max2, 0, 1*sizeof(float)).wait());
			{
			  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
			      
                                              
             using group_load = dpct::group::workgroup_load<NUM8BIT, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);
             sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
             sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
             sycl::accessor dacc_p(buff_p, cgh, sycl::read_write);
             sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
             sycl::accessor dacc_state2(buff_state2, cgh, sycl::read_write);
             
            //__shared__ vars
            sycl::local_accessor<float, 1> smem_quantiles1_acc_ct1(sycl::range<1>(256), cgh);        
			      sycl::local_accessor<float, 1> smem_quantiles2_acc_ct1(sycl::range<1>(256), cgh);
              

			      cgh.parallel_for(
			        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)), 
			        [=](sycl::nd_item<3> item_ct1) {
			          kPreconditionOptimizerStatic8bit2State<T, OPTIMIZER>(p, g, state1, state2, unorm, beta1, beta2, eps, step, quantiles1, quantiles2, max1, max2, new_max1, new_max2, gnorm_scale, n, item_ct1, smem_quantiles1_acc_ct1.get_pointer(), smem_quantiles2_acc_ct1.get_pointer(), tacc, dacc_g, dacc_state1, dacc_state2);
			        });
			    });
			}
			
			{
			  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
			      
                    
            using group_load = dpct::group::workgroup_load<NUM8BIT, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);
             sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
             sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
             sycl::accessor dacc_p(buff_p, cgh, sycl::read_write);
             sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
             sycl::accessor dacc_state2(buff_state2, cgh, sycl::read_write);

            //__shared__ vars
            sycl::local_accessor<float, 1> smem_quantiles1_acc_ct1(sycl::range<1>(256), cgh);        
			      sycl::local_accessor<float, 1> smem_quantiles2_acc_ct1(sycl::range<1>(256), cgh);
            
			      cgh.parallel_for(
			        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
			        [=](sycl::nd_item<3> item_ct1) {
			          kOptimizerStatic8bit2State<T, OPTIMIZER>(p, g, state1, state2, unorm, max_unorm, param_norm, beta1, beta2, eps, step, lr, quantiles1, quantiles2, max1, max2, new_max1, new_max2, weight_decay, gnorm_scale, n, item_ct1, smem_quantiles1_acc_ct1.get_pointer(), smem_quantiles2_acc_ct1.get_pointer(), tacc, dacc_g, dacc_p, dacc_state1, dacc_state2);
			        });
			    });
			}
			
		break;
		case MOMENTUM:
    case RMSPROP:
    case ADAGRAD:
			
      //DPCT_CHECK_ERROR(q_ct1.memset(new_max1, 0, 1*sizeof(float)).wait());
			{
			  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
			      
             using group_load = dpct::group::workgroup_load<NUM8BIT, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);
             sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
             sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
             sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
             
             //__shared__ vars
            sycl::local_accessor<float, 1> smem_quantiles1_acc_ct1(sycl::range<1>(256), cgh);                   
       
            cgh.parallel_for(
			        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)), 
			        [=](sycl::nd_item<3> item_ct1) {
			          kPreconditionOptimizerStatic8bit1State<T, OPTIMIZER>(p, g, state1, unorm, beta1, beta2, eps, step, quantiles1, max1, new_max1, weight_decay, gnorm_scale, n, item_ct1, smem_quantiles1_acc_ct1.get_pointer(), tacc, dacc_g, dacc_state1);
			        });
			    });
			}
			
			
			{
			  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
             
             using group_load = dpct::group::workgroup_load<NUM8BIT, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);
             sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
             sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
             sycl::accessor dacc_p(buff_p, cgh, sycl::read_write);
             sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
            
            //__shared__ vars
            sycl::local_accessor<float, 1> smem_quantiles1_acc_ct1(sycl::range<1>(256), cgh);
			      cgh.parallel_for(
			        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
			        [=](sycl::nd_item<3> item_ct1) {
			          kOptimizerStatic8bit1State<T, OPTIMIZER>(p, g, state1, unorm, max_unorm, param_norm, beta1, beta2, eps, step, lr, quantiles1, max1, new_max1, weight_decay, gnorm_scale, n, item_ct1, smem_quantiles1_acc_ct1.get_pointer(), tacc, dacc_g, dacc_p, dacc_state1);
			        });
			    });
			}
		
			break;
    case LION:
      
      {
        dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
        q_ct1.submit(
          [&](sycl::handler &cgh) {
            
            using group_load = dpct::group::workgroup_load<NUM8BIT, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);
             sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
             sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
             sycl::accessor dacc_p(buff_p, cgh, sycl::read_write);
             sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);            
            
            //__shared__ vars
            sycl::local_accessor<float, 1> smem_quantiles1_acc_ct1(sycl::range<1>(256), cgh);
            
            cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
              [=](sycl::nd_item<3> item_ct1) {
                kOptimizerStatic8bit1State<T, OPTIMIZER>(p, g, state1, unorm, max_unorm, param_norm, beta1, beta2, eps, step, lr, quantiles1, max1, new_max1, weight_decay, gnorm_scale, n, item_ct1, smem_quantiles1_acc_ct1.get_pointer(), tacc, dacc_g, dacc_p, dacc_state1);
              });
          });
      }
     
      DPCT_CHECK_ERROR(q_ct1.memset(new_max1, 0, 1*sizeof(float)).wait());
      {
        dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
        q_ct1.submit(
          [&](sycl::handler &cgh) {
            
            using group_load = dpct::group::workgroup_load<NUM8BIT, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
             size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);
             sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
             sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
             sycl::accessor dacc_state1(buff_state1, cgh, sycl::read_write);
            //__shared__ vars
            sycl::local_accessor<float, 1> smem_quantiles1_acc_ct1(sycl::range<1>(256), cgh);
            
            cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)), 
              [=](sycl::nd_item<3> item_ct1) {
                kPreconditionOptimizerStatic8bit1State<T, OPTIMIZER>(p, g, state1, unorm, beta1, beta2, eps, step, quantiles1, max1, new_max1, weight_decay, gnorm_scale, n, item_ct1, smem_quantiles1_acc_ct1.get_pointer(), tacc, dacc_g, dacc_state1);
              });
          });
      }
     
      break;
		default:
			break;
	}
 
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}




//============================8 bit blockwise optimizer===============================

#define BLOCKSIZE_2STATE 2048
#define NUM_2STATE 8
#define BLOCKSIZE_1STATE 2048
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
			     
            using group_load = dpct::group::workgroup_load<NUM_2STATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
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
			      
            using group_load = dpct::group::workgroup_load<NUM_1STATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
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

//============================percentile clipping===============================

template<typename T> void percentileClipping(T * g, float *gnorm_vec, int step, const int n)
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    sycl::context ctx = q_ct1.get_context();
    
    int num_blocks = n/2048;
    num_blocks = n % 2048 == 0 ? num_blocks : num_blocks + 1;
    int size = NUM_BLOCK;
  
    sycl::buffer<T, 1> buff_g(g,sycl::range<1>(size));
    q_ct1.memset(&gnorm_vec[step % 100], 0, 1*sizeof(float)).wait();
  
  {
    dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
    q_ct1.submit(
      [&](sycl::handler &cgh) {
        
         using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, T,  T *, sycl::nd_item<3>>;
         size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);
         sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            
         sycl::accessor dacc_g(buff_g, cgh, sycl::read_write);
                  
      cgh.parallel_for(    
       sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
      [=](sycl::nd_item<3> item_ct1) {
        kPercentileClipping<T, 2048, 4>(g, gnorm_vec, step, n, item_ct1, tacc, dacc_g);
      });
    });
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
  
  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
    q_ct1.submit(
		[&](sycl::handler &cgh) {
            
          using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, int,  int *, sycl::nd_item<3>>;
          size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);
          sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);  
          sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
            
            //__shared__ vars
            sycl::local_accessor<float, 1> smem_rowStats_acc_ct1(sycl::range<1>(256), cgh);
            
			      cgh.parallel_for(
			        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, BLOCKSIZE_1STATE/NUM_1STATE), sycl::range<3>(1, 1, BLOCKSIZE_1STATE/NUM_1STATE)), 
			        [=](sycl::nd_item<3> item_ct1) {
  kdequant_mm_int32_fp16<4, 128, 512>(A, rowStats, colStats, out, newRowStats, newcolStats, bias, numRows, numCols, tileCols, n, item_ct1,smem_rowStats_acc_ct1.get_pointer(), tacc, dacc_A );
           });
  
  });
  
}

//==================================func===========================

template <typename T, int FUNC> void func(T *A, T *B, T value, long n)
{
  int threads = 512;
  int blocks = n/threads;
  blocks = n % threads == 0 ? blocks : blocks + 1;
  blocks = blocks > 65535 ? 65535 : blocks;
  
  dpct::get_in_order_queue().submit(
    [&](sycl::handler &cgh) {
    
  cgh.parallel_for(
    sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
    [=](sycl::nd_item<3> item_ct1) {
      kfunc<T, FUNC>(A, B, value, n, item_ct1);
    });
  });
  
}
//========================GEMM============================




//===========================Row col stats=================================

//===========================Row col stats=================================

#define STATS_THREADS 64
#define STATS_ITEMS 4
#define STATS_ROWS 16

void getColRowStats(sycl::half * A, float *rowStats, float *colStats, int *nnz_count_row, float nnz_threshold, int rows, int cols)
{
  
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
    
  
  int tile_cols = STATS_THREADS*STATS_ITEMS;
  int tiledCols = fill_up_to_nearest_multiple(cols, tile_cols);
  int tiledRows = fill_up_to_nearest_multiple(rows, STATS_ROWS);
	int row_tiles = (tiledRows/STATS_ROWS);
	int col_tiles = (tiledCols/tile_cols);
	row_tiles = row_tiles > 0 ? row_tiles : 1;
	col_tiles = col_tiles > 0 ? col_tiles : 1;
  int num_blocks = row_tiles * col_tiles;
  
  int size = NUM_BLOCK;
  
  sycl::buffer<sycl::half, 1> buff_A(A,sycl::range<1>(size));

  if(nnz_threshold == 0.0)
    {
    
			  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
            
            using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, sycl::half,  sycl::half *, sycl::nd_item<3>>;
            size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);
            sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
            
            //__shared__ vars
            sycl::local_accessor<float, 1> smem_row_absmax_values_acc_ct1(sycl::range<1>(256), cgh);
			      sycl::local_accessor<int, 1> smem_row_nnz_values_acc_ct1(sycl::range<1>(256), cgh);
                        
                        
       cgh.parallel_for(      
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
            [=](sycl::nd_item<3> item_ct1) {
              kgetColRowStats<sycl::half, STATS_THREADS, STATS_ITEMS, STATS_ROWS, STATS_THREADS*STATS_ITEMS, 0>(A, rowStats, colStats,
               nnz_count_row,    nnz_threshold, rows, cols, tiledRows, tiledCols,item_ct1, smem_row_absmax_values_acc_ct1.get_pointer(), smem_row_nnz_values_acc_ct1.get_pointer(), tacc, dacc_A);
            });
       });
    }
  else if(nnz_threshold != 0.0)
    {
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
		    [&](sycl::handler &cgh) {
            
            using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, sycl::half,  sycl::half *, sycl::nd_item<3>>;
            size_t temp_storage_size = group_load::get_local_memory_size(THREADS_ESTIMATE);
            sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
            sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
            
            //__shared__ vars
            sycl::local_accessor<float, 1> smem_row_absmax_values_acc_ct1(sycl::range<1>(256), cgh);
			      sycl::local_accessor<int, 1> smem_row_nnz_values_acc_ct1(sycl::range<1>(256), cgh);
               
                        
      cgh.parallel_for(      
          sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
          [=](sycl::nd_item<3> item_ct1) {
            kgetColRowStats<sycl::half, STATS_THREADS, STATS_ITEMS, STATS_ROWS, STATS_THREADS*STATS_ITEMS, 0>(A, rowStats, colStats,
             nnz_count_row, nnz_threshold, rows, cols, tiledRows, tiledCols,item_ct1, smem_row_absmax_values_acc_ct1.get_pointer(), smem_row_nnz_values_acc_ct1.get_pointer(), tacc, dacc_A);
      });
    });
    }

}

//=================================================double row col quant=====================================
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
            
            using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, sycl::half,  sycl::half *, sycl::nd_item<3>>;
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
            
            using group_load = dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, sycl::half,  sycl::half *, sycl::nd_item<3>>;
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

//======================================= transform row to format===============================================

template <int FORMAT, int TRANSPOSE> void transformRowToFormat(char * A, char *out, int rows, int cols)
{
  
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
	int num_blocks = 0;
  int size = NUM_BLOCK;
  
  sycl::buffer<char, 1> buff_A(A,sycl::range<1>(size));
  sycl::buffer<char, 1> buff_out(out,sycl::range<1>(size));
  
  
  int threads = 256;
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
  if(FORMAT == COL_TURING)
  {
    if(TRANSPOSE)
      outRows = fill_up_to_nearest_multiple(cols, 8);
    else
      outRows = fill_up_to_nearest_multiple(rows, 8);
  }
  else if(FORMAT == COL_AMPERE)
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
  int threads = 256;
  // we load 128 column values per warp
  int tiledCols = tiledCols = fill_up_to_nearest_multiple(cols, 32);
  int tiledRows = 0;

	int num_blocks = idx_size;

  if(FORMAT == COL_TURING)
  {
      tiledRows = fill_up_to_nearest_multiple(rows, 8);
  }
  else if(FORMAT == COL_AMPERE)
  {
      tiledRows = fill_up_to_nearest_multiple(rows, 32);
	}

  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
  
  dpct::get_in_order_queue().submit(
    [&](sycl::handler &cgh) {
    
    cgh.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)), 
      [=](sycl::nd_item<3> item_ct1) {
           kExtractOutliers<FORMAT>(A, idx, out, idx_size, rows, cols, tiledRows, tiledCols, item_ct1);
    });
   });
}
//==============================================================
//                   TEMPLATE DEFINITIONS
//==============================================================

void getColRowStats(sycl::half *A, float *rowStats, float *colStats,
                    int *nnz_count_row, float nnz_threshold, int rows,
                    int cols);


template void estimateQuantiles(sycl::half *A, float *code, float offset, int n);
template void estimateQuantiles(float *A, float *code, float offset, int n);

template void quantizeBlockwise<sycl::half, 1, General8bit>(float * code, sycl::half *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<sycl::half, 0, General8bit>(float * code, sycl::half *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<sycl::half, 0, FP4>(float * code, sycl::half *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<sycl::half, 0, NF4>(float * code, sycl::half *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<float, 1, General8bit>(float * code, float *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<float, 0, General8bit>(float * code, float *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<float, 0, FP4>(float * code, float *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<float, 0, NF4>(float * code, float *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<bf16, 1, General8bit>(float * code, bf16 *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<bf16, 0, General8bit>(float * code, bf16 *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<bf16, 0, FP4>(float * code, bf16 *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<bf16, 0, NF4>(float * code, bf16 *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);

template void dequantizeBlockwise<float, General8bit>(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n);
template void dequantizeBlockwise<float, FP4>(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n);
template void dequantizeBlockwise<float, NF4>(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n);
template void dequantizeBlockwise<sycl::half, General8bit>(float *code, unsigned char *A, float *absmax, sycl::half *out, int blocksize, const int n);
template void dequantizeBlockwise<sycl::half, FP4>(float *code, unsigned char *A, float *absmax, sycl::half *out, int blocksize, const int n);
template void dequantizeBlockwise<sycl::half, NF4>(float *code, unsigned char *A, float *absmax, sycl::half *out, int blocksize, const int n);
template void dequantizeBlockwise<bf16, General8bit>(float *code, unsigned char *A, float *absmax, bf16 *out, int blocksize, const int n);
template void dequantizeBlockwise<bf16, FP4>(float *code, unsigned char *A, float *absmax, bf16 *out, int blocksize, const int n);
template void dequantizeBlockwise<bf16, NF4>(float *code, unsigned char *A, float *absmax, bf16 *out, int blocksize, const int n);

#define MAKE_optimizer32bit(name, gtype) \
template void optimizer32bit<gtype, name>(gtype* g, gtype* p, \
                float* state1, float* state2, float* unorm, float max_unorm, float param_norm, \
                const float beta1, const float beta2, const float eps, const float weight_decay, \
                const int step, const float lr, const float gnorm_scale, const bool skip_zeros, const int n);

MAKE_optimizer32bit(ADAM, sycl::half)
MAKE_optimizer32bit(ADAM, float)
MAKE_optimizer32bit(ADAM, bf16)
MAKE_optimizer32bit(MOMENTUM, sycl::half)
MAKE_optimizer32bit(MOMENTUM, float)
MAKE_optimizer32bit(RMSPROP, sycl::half)
MAKE_optimizer32bit(RMSPROP, float)
MAKE_optimizer32bit(LION, sycl::half)
MAKE_optimizer32bit(LION, float)
MAKE_optimizer32bit(LION, bf16)
MAKE_optimizer32bit(ADAGRAD, sycl::half)
MAKE_optimizer32bit(ADAGRAD, float)

#define MAKE_optimizerStatic8bit(name, gtype) \
template void optimizerStatic8bit<gtype, name>(gtype* p, gtype* g, unsigned char* state1, unsigned char* state2, \
                float *unorm, float max_unorm, float param_norm, \
                float beta1, float beta2, \
                float eps, int step, float lr,  \
                float* quantiles1, float* quantiles2, \
                float* max1, float* max2, float* new_max1, float* new_max2, \
                float weight_decay, \
                const float gnorm_scale, int n); \

MAKE_optimizerStatic8bit(ADAM, sycl::half)
MAKE_optimizerStatic8bit(ADAM, float)
MAKE_optimizerStatic8bit(MOMENTUM, sycl::half)
MAKE_optimizerStatic8bit(MOMENTUM, float)
MAKE_optimizerStatic8bit(RMSPROP, sycl::half)
MAKE_optimizerStatic8bit(RMSPROP, float)
MAKE_optimizerStatic8bit(LION, sycl::half)
MAKE_optimizerStatic8bit(LION, float)

#define MAKE_optimizerStatic8bitBlockwise(gtype, optim_name) \
template void optimizerStatic8bitBlockwise<gtype, optim_name>(gtype* p, gtype* g, \
                unsigned char* state1, unsigned char* state2, float beta1, float beta2, float eps, int step, float lr,  \
                float* quantiles1, float* quantiles2, float* absmax1, float* absmax2, float weight_decay, const float gnorm_scale, bool skip_zeros, int n); \

MAKE_optimizerStatic8bitBlockwise(sycl::half, ADAM);
MAKE_optimizerStatic8bitBlockwise(float, ADAM);
MAKE_optimizerStatic8bitBlockwise(sycl::half, MOMENTUM);
MAKE_optimizerStatic8bitBlockwise(float, MOMENTUM);
MAKE_optimizerStatic8bitBlockwise(sycl::half, RMSPROP);
MAKE_optimizerStatic8bitBlockwise(float, RMSPROP);
MAKE_optimizerStatic8bitBlockwise(sycl::half, LION);
MAKE_optimizerStatic8bitBlockwise(float, LION);
MAKE_optimizerStatic8bitBlockwise(bf16, LION);
MAKE_optimizerStatic8bitBlockwise(sycl::half, ADAGRAD);
MAKE_optimizerStatic8bitBlockwise(float, ADAGRAD);

template void percentileClipping(float * g, float *gnorm_vec, int step, const int n);
template void percentileClipping(sycl::half * g, float *gnorm_vec, int step, const int n);

MAKE_optimizerStatic8bitBlockwise(bf16, ADAM);

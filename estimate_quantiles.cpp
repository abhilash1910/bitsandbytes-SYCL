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
}
}


//#define dpct::group::workgroup_load<NUM_ESTIMATE, dpct::group::load_algorithm::BLOCK_LOAD_DIRECT, int,  int *, sycl::nd_item<3>> wg_load;


template<typename T>
SYCL_EXTERNAL 
void kEstimateQuantiles(const T *A, float *code, const float offset, const T max_val, const int n,
                        const sycl::nd_item<3> &item_ct1, sycl_la tacc, const sycl::accessor<T, 1> &dacc_A, const sycl_dacc_float &dacc_code)
{
  const int n_full = (BLOCK_ESTIMATE*(n/BLOCK_ESTIMATE)) + (n % BLOCK_ESTIMATE == 0 ? 0 : BLOCK_ESTIMATE);
  int valid_items = (item_ct1.get_group(2)+1 == item_ct1.get_group_range(2)) ? n - (item_ct1.get_group(2)*BLOCK_ESTIMATE) : BLOCK_ESTIMATE;
  const int base_idx = (item_ct1.get_group(2) * BLOCK_ESTIMATE);
  const float reciprocal_num_blocks = 1.0f/(n < 4096 ? 1.0f : (n/BLOCK_ESTIMATE));
  
  using group_load = dpct_::group::workgroup_load<NUM_ESTIMATE, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, int,  int *, sycl::nd_item<3>>;
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


template <typename T> void estimateQuantiles(T *A, float *code, float offset, int n)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  int num_blocks = n/4096;
  num_blocks = n % 4096 == 0 ? num_blocks : num_blocks + 1;
  //DPCT_CHECK_ERROR(q_ct1.memset(code, 0, 256*sizeof(float)).wait());
  sycl::context ctx = q_ct1.get_context();
  int size = 512;
  
  
  sycl::buffer<T, 1> buff_A(A,sycl::range<1>(size));
  sycl::buffer<float, 1> buff_code(code,sycl::range<1>(size));
  {
    //dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
    q_ct1.submit(
      [&](sycl::handler &cgh) {
      
        using group_load = dpct_::group::workgroup_load<NUM_ESTIMATE, dpct_::group::load_algorithm::BLOCK_LOAD_DIRECT, int,  int *, sycl::nd_item<3>>;
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
int main(){

int data[512];
for(int i=0;i<512;i++){data[i]=i;}
float offset =1.0f;
float code[512];
for(int i=0;i<512;i++){code[i]=1.0f;}

estimateQuantiles<int>(data, code, offset, 512);

return 0;

}
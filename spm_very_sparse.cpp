#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
#include <dpct/dpl_extras/dpcpp_extensions.h>
#include <dpct/lib_common_utils.hpp>
#include <dpct/sparse_utils.hpp>

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
typedef sycl::accessor<unsigned char, 1> sycl_dacc_uc;




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


template <typename T, int BITS> void spmm_coo_very_sparse_naive(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, sycl::half *values, T *B, sycl::half *out, float *dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB)
{

  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
	int size = NUM_BLOCK;
  
  sycl::buffer<int, 1> buff_max_count(max_count,sycl::range<1>(size));
  sycl::buffer<int, 1> buff_max_idx(max_idx,sycl::range<1>(size));
  sycl::buffer<int, 1> buff_offset_rowidx(offset_rowidx,sycl::range<1>(size));
  sycl::buffer<int, 1> buff_rowidx(rowidx,sycl::range<1>(size));
  sycl::buffer<int, 1> buff_colidx(colidx,sycl::range<1>(size));
  sycl::buffer<sycl::half, 1> buff_values(values,sycl::range<1>(size));
  sycl::buffer<sycl::half, 1> buff_out(out,sycl::range<1>(size));
  sycl::buffer<T, 1> buff_B(B, sycl::range<1>(size));
  sycl::buffer<float, 1> buff_dequant_stats(dequant_stats,sycl::range<1>(size));
  

  {
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp16});
    q_ct1.submit(
      [&](sycl::handler &cgh) {
        
         sycl::accessor dacc_max_count(buff_max_count, cgh, sycl::read_write);
         sycl::accessor dacc_max_idx(buff_max_idx, cgh, sycl::read_write);
         sycl::accessor dacc_offset_rowidx(buff_offset_rowidx, cgh, sycl::read_write);
         sycl::accessor dacc_colidx(buff_colidx, cgh, sycl::read_write);
         sycl::accessor dacc_rowidx(buff_rowidx, cgh, sycl::read_write);
         sycl::accessor dacc_values(buff_values, cgh, sycl::read_write);
         sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
         sycl::accessor dacc_dequant_stats(buff_dequant_stats, cgh, sycl::read_write);
         sycl::accessor dacc_B(buff_B, cgh, sycl::read_write);
         
        
        //smem
        sycl::local_accessor<sycl::half, 1> smem_dequant_stats_acc_ct1(sycl::range<1>(2048/*SMEM_SIZE*/), cgh);
   
        cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, nnz_rows) * sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)), 
          [=](sycl::nd_item<3> item_ct1) {
            kspmm_coo_very_sparse_naive<T, 8, BITS>(max_count, max_idx, offset_rowidx, rowidx, colidx, values, B, out, dequant_stats, nnz, rowsA, rowsB, colsB, item_ct1, smem_dequant_stats_acc_ct1.get_pointer(), dacc_max_count, dacc_max_idx, dacc_offset_rowidx, dacc_rowidx, dacc_colidx, dacc_values, dacc_B, dacc_out, dacc_dequant_stats);
          });
      });
  }
  
}




int main(){

int max_count[512];
int max_idx[512];
int offset_rowidx[512];
int rowidx[512];
int colidx[512];

sycl::half values[512];
float B[512],dequant_stats[512];
sycl::half out[512];

int nnz_rows=1;
int nnz = nnz_rows;
int rowsA=1;
int rowsB =rowsA;
int colsB = 1;

for(int i=0;i<512;i++){ max_count[i]=1;max_idx[i]=0,offset_rowidx[i]=i-1;rowidx[i]=i;colidx[i]=0;values[i]=sycl::half(0.5f);B[i]=1.0f;dequant_stats[i]=1.0f;}


spmm_coo_very_sparse_naive<float, 8>(max_count, max_idx, offset_rowidx, rowidx, colidx, values, B, out, dequant_stats, nnz_rows, nnz, rowsA, rowsB, colsB);

return 0;

}
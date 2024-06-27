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
typedef sycl::accessor<unsigned char, 1> sycl_dacc_uc;


//===================================gemm_device ================


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
        sycl::ext::oneapi::experimental::matrix::joint_matrix_load(sg_size, b_frag,  d_B, 16);
    
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
    dacc_out[col_offset + warp_lane] = dacc_A[warp_lane];
#endif
}


//===========================gemm_host============================================


template <typename T> void gemm_host(int m, int n, int k, T * A,  T* B,  T * out,  int lda, int ldb, int ldc, int bits)
{

	int num_blocks = (m+31)/32;
 
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
    
  int size = NUM_BLOCK;
  
  sycl::buffer<T, 1> buff_A (A, sycl::range<1>(size));
  sycl::buffer<T, 1> buff_B (B, sycl::range<1>(size));
  sycl::buffer<T, 1> buff_out (out, sycl::range<1>(size));
  
	//cout << num_blocks << endl;
	//cout << lda << endl;
	//cout << ldb << endl;
	//cout << ldc << endl;

	//cout << m << endl;
	//cout << n << endl;
	//cout << k << endl;
  //if(bits == 32)
    //gemm_device<T, 32, 128><<< num_blocks, 128, 0, 0 >>>(m,  n,  k, A,  B,  out, lda, ldb, ldc);
    //gemm_device<T, 32, 32><<< num_blocks, 32, 0, 0 >>>(m,  n,  k, A,  B,  out, lda, ldb, ldc);
  if(bits == 16)
    //gemm_device<T, 16, 256><<< num_blocks, 256, 0, 0 >>>(m,  n,  k, A,  B,  out, lda, ldb, ldc);
    {
      dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp16});
      dpct::get_in_order_queue().submit(
        [&](sycl::handler &cgh) {
          
          sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
          sycl::accessor dacc_B(buff_B, cgh, sycl::read_write);
          sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
          
          //__shared__ vars
          sycl::local_accessor<T, 1> smem_A_acc_ct1(sycl::range<1>(224/*8*16 + (2*16*(batch_size_warps-1))*/), cgh);
          sycl::local_accessor<T, 1> smem_B_acc_ct1(sycl::range<1>(4192/*2*batch_size_warps*16*32 + (2*16*(batch_size_warps-1))*/), cgh);

          cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 160), sycl::range<3>(1, 1, 160)), 
            [=](sycl::nd_item<3> item_ct1) {
              gemm_device<T, 16, 160>(m, n, k, A, B, out, lda, ldb, ldc, item_ct1, smem_A_acc_ct1.get_pointer(), smem_B_acc_ct1.get_pointer(),
              dacc_A, dacc_B, dacc_out);
            });
        });
    }
    //gemm_device<T, 16, 128><<< num_blocks, 128, 0, 0 >>>(m,  n,  k, A,  B,  out, lda, ldb, ldc);
    //gemm_device<T, 16, 96><<< num_blocks, 96, 0, 0 >>>(m,  n,  k, A,  B,  out, lda, ldb, ldc);
    //gemm_device<T, 16, 32><<< num_blocks, 32, 0, 0 >>>(m,  n,  k, A,  B,  out, lda, ldb, ldc);
    //gemm_device<T, 16, 64><<< num_blocks, 64, 0, 0 >>>(m,  n,  k, A,  B,  out, lda, ldb, ldc);
   
  
}



int main(){

sycl::half A[512];
sycl::half B[512];
sycl::half out[512];
float row_scale[512];
int8_t C[512];
int m = 2;
int n = m;
int k = m;
int lda = 2;
int ldb = lda;
int ldc = lda;
int bits = 2;
for(int i=0;i<512;i++){ A[i]=sycl::half(1.0f),B[i]=sycl::half(1.0f);out[i]=sycl::half(0.0f);}

gemm_host<sycl::half>(m, n, k, A, B, out, lda, ldb, ldc, bits);



return 0;

}
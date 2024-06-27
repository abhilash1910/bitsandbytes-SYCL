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

//===================================gemm_4bit ================


#define WARPS 3

const float nf4_data[16] = {-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0};



template <typename T, int THREADS> SYCL_EXTERNAL void kgemm_4bit_inference(int M, int N, int K, T * __restrict__ const A, unsigned char *B,  float *absmax, T * out,  int lda, int ldb, int ldc, int blocksize, const sycl::nd_item<3> &item_ct1, T *smem_A, unsigned char *smem_B, T *smem_C,
const sycl::accessor<T, 1> &dacc_A, const sycl_dacc_uc &dacc_B, const sycl::accessor<T, 1> &dacc_out)
{

#if DPCT_COMPATIBILITY_TEMP >= 750
	
  int col_offset = item_ct1.get_group(2) *32;
  const int warp_id = item_ct1.get_local_id(2) / 32;
  const int warp_idx = item_ct1.get_local_id(2) % 32;
  const int half_warp_id = item_ct1.get_local_id(2) / 16;
  const int half_warp_lane = item_ct1.get_local_id(2) % 16;
  const int batch_size_warps = (WARPS-1)*2;

  T quant_map[16];

  #pragma unroll 16
  for(int i = 0; i < 16; i++)
    quant_map[i] = nf4_data[i];
  //__shared__ T quant_map[16*160];

  T local_A[2];
  T local_B[64];
  unsigned char local_B_4bit[32];


  const int a_tile_offset = 16;
  const int b_tile_offset = (16*32 + 16);

  auto d_A = dacc_A.template get_multi_ptr<sycl::access::decorated::yes>();
  auto d_B = dacc_B.get_multi_ptr<sycl::access::decorated::yes>();
  auto sg_size = item_ct1.get_sub_group();
  
   
   sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, T, sycl::ext::oneapi::experimental::matrix::use::a, 8, 16, sycl::ext::oneapi::experimental::matrix::layout::row_major> a_frag{};
   sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, unsigned char, sycl::ext::oneapi::experimental::matrix::use::b, 16, 32, sycl::ext::oneapi::experimental::matrix::layout::col_major> b_frag{};
   sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, T, sycl::ext::oneapi::experimental::matrix::use::accumulator, 8, 32> c_frag{};
   sycl::ext::oneapi::experimental::matrix::joint_matrix_fill(sg_size, c_frag, 0.0f); 
   
  

  //wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> a_frag;
  //wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> b_frag;
  //wmma::fragment<wmma::accumulator, 8, 32, 16, half> c_frag; 
  //wmma::fill_fragment(c_frag, 0.0f);
   
  for(int i = item_ct1.get_local_id(2); i < (8*32); i+=item_ct1.get_local_range(2))
    smem_C[i] = 0.0f;

  item_ct1.barrier(sycl::access::fence_space::local_space);

  int ticktock = 0;
  int idx = 0 + item_ct1.get_local_id(2);
  int loaded_values = 0;
  // prefetch
  if(idx < K && warp_id < (WARPS-1))
  {
    if(loaded_values == 0)
    {
      local_A[0] = dacc_A[idx];
      local_A[1] = dacc_A[idx+item_ct1.get_local_range(2)-32];

      #pragma unroll 32
      for(int col = 0; col < 32; col++)
        local_B_4bit[col] = dacc_B[(col_offset+col)*ldb+idx];

      loaded_values = 1;
    }
    else
    {
      local_A[0] = local_A[1];
      loaded_values--;

      #pragma unroll 64
      for(int col = 0; col < 64; col+=2)
      {
        //local_B[col] = dhDequantizeNF4(local_B_4bit[col/2] >> 4)*T(1.0f);
        //local_B[col+1] = dhDequantizeNF4(local_B_4bit[col/2] & 0x0F)*T(1.0f);
        //local_B[col] = d2DequantizeFP4(local_B_4bit[col/2] >> 4)*(float)(17.0);
        //local_B[col+1] = d2DequantizeFP4(local_B_4bit[col/2] & 0x0F)*(float)(17.0);
        //local_B[col] = 127*(local_B_4bit[col/2] >> 4)*(float)(17.0);
        //local_B[col+1] = 127*(local_B_4bit[col/2] & 0x0F)*(float)(17.0);

        //local_B[col] = quant_map[(local_B_4bit[col/2] >> 4)]*T(17.0);
        //local_B[col+1] = quant_map[(local_B_4bit[col/2] & 0x0F)]*T(17.0);
        local_B[col] = quant_map[160*(local_B_4bit[col/2] >> 4)+warp_idx]*T(17.0);
        local_B[col+1] = quant_map[160*(local_B_4bit[col/2] & 0x0F)+warp_idx]*T(17.0);
      }
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
    //if(threadIdx.x == 0)
      //printf("aa %i %i\n", idx, loaded_values);

  //for(int base_idx = blockDim.x-32; base_idx < K; base_idx+=blockDim.x-32)
  for(int base_idx = item_ct1.get_local_range(2)-32; base_idx < K; base_idx+=item_ct1.get_local_range(2)-32)
  {
    idx = base_idx + item_ct1.get_local_id(2);
    //if(threadIdx.x == 0)
      //printf("%i %i\n", idx, loaded_values);

    //__syncthreads();
    if(idx < K && warp_id < (WARPS-1))
    {
      if(loaded_values == 0)
      {
        local_A[0] = dacc_A[idx];
        local_A[1] = dacc_A[idx+item_ct1.get_local_range(2)-32];

        #pragma unroll 32
        for(int col = 0; col < 32; col++)
        {
          local_B_4bit[col] = dacc_B[(col_offset+col)*ldb+idx];
          local_B_4bit[col+16] = dacc_B[(col_offset+col)*ldb+idx];
        }

        loaded_values = 1;
      }
      else
      {
        local_A[0] = local_A[1];
        loaded_values--;

        int absidx = (idx + col_offset)/blocksize;
        
        sycl::half local_absmax =  absmax[absidx];

        #pragma unroll 64
        for(int col = 0; col < 64; col+=2)
        {
          //local_B[col] = dhDequantizeNF4(local_B_4bit[col/2] >> 4)*T(absidx);
          //local_B[col+1] = dhDequantizeNF4(local_B_4bit[col/2] & 0x0F)*T(absidx);
          //local_B[col] = T(127)*T(local_B_4bit[col/2] >> 4)*T(absidx);
          //local_B[col+1] = T(127)*T(local_B_4bit[col/2] & 0x0F)*T(absidx);

          //local_B[col] = quant_map[160*(local_B_4bit[col/2] >> 4)+warp_idx]*T(local_absmax);
          //local_B[col+1] = quant_map[160*(local_B_4bit[col/2] & 0x0F)+warp_idx]*T(local_absmax);
          local_B[col] = quant_map[(local_B_4bit[col/2] >> 4)]*T(absidx);
          local_B[col+1] = quant_map[(local_B_4bit[col/2] & 0x0F)]*T(absidx);
        }
        //printnonzero<T>(local_B, 128, "");
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
        
        dacc_A[(ticktock*batch_size_warps + k)*a_tile_offset] = smem_A[(ticktock*batch_size_warps + k)*a_tile_offset];
        dacc_B[(ticktock*batch_size_warps + k)*b_tile_offset] = smem_B[(ticktock*batch_size_warps + k)*b_tile_offset];
        d_A = dacc_A.template get_multi_ptr<sycl::access::decorated::yes>();
        d_B = dacc_B.get_multi_ptr<sycl::access::decorated::yes>();
        
        //wmma::load_matrix_sync(a_frag, &(smem_A[(ticktock*batch_size_warps + k)*a_tile_offset]), 16); //  111 mu
        sycl::ext::oneapi::experimental::matrix::joint_matrix_load(sg_size, a_frag, d_A, 16);
    
        
        //wmma::load_matrix_sync(b_frag, &(smem_B[(ticktock*batch_size_warps + k)*b_tile_offset]), 16); // 35 mu
        sycl::ext::oneapi::experimental::matrix::joint_matrix_load(sg_size, b_frag, d_B, 16);
    
        
        //wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        sycl::ext::oneapi::experimental::matrix::joint_matrix_mad(sg_size, c_frag, a_frag, b_frag, c_frag);
        
      }
  }

  item_ct1.barrier(sycl::access::fence_space::local_space);
  //if(threadIdx.x == 0)
  //{
  //  printnonzero<T>(smem_A, 8*16 + (2*16*(batch_size_warps-1)), "A: ");
  //  printnonzero<T>(smem_B, 2*batch_size_warps*16*32 + (2*16*(batch_size_warps-1)), "B: ");
  //}
  if(warp_id != (WARPS-1)){ return; }
  // only warp_id == (WARPS-1) from here
  int warp_lane = item_ct1.get_local_id(2) % 32;

  ticktock = ticktock == 0 ? 1 : 0;
  for(int k = 0; k < batch_size_warps; k++)
  {
    //if(warp_lane == 0)
      //printf("%i %i %i %i\n", (ticktock*batch_size_warps + k)*a_tile_offset, k, ticktock, threadIdx.x);
    
    dacc_A[(ticktock*batch_size_warps + k)*a_tile_offset] = smem_A[(ticktock*batch_size_warps + k)*a_tile_offset];
    dacc_B[(ticktock*batch_size_warps + k)*b_tile_offset] = smem_B[(ticktock*batch_size_warps + k)*b_tile_offset];
    d_A = dacc_A.template get_multi_ptr<sycl::access::decorated::yes>();
    d_B = dacc_B.get_multi_ptr<sycl::access::decorated::yes>();
    
    //wmma::load_matrix_sync(a_frag, &(smem_A[(ticktock*batch_size_warps + k)*a_tile_offset]), 16); //  111 mu
    sycl::ext::oneapi::experimental::matrix::joint_matrix_load(sg_size, a_frag, d_A, 16);

    //wmma::load_matrix_sync(b_frag, &(smem_B[(ticktock*batch_size_warps + k)*b_tile_offset]), 16); // 35 mu
    sycl::ext::oneapi::experimental::matrix::joint_matrix_load(sg_size, b_frag, d_B, 16);

    
    //wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    sycl::ext::oneapi::experimental::matrix::joint_matrix_mad(sg_size, c_frag, a_frag, b_frag, c_frag);
    
  }

  // 129 mu
  if(warp_id == (WARPS-1))
    
    //wmma::store_matrix_sync(&(smem_C[0]), c_frag, 32, wmma::mem_row_major);
    sycl::ext::oneapi::experimental::matrix::joint_matrix_store(sg_size, c_frag, d_A, 32, sycl::ext::oneapi::experimental::matrix::layout::row_major);

  //printnonzero<T>(smem_C, 32, "");  
  
  if(col_offset + warp_lane < M)
    // use smem_A itself
    dacc_out[col_offset + warp_lane] = dacc_A[warp_lane];
#endif
}




//============================gemm 4bit inference ================================

template <typename T> void gemm_4bit_inference(int m, int n, int k, T * A,  unsigned char* B,  float *absmax, T * out,  int lda, int ldb, int ldc, int blocksize)
{

	int num_blocks = (m+31)/32;

	//cout << num_blocks << endl;
	//cout << lda << endl;
	//cout << ldb << endl;
	//cout << ldc << endl;

	//cout << m << endl;
	//cout << n << endl;
	//cout << k << endl;
 
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
    
  int size = NUM_BLOCK;
  
  sycl::buffer<T, 1> buff_A (A, sycl::range<1>(size));
  sycl::buffer<unsigned char, 1> buff_B (B, sycl::range<1>(size));
  sycl::buffer<T, 1> buff_out (out, sycl::range<1>(size));
 
  {
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp16});
    dpct::get_in_order_queue().submit(
      [&](sycl::handler &cgh) {
        
        sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
        sycl::accessor dacc_B(buff_B, cgh, sycl::read_write);
        sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);  
        
        //__shared__ vars
        sycl::local_accessor<T, 1> smem_A_acc_ct1(sycl::range<1>(176/*8*16 + (16*(batch_size_warps-1))*/), cgh);
        sycl::local_accessor<unsigned char, 1> smem_B_acc_ct1(sycl::range<1>(4192/*2*batch_size_warps*16*32 + (2*16*(batch_size_warps-1))*/), cgh);
        sycl::local_accessor<T, 1> smem_C_acc_ct1(sycl::range<1>(8*32), cgh);

        cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 96), sycl::range<3>(1, 1, 96)), 
          [=](sycl::nd_item<3> item_ct1) {
            kgemm_4bit_inference<T, 96>(m, n, k, A, B, absmax, out, lda, ldb, ldc, blocksize, item_ct1, smem_A_acc_ct1.get_pointer(), smem_B_acc_ct1.get_pointer(), smem_C_acc_ct1.get_pointer(), dacc_A, dacc_B, dacc_out);
          });
      });
  }
  
}



int main(){

int A[512];
unsigned char B[512];
int out[512];
float row_scale[512];
int8_t C[512];
int m = 2;
int n = m;
int k = m;
int lda = 2;
int ldb = lda;
int ldc = lda;
int bits = 2;
int blocksize =512;
float absmax[512];
int threads =2;
for(int i=0;i<512;i++){ A[i]=1,B[i]=1;out[i]=1;absmax[i]=0.5f;}

gemm_4bit_inference<int>(m, n, k, A, B, absmax, out, lda, ldb, ldc, blocksize);



return 0;

}
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

//===================================gemm_4bit naive================



#define num_values_4bit 32
/*
DPCT1110:49: The total declared local variable size in device function kgemm_4bit_inference_naive exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
template <typename T, int THREADS, int BITS> SYCL_EXTERNAL void kgemm_4bit_inference_naive(int M, int N, int K, T * __restrict__ const A, unsigned char *B,  float *absmax, const float *datatype, T * out,  int lda, int ldb, int ldc, int blocksize, const sycl::nd_item<3> &item_ct1, T *quant_map, const sycl::accessor<T, 1> &dacc_A, const sycl_dacc_uc &dacc_B, const sycl::accessor<T, 1> &dacc_out, const sycl_dacc_float &dacc_absmax, const sycl_dacc_float &dacc_datatype)
{

  // per threadblock:
  // load step-by-step in chunks of [32,warps]: 1x32 * [32,warps] -> [1,warps]
  // 4 warps -> 4 loads per iter
  // 1x32 * 32x4 -> 1x4 outputs per thread block
   // datatype absmax 
  const int warp_idx = item_ct1.get_local_id(2) / 32;
  const int warp_lane = item_ct1.get_local_id(2) % 32;
  const int row_B = (THREADS/32)*item_ct1.get_group(2) + warp_idx;
  const int num_values_8bit = num_values_4bit/2;
  float local_C = 0.0f;

  unsigned char local_B_4bit[num_values_8bit];
  T local_B[num_values_4bit/4];
  T local_A[num_values_4bit/4];
  
	T local_absmax = T(0.0f);

  for(int i = item_ct1.get_local_id(2); i < 16; i++)
    quant_map[i] = T(dacc_datatype[i]);
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // A: [1, K]
  // B: [N, K]
  for(int inner_idx = warp_lane*num_values_4bit; inner_idx < K; inner_idx += 32*num_values_4bit)
  {
    int inner_idx_halved = inner_idx/2;
    int offset_B = ldb*row_B;
    int absidx = ((2*offset_B)+inner_idx)/blocksize;
	  
	  local_absmax =  dacc_absmax[absidx];

    if(row_B < M)
    {
      if((inner_idx_halved + num_values_8bit) < (K/2))
      {
        // this is the most important for performance considerations
        reinterpret_cast<sycl::int4(&)[num_values_8bit]>(local_B_4bit)[0] = reinterpret_cast<sycl::int4*>(B)[(offset_B+(inner_idx_halved))/(num_values_8bit)];
      }
      else
      {
        #pragma unroll
        for(int j = 0; j < (num_values_8bit); j++)
          if((inner_idx_halved) + j < (K/2))
            local_B_4bit[j] = dacc_B[offset_B+inner_idx_halved + j];
          else
            local_B_4bit[j] = 0b01110111;
      }
    }
    else
    {
      #pragma unroll
      for(int j = 0; j < (num_values_8bit); j++)
          local_B_4bit[j] = 0b01110111;
    }

    for(int i = 0; i < 4; i++)
    {
      #pragma unroll
      for(int k = 0; k < num_values_8bit/4; k++)
      {
        #if DPCT_COMPATIBILITY_TEMP >= 800
          local_B[k*2] = quant_map[local_B_4bit[(i*num_values_8bit/4) + k] >> 4]*local_absmax;
          local_B[k*2 + 1] = quant_map[local_B_4bit[(i*num_values_8bit/4) + k] & 0x0F]*local_absmax;
        #else
          // bf16 multipliation not supported
          local_B[k*2] = T((float)quant_map[local_B_4bit[(i*num_values_8bit/4) + k] >> 4]*(float)local_absmax);
          local_B[k*2 + 1] = T((float)quant_map[local_B_4bit[(i*num_values_8bit/4) + k] & 0x0F]*(float)local_absmax);
        #endif
      }

      if(inner_idx+(num_values_4bit/4) + (i*num_values_4bit/4) < K)
      {
        // this is also relatively important for performance
        if(BITS==16)
        {
          reinterpret_cast<sycl::int4(&)[num_values_4bit]>(local_A)[0] = reinterpret_cast<sycl::int4*>(A)[inner_idx/(num_values_4bit/4) + i];
        }
        else
        {
          reinterpret_cast<sycl::int4(&)[num_values_4bit]>(local_A)[0] = reinterpret_cast<sycl::int4*>(A)[inner_idx/(num_values_4bit/8) + (2*i) + 0];
          reinterpret_cast<sycl::int4(&)[num_values_4bit]>(local_A)[1] = reinterpret_cast<sycl::int4*>(A)[inner_idx/(num_values_4bit/8) + (2*i) + 1];
        }

      }
      else
        #pragma unroll
        for(int k = 0; k < num_values_4bit/4; k++)
          if(inner_idx + (i*num_values_4bit/4) + k < K)
            local_A[k] = dacc_A[inner_idx + k + (i*num_values_4bit/4)];
          else
            local_A[k] = T(0.0f);


      // accumulate in float; small performance hit for Ampere, but lower error for outputs
      #pragma unroll
      for(int k = 0; k < num_values_4bit/4; k++)
      {
        #if DPCT_COMPATIBILITY_TEMP >= 800
          local_C += (float)(local_A[k]*local_B[k]);
        #else
          // bf16 multipliation not supported
          local_C += ((float)local_A[k]*(float)local_B[k]);
        #endif
      }
    }
  }

  local_C = sycl::reduce_over_group(item_ct1.get_sub_group(), local_C, sycl::plus<>());

  if(row_B < M && warp_lane == 0)
    dacc_out[row_B] = T(local_C);

}

//============================gemm 4bit inference naive================================

template <typename T, int BITS> void gemm_4bit_inference_naive(int m, int n, int k, T * A,  unsigned char* B,  float *absmax, float *datatype, T * out,  int lda, int ldb, int ldc, int blocksize)
{

	int num_blocks = (m+3)/4;
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
    
  int size = NUM_BLOCK;
  
  sycl::buffer<T, 1> buff_A (A, sycl::range<1>(size));
  sycl::buffer<unsigned char, 1> buff_B (B, sycl::range<1>(size));
  sycl::buffer<T, 1> buff_out (out, sycl::range<1>(size));
  sycl::buffer<float, 1> buff_absmax(absmax, sycl::range<1>(size));
  sycl::buffer<float, 1> buff_datatype(datatype, sycl::range<1>(size));
  
  {
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp16});
    dpct::get_in_order_queue().submit(
      [&](sycl::handler &cgh) {
        
        sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
        sycl::accessor dacc_B(buff_B, cgh, sycl::read_write);
        sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);  
        sycl::accessor dacc_absmax(buff_absmax, cgh, sycl::read_write);
        sycl::accessor dacc_datatype(buff_datatype, cgh, sycl::read_write);
        sycl::local_accessor<T, 1> quant_map_acc_ct1(sycl::range<1>(16), cgh);

        cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)), 
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            kgemm_4bit_inference_naive<T, 128, BITS>(m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize, item_ct1, quant_map_acc_ct1.get_pointer(), dacc_A, dacc_B, dacc_out, dacc_absmax, dacc_datatype);
          });
      });
  }
 
}


int main(){

int A[512];
unsigned char B[512];
int out[512];
float datatype[512];
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
for(int i=0;i<512;i++){ A[i]=1,B[i]=1;out[i]=1;absmax[i]=0.5f;datatype[i]=0.5f;}

gemm_4bit_inference_naive<int, 4>(m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize);



return 0;

}
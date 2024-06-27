#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <limits>
#include <cassert>
#include <dpct/lib_common_utils.hpp>

#include "oneapi/dnnl/dnnl.hpp"
#define NUM_BLOCK 512

typedef sycl::ext::oneapi::bfloat16 bf16;
typedef sycl::local_accessor<uint8_t ,1> sycl_la;

typedef sycl::accessor<int, 1> sycl_dacc;
typedef sycl::accessor<float, 1> sycl_dacc_float;
typedef sycl::accessor<unsigned char, 1> sycl_dacc_uc;
typedef sycl::accessor<char, 1> sycl_dacc_char;


SYCL_EXTERNAL void kHistogramScatterAdd2D(float* histogram, int *index1, int *index2, float *src, const int maxidx1, const int n,
                            const sycl::nd_item<3> &item_ct1, const sycl_dacc_float &dacc_histogram, const sycl_dacc &dacc_index1, 
                            const sycl_dacc &dacc_index2, const sycl_dacc_float &dacc_src)
{
  const int tid = item_ct1.get_local_id(2) + (item_ct1.get_local_range(2)*item_ct1.get_group(2));
  const int numThreads = item_ct1.get_local_range(2)*item_ct1.get_group_range(2);

  for(int i = tid; i < n; i+=numThreads)
  {
      int idx = (dacc_index1[i]*maxidx1) + dacc_index2[i];
      dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&dacc_histogram[idx], dacc_src[i]);
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

int main(){

float histogram[512];
float src[512];
int index1[512];
int index2[512];

int maxidx1 =0;
int n=512;
for(int i=0;i<512;i++){ histogram[i]=1.0f;src[i]=1.0f;index1[i]=1;index2[i]=1;}

histogramScatterAdd2D(histogram, index1, index2, src, maxidx1, n);

return 0;


}

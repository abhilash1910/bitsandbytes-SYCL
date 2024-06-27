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

//===================================================
#include <oneapi/dnnl/dnnl.hpp> 
#include <oneapi/dnnl/dnnl_sycl.hpp>

int roundoff(int v, int d) {
    return (v + d - 1) / d * d;
}


template<int ORDER> int get_leading_dim(int dim1, int dim2)
{
	switch(ORDER)
	{
		case 0://ROW
      return dim2;
			break;
    case 1://COL
      return dim1;
      break;
    case 2://COL32
      // 32*row tiles
      return dim1*32;
      break;
    case 3://COL_TURING
      return 32*roundoff(dim1, 8);
      break;
    case 4://COL_AMPERE
      // 32*32 tiles
      return 32*roundoff(dim1, 32);
      break;
		default:
			return 0;
			break;
  }
}

template int get_leading_dim<0>(int dim1, int dim2);//ROW
template int get_leading_dim<1>(int dim1, int dim2);//COL
template int get_leading_dim<2>(int dim1, int dim2);//COL32



/*
inline engine make_engine(
        const sycl::device &adevice, const sycl::context &acontext) {
    dnnl_engine_t aengine;
    dnnl_sycl_interop_engine_create(&aengine,
                              static_cast<const void *>(&adevice),
                              static_cast<const void *>(&acontext));
    return engine(aengine);
}
*/

template <typename T, int SRC, int TARGET, bool transpose, int DTYPE> void transform( T *A, T *out, int dim1, int dim2)
{

  using namespace dnnl;
  using tag = memory::format_tag;
  using dt = memory::data_type;
  void *Aout;
  auto dev = sycl::device(sycl::gpu_selector_v);
  auto ctx = sycl::context(dev);
  int ldA = get_leading_dim<SRC>(dim1, dim2);
  int ldOut = get_leading_dim<TARGET>(dim1, dim2);
  int ldAOut = get_leading_dim<TARGET>(dim1, dim2);
  
  dnnl::engine engine = sycl_interop::make_engine(dev, ctx);
  // column major 
  const memory::dims a_strides = memory::dims {1, ldA};
  const auto a_md = DTYPE ==32 ? memory::desc({dim1, dim2}, dt::s32, a_strides) : memory::desc({dim1, dim2}, dt::s8, a_strides);
  const memory::dims out_strides = memory::dims {ldOut, 1};
  const auto out_md = DTYPE ==32 ? memory::desc({dim1, dim2}, dt::s32, out_strides) : memory::desc({dim1, dim2}, dt::s8, out_strides);
  const memory::dims aout_strides = memory::dims {ldAOut, 1};
  const auto aout_md = DTYPE == 32 ? memory::desc({dim1, dim2}, dt::s32, aout_strides) : memory::desc({dim1, dim2}, dt::s8, aout_strides);
  
  //memory align
  memory a_mem(a_md, engine, A);
  memory out_mem(out_md, engine, out);
  memory aout_mem(aout_md, engine, Aout);
  
  //create dnnl stream
  auto q_ct1 = sycl::queue(ctx, dev);
  dnnl::stream stream = sycl_interop::make_stream(engine, q_ct1);
  
  primitive_attr attr;
  
  auto matmul_pd = matmul::primitive_desc(engine, a_md, out_md, aout_md, attr);
  auto matmul_prim = matmul(matmul_pd);
  std::unordered_map<int, memory> matmul_args;
  matmul_args.insert({DNNL_ARG_SRC, a_mem});
  matmul_args.insert({DNNL_ARG_WEIGHTS, out_mem});
  matmul_args.insert({DNNL_ARG_DST, aout_mem});

  matmul_prim.execute(stream, matmul_args);
  stream.wait();

}


/*
template void transform<int8_t, ROW, COL, false, 8>(int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, ROW, false, 8>( int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, COL32, false, 8>(int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int32_t, ROW, COL32, false, 32>( int32_t *A, int32_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, COL_TURING, false, 8>( int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, COL_AMPERE, false, 8>( int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, COL32, ROW, false, 8>( int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int32_t, COL32, ROW, false, 32>( int32_t *A, int32_t *out, int dim1, int dim2);
*/





int main(){

float A[512];
float out[512];
int dim1 = 2;
int dim2 = 2;

for(int i=0;i<512;i++){ A[i]=0.5f;out[i]=1.0f;}

transform<float, 0, 0, false, 8>(A, out, dim1, dim2);

transform<float, 0, 0, true, 8>(A, out, dim1, dim2);



return 0;

}
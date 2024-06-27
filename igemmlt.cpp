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

template <int FORMATB, int DTYPE_OUT, int SCALE_ROWS> int igemmlt( int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc)
 try {
    
    using tag = memory::format_tag;
    using dt = memory::data_type;
    auto dev = sycl::device(sycl::gpu_selector_v);
    auto ctx = sycl::context(dev);
    
    dnnl::engine engine = sycl_interop::make_engine(dev, ctx);
    // column major 
    const memory::dims a_strides = memory::dims {1, lda};
    const auto a_md = memory::desc({m, k}, dt::s8, a_strides);
    const memory::dims b_strides = memory::dims {ldb, 1};
    const auto b_md = memory::desc({k, n}, dt::s8, b_strides);
    const memory::dims c_strides = memory::dims {ldc, 1};
    const auto c_md = DTYPE_OUT == 32 ? memory::desc({m, n}, dt::s32, c_strides) : memory::desc({m, n}, dt::s8, c_strides);
    
    //memory align
    memory a_mem(a_md, engine);
    memory b_mem(b_md, engine);
    memory c_mem(c_md, engine);
    memory scales_C_mem({{1}, dt::f32, {1}}, engine, row_scale);
    
    //create dnnl stream
    auto q_ct1 = sycl::queue(ctx, dev);
    dnnl::stream stream = sycl_interop::make_stream(engine, q_ct1);
    
    primitive_attr attr;
    if (SCALE_ROWS) {
        attr.set_scales_mask(DNNL_ARG_DST, /* mask */ 1 << 1);
    }
    
    auto matmul_pd = matmul::primitive_desc(engine, a_md, b_md, c_md, attr);
    auto matmul_prim = matmul(matmul_pd);
    std::unordered_map<int, memory> matmul_args;
    matmul_args.insert({DNNL_ARG_SRC, a_mem});
    matmul_args.insert({DNNL_ARG_WEIGHTS, b_mem});
    matmul_args.insert({DNNL_ARG_DST, c_mem});

    if (SCALE_ROWS) {
      matmul_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, scales_C_mem});
    }
    matmul_prim.execute(stream, matmul_args);
    stream.wait();

}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/*
template int igemmlt<COL_TURING, 32, 0>( int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_TURING, 8, 0>( int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_TURING, 8, 1>( int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_AMPERE, 32, 0>( int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_AMPERE, 8, 0>( int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_AMPERE, 8, 1>( int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
*/




int main(){

int8_t A[512];
int8_t B[512];
int8_t out[512];
float row_scale[512];
int8_t C[512];
int m = 2;
int n = m;
int k = m;
int lda = 2;
int ldb = lda;
int ldc = lda;

for(int i=0;i<512;i++){ row_scale[i]=0.5f;A[i]=1,B[i]=1;out[i]=1;C[i]=0;}

igemmlt<4, 32, 0>( m, n, k, A, B, C, row_scale, lda, ldb, ldc);



return 0;

}
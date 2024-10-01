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
#include "blas_utils.h"


#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

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


/// Load linear segment items into block format across threads
/// Helper for Block Load
namespace dpct{
namespace group{
/*
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

*/










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

void gemmex(Context *context, bool transposeA, bool transposeB, int m, int n, int k, void *A, void *B, void *C, int lda, int ldb, int ldc)
 try {
  const int falpha = 1;
  const int fbeta = 0;
  const void * alpha = &falpha;
  const void * beta = &fbeta;
	int status;

   dpct::gemm(*context->m_handle, transposeA ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans, transposeB ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans, m, n, k, alpha, A, dpct::library_data_t::real_int8, lda, B, dpct::library_data_t::real_int8, ldb, beta, C, dpct::library_data_t::real_int32, ldc, dpct::library_data_t::real_int32);


}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void strided_gemmex(Context *context, bool transposeA, bool transposeB, int m, int n, int k, void *A, void *B, void *C, int lda, int ldb, int ldc,
                    long long int strideA, long long int strideB, long long int strideC, int batchCount)
 try {
  const int falpha = 1;
  const int fbeta = 0;
  const void * alpha = &falpha;
  const void * beta = &fbeta;
	int status;

  //cout << transposeA << transposeB << endl;
  //printf("%i %i %i\n", m,n,k);
  //printf("%i %i %i\n", lda,ldb,ldc);
  //printf("%i %i %i\n", strideA, strideB, strideC);
  //printf("%i\n", batchCount);

   dpct::gemm_batch(*context->m_handle, transposeA ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans, transposeB ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans, m, n, k, alpha, A, dpct::library_data_t::real_int8, lda, (long long int)strideA, B, dpct::library_data_t::real_int8, ldb, (long long int)strideB, beta, C, dpct::library_data_t::real_int32, ldc, (long long int)strideC, batchCount, dpct::library_data_t::real_int32);

}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int roundoff(int v, int d) {
    return (v + d - 1) / d * d;
}


template<int ORDER> int get_leading_dim(int dim1, int dim2)
{
	switch(ORDER)
	{
		case ROW:
      return dim2;
			break;
    case COL:
      return dim1;
      break;
    case COL32:
      // 32*row tiles
      return dim1*32;
      break;
    case COL_TURING:
      return 32*roundoff(dim1, 8);
      break;
    case COL_AMPERE:
      // 32*32 tiles
      return 32*roundoff(dim1, 32);
      break;
		default:
			return 0;
			break;
  }
}

template<int ORDER> dpct::blas_gemm::experimental::order_t get_order()
{
	switch(ORDER)
	{
		case ROW:
      return dpct::blas_gemm::experimental::order_t::row;
			break;
    case COL:
      return dpct::blas_gemm::experimental::order_t::col;
      break;
    case COL32:
      return dpct::blas_gemm::experimental::order_t::col32;
      break;
    case COL_TURING:
      return dpct::blas_gemm::experimental::order_t::col4_4r2_8c;
      break;
    case COL_AMPERE:
      return dpct::blas_gemm::experimental::order_t::col32_2r_4r4;
      break;
		default:
			break;
  }

	return dpct::blas_gemm::experimental::order_t::row;
}
template dpct::blas_gemm::experimental::order_t get_order<ROW>();
template dpct::blas_gemm::experimental::order_t get_order<COL>();
template dpct::blas_gemm::experimental::order_t get_order<COL32>();
template dpct::blas_gemm::experimental::order_t get_order<COL_TURING>();
template dpct::blas_gemm::experimental::order_t get_order<COL_AMPERE>();

template int get_leading_dim<ROW>(int dim1, int dim2);
template int get_leading_dim<COL>(int dim1, int dim2);
template int get_leading_dim<COL32>(int dim1, int dim2);

//=================================transform GEMM==============================

template <typename T, int SRC, int TARGET, bool transpose, int DTYPE>
void transform(dpct::queue_ptr ltHandle_, T *A,
               T *out, int dim1, int dim2)
{
#ifdef NO_CUBLASLT
#else
  dpct::blas_gemm::experimental::descriptor_ptr ltHandle;
  dpct::blas_gemm::experimental::order_t orderA = get_order<SRC>();
  dpct::blas_gemm::experimental::order_t orderOut = get_order<TARGET>();
  int ldA = get_leading_dim<SRC>(dim1, dim2);
  int ldOut = get_leading_dim<TARGET>(dim1, dim2);

  dpct::blas_gemm::experimental::matrix_layout_ptr A_desc = NULL,
                                                   out_desc = NULL;
  dpct::blas_gemm::experimental::transform_desc_ptr A2Out_desc = NULL;
  oneapi::mkl::transpose opTranspose = oneapi::mkl::transpose::trans;
  float transformAlpha = 1.0f, transformBeta = 0.0f;


  if(DTYPE == 8)
  {
    checkCublasStatus(DPCT_CHECK_ERROR(
        A_desc = new dpct::blas_gemm::experimental::matrix_layout_t(
            dpct::library_data_t::real_int8, dim1, dim2, ldA)));
    checkCublasStatus(DPCT_CHECK_ERROR(
        out_desc = new dpct::blas_gemm::experimental::matrix_layout_t(
            dpct::library_data_t::real_int8, dim1, dim2, ldOut)));
  }
  else if(DTYPE == 32)
  {
    checkCublasStatus(DPCT_CHECK_ERROR(
        A_desc = new dpct::blas_gemm::experimental::matrix_layout_t(
            dpct::library_data_t::real_int32, dim1, dim2, ldA)));
    checkCublasStatus(DPCT_CHECK_ERROR(
        out_desc = new dpct::blas_gemm::experimental::matrix_layout_t(
            dpct::library_data_t::real_int32, dim1, dim2, ldOut)));
  }
  else
  {
    printf("ERROR WRONG TYPE FOR TRANSFORM: %i\n", DTYPE);
  }

  checkCublasStatus(DPCT_CHECK_ERROR(A_desc->set_attribute(
      dpct::blas_gemm::experimental::matrix_layout_t::attribute::order,
      &orderA)));
  checkCublasStatus(DPCT_CHECK_ERROR(out_desc->set_attribute(
      dpct::blas_gemm::experimental::matrix_layout_t::attribute::order,
      &orderOut)));

  checkCublasStatus(DPCT_CHECK_ERROR(
      A2Out_desc = new dpct::blas_gemm::experimental::transform_desc_t(
          dpct::library_data_t::real_float)));

  if (transpose) {
    checkCublasStatus(DPCT_CHECK_ERROR(A2Out_desc->set_attribute(
        dpct::blas_gemm::experimental::transform_desc_t::attribute::trans_a,
        &opTranspose)));
  }

  checkCublasStatus(
      DPCT_CHECK_ERROR(dpct::blas_gemm::experimental::matrix_transform(
          A2Out_desc, &transformAlpha, A, A_desc, &transformBeta, NULL, NULL,
          out, out_desc, 0)));

  if (A_desc) checkCublasStatus(DPCT_CHECK_ERROR(delete (A_desc)));
  if (out_desc) checkCublasStatus(DPCT_CHECK_ERROR(delete (out_desc)));
  if (A2Out_desc) checkCublasStatus(DPCT_CHECK_ERROR(delete (A2Out_desc)));
#endif
}

/*
template <typename T, int SRC, int TARGET, bool transpose, int DTYPE> void transform( dpct::queue_ptr ltHandle, T *A, T *out, int dim1, int dim2)
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
*/
template void transform<int8_t, ROW, COL, false, 8>(dpct::queue_ptr ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, ROW, false, 8>(dpct::queue_ptr ltHandle,  int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, COL32, false, 8>(dpct::queue_ptr ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int32_t, ROW, COL32, false, 32>(dpct::queue_ptr ltHandle,  int32_t *A, int32_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, COL_TURING, false, 8>(dpct::queue_ptr ltHandle,  int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, COL_AMPERE, false, 8>(dpct::queue_ptr ltHandle,  int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, COL32, ROW, false, 8>(dpct::queue_ptr ltHandle,  int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int32_t, COL32, ROW, false, 32>(dpct::queue_ptr ltHandle,  int32_t *A, int32_t *out, int dim1, int dim2);


//========================igemmlt============================================
/*
template <int FORMATB, int DTYPE_OUT, int SCALE_ROWS> int igemmlt( int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc){

      dpct::device_ext &dev = dpct::get_current_device();
  
      sycl::queue &q = dev.in_order_queue();
  
     // Get the device associated with the queue
        //sycl::device dev = q.get_device();
        // Get the context associated with the queue
        sycl::context ctx = q.get_context();
        const dnnl::engine eng = dnnl::sycl_interop::make_engine(dev, ctx);
        const dnnl::stream stream = dnnl::sycl_interop::make_stream(eng, q);
        dnnl::memory::dims a_dims = { m, k };
        dnnl::memory::dims b_dims = { k, n };
        dnnl::memory::dims c_dims = { m, n };
        const auto a_in_md = dnnl::memory::desc(a_dims, at, a_trans ? tag::ba : tag::ab);
        const auto b_in_md = dnnl::memory::desc(b_dims, bt, b_trans ? tag::ba : tag::ab);
        const auto c_md = dnnl::memory::desc(c_dims, ct, tag::ab);
        auto a_mem = dnnl::memory(a_in_md, eng, (void*)a);
        auto b_mem = dnnl::memory(b_in_md, eng, (void*)b);
        auto matmul_pd = dnnl::matmul::primitive_desc(eng, a_in_md, b_in_md, c_md);
        auto c_mem = dnnl::memory(matmul_pd.dst_desc(), eng, c);

        // Create the primitive.
        auto matmul_prim = dnnl::matmul(matmul_pd);
        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> matmul_args;
        matmul_args.insert({ DNNL_ARG_SRC, a_mem });
        matmul_args.insert({ DNNL_ARG_WEIGHTS, b_mem });
        matmul_args.insert({ DNNL_ARG_DST, c_mem });

        matmul_prim.execute(stream, matmul_args);
        


}

*/



template <int FORMATB, int DTYPE_OUT, int SCALE_ROWS>
int igemmlt(int m,
            int n, int k, const int8_t *A, const int8_t *B, void *C,
            float *row_scale, int lda, int ldb, int ldc) try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
#ifdef NO_CUBLASLT
	return ERR_NOT_IMPLEMENTED;
#else
    dpct::blas_gemm::experimental::descriptor_ptr ltHandle;
    int has_error = 0;
    dpct::blas_gemm::experimental::matmul_desc_ptr matmulDesc = NULL;
    dpct::blas_gemm::experimental::matrix_layout_ptr Adesc = NULL, Bdesc = NULL,
                                                     Cdesc = NULL;
    oneapi::mkl::transpose opT = oneapi::mkl::transpose::trans;
    dpct::blas_gemm::experimental::pointer_mode_t alphaVec = dpct::blas_gemm::
        experimental::pointer_mode_t::alpha_device_vector_beta_zero;
    dpct::blas_gemm::experimental::order_t col32 =
        dpct::blas_gemm::experimental::order_t::col32;
    dpct::blas_gemm::experimental::order_t col_turing =
        dpct::blas_gemm::experimental::order_t::col4_4r2_8c;
    dpct::blas_gemm::experimental::order_t col_ampere =
        dpct::blas_gemm::experimental::order_t::col32_2r_4r4;

    has_error |= checkCublasStatus(DPCT_CHECK_ERROR(
        Adesc = new dpct::blas_gemm::experimental::matrix_layout_t(
            dpct::library_data_t::real_int8, m, k, lda)));
    has_error |= checkCublasStatus(DPCT_CHECK_ERROR(
        Bdesc = new dpct::blas_gemm::experimental::matrix_layout_t(
            dpct::library_data_t::real_int8, n, k, ldb)));

    has_error |= checkCublasStatus(DPCT_CHECK_ERROR(Adesc->set_attribute(
        dpct::blas_gemm::experimental::matrix_layout_t::attribute::order,
        &col32)));
    if(FORMATB == COL_TURING)
      has_error |= checkCublasStatus(DPCT_CHECK_ERROR(Bdesc->set_attribute(
          dpct::blas_gemm::experimental::matrix_layout_t::attribute::order,
          &col_turing)));
    else
      has_error |= checkCublasStatus(DPCT_CHECK_ERROR(Bdesc->set_attribute(
          dpct::blas_gemm::experimental::matrix_layout_t::attribute::order,
          &col_ampere)));

    if(DTYPE_OUT == 32)
    {
      has_error |= checkCublasStatus(DPCT_CHECK_ERROR(
          matmulDesc = new dpct::blas_gemm::experimental::matmul_desc_t(
              dpct::blas::compute_type::i32, dpct::library_data_t::real_int32)));
      has_error |= checkCublasStatus(DPCT_CHECK_ERROR(matmulDesc->set_attribute(
          dpct::blas_gemm::experimental::matmul_desc_t::attribute::trans_b,
          &opT)));
      has_error |= checkCublasStatus(DPCT_CHECK_ERROR(
          Cdesc = new dpct::blas_gemm::experimental::matrix_layout_t(
              dpct::library_data_t::real_int32, m, n, ldc)));
      has_error |= checkCublasStatus(DPCT_CHECK_ERROR(Cdesc->set_attribute(
          dpct::blas_gemm::experimental::matrix_layout_t::attribute::order,
          &col32)));
      int alpha = 1, beta = 0;
      has_error |= checkCublasStatus(
          DPCT_CHECK_ERROR(dpct::blas_gemm::experimental::matmul(
              ltHandle, matmulDesc, &alpha, A, Adesc, B, Bdesc, &beta,
              (int32_t *)C, Cdesc, (int32_t *)C, Cdesc, 0)));
    }
    else
    {
      has_error |= checkCublasStatus(DPCT_CHECK_ERROR(
          matmulDesc = new dpct::blas_gemm::experimental::matmul_desc_t(
              dpct::blas::compute_type::i32, dpct::library_data_t::real_float)));
      has_error |= checkCublasStatus(DPCT_CHECK_ERROR(matmulDesc->set_attribute(
          dpct::blas_gemm::experimental::matmul_desc_t::attribute::trans_b,
          &opT)));
      has_error |= checkCublasStatus(DPCT_CHECK_ERROR(
          Cdesc = new dpct::blas_gemm::experimental::matrix_layout_t(
              dpct::library_data_t::real_int8, m, n, ldc)));
      has_error |= checkCublasStatus(DPCT_CHECK_ERROR(Cdesc->set_attribute(
          dpct::blas_gemm::experimental::matrix_layout_t::attribute::order,
          &col32)));
      if(!SCALE_ROWS)
      {
        float alpha = 1.0f, beta = 0.0f;
        has_error |= checkCublasStatus(
            DPCT_CHECK_ERROR(dpct::blas_gemm::experimental::matmul(
                ltHandle, matmulDesc, &alpha, A, Adesc, B, Bdesc, &beta,
                (int8_t *)C, Cdesc, (int8_t *)C, Cdesc, 0)));
      }
      else
      {
        has_error |=
            checkCublasStatus(DPCT_CHECK_ERROR(matmulDesc->set_attribute(
                dpct::blas_gemm::experimental::matmul_desc_t::attribute::
                    pointer_mode,
                &alphaVec)));
        has_error |= checkCublasStatus(
            DPCT_CHECK_ERROR(dpct::blas_gemm::experimental::matmul(
                ltHandle, matmulDesc, row_scale, A, Adesc, B, Bdesc, NULL,
                (int8_t *)C, Cdesc, (int8_t *)C, Cdesc, 0)));
      }
    }

    if (Cdesc) has_error |= checkCublasStatus(DPCT_CHECK_ERROR(delete (Cdesc)));
    if (Bdesc) has_error |= checkCublasStatus(DPCT_CHECK_ERROR(delete (Bdesc)));
    if (Adesc) has_error |= checkCublasStatus(DPCT_CHECK_ERROR(delete (Adesc)));
    if (matmulDesc) has_error |=
        checkCublasStatus(DPCT_CHECK_ERROR(delete (matmulDesc)));
    if(has_error == 1)
      printf("error detected");

    return has_error;
#endif // NO_CUBLASLT
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
/*

template <int FORMATB, int DTYPE_OUT, int SCALE_ROWS>
int igemmlt(int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc) try {
    using namespace oneapi;

    // Get the current device, queue, and context
    dpct::device_ext &dev = dpct::get_current_device();
    sycl::queue &q_ct1 = dev.in_order_queue();

    // Assuming A, B, C are already allocated and copied to device memory.
    // Convert input pointers to appropriate types
    auto A_dev = (float *)A;
    auto B_dev = (float *)B;
    auto C_dev = (float *)C;

    // Initialize alpha and beta for GEMM
    float alpha = 1.0f;
    float beta = 0.0f;
    // Perform matrix multiplication using oneMKL GEMM function
    if (DTYPE_OUT == 32) {
        // C is int32_t
        mkl::blas::column_major::gemm(q_ct1, mkl::transpose::nontrans, mkl::transpose::nontrans, m, n, k,
                                      alpha, A_dev, lda, B_dev, ldb, beta, (float *)C_dev, ldc );
    } else {
        // C is int8_t, perform operation with scaling
        mkl::blas::column_major::gemm(q_ct1, mkl::transpose::nontrans, mkl::transpose::nontrans, m, n, k,
                                      alpha, A_dev, lda, B_dev, ldb, beta, (float *)C_dev, ldc);
    }

    // If row scaling is needed, apply the scaling manually
    if (SCALE_ROWS) {
        q_ct1.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::range<1>(m * n), [=](sycl::id<1> idx) {
                int row = idx[0] / n;
                if (DTYPE_OUT == 32) {
                    ((int32_t *)C_dev)[idx] *= row_scale[row];
                } else {
                    ((int8_t *)C_dev)[idx] *= row_scale[row];
                }
            });
        }).wait();
    }

    return 0;
} catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    return 1;
}*/

/*
template <int FORMATB, int DTYPE_OUT, int SCALE_ROWS> int igemmlt( int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc){
 std::cout<<"enter"<<std::endl;
 
   try{ 
    using tag = memory::format_tag;
    using dt = memory::data_type;
    auto dev = sycl::device(sycl::gpu_selector_v);
    auto ctx = sycl::context(dev);
    std::cout<<"enter context"<<std::endl;
 
    
    // column major 
    const memory::dims a_strides = memory::dims {1, lda};
    std::cout<<"memory dims"<<std::endl;
    const auto a_md = memory::desc({m, k}, dt::s8, a_strides);
    std::cout<<"amd"<<std::endl;
    const memory::dims b_strides = memory::dims {ldb, 1};
    const auto b_md = memory::desc({k, n}, dt::s8, b_strides);
    std::cout<<"bmd"<<std::endl;
    const memory::dims c_strides = memory::dims {ldc, 1};
    std::cout<<"c strides"<<std::endl;
    //auto c_md = memory::desc({m, n}, dt::s32, c_strides);
    //std::cout<<"cmd"<<std::endl;
    //memory align
     dnnl::engine engine = sycl_interop::make_engine(dev, ctx);
     
     
     
   
    memory a_mem(a_md, engine);
    memory b_mem(b_md, engine);
    std::cout<<"A & B mem"<<std::endl;
    memory c_mem((memory::desc({m, n}, dt::s8, c_strides)), engine);
    //memory scales_C_mem({{1}, dt::f32, {1}}, engine, row_scale);
    //sycl::ext::oneapi::experimental::printf("Memory");
    std::cout<<"Memory"<<std::endl;
    //create dnnl stream
    auto q_ct1 = sycl::queue(ctx, dev);
    dnnl::stream stream = sycl_interop::make_stream(engine, q_ct1);
    std::cout<<"stream stat"<<std::endl;
    primitive_attr attr;
    //if (SCALE_ROWS) {
    //    attr.set_scales_mask(DNNL_ARG_DST,  1 << 1);
    //}
    
    auto matmul_pd = matmul::primitive_desc(engine, a_md, b_md, (memory::desc({k, n}, dt::s8, c_strides)), attr);
    auto matmul_prim = matmul(matmul_pd);
    std::unordered_map<int, memory> matmul_args;
    matmul_args.insert({DNNL_ARG_SRC, a_mem});
    matmul_args.insert({DNNL_ARG_WEIGHTS, b_mem});
    matmul_args.insert({DNNL_ARG_DST, c_mem});
    /*
    if (SCALE_ROWS) {
      matmul_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, scales_C_mem});
    }
    
    try{
    
    matmul_prim.execute(stream, matmul_args);
    stream.wait();
    
    std::cout<<"wait for stream"<<std::endl;
    sycl::ext::oneapi::experimental::printf("Wait for stream");
    return 0;
    
    
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  return 1;
}
return 0;}
*/

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

//============================gemm 4 bit inference naive =================

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

//===============================spm _coo _very _sparse=========================

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

//==============================================================
//                   TEMPLATE DEFINITIONS
//==============================================================

template void gemm_host<float>(int m, int n, int k, float * A,  float* B,  float * out,  int lda, int ldb, int ldc, int bits);
template void gemm_host<sycl::half>(int m, int n, int k, sycl::half * A,  sycl::half* B,  sycl::half * out,  int lda, int ldb, int ldc, int bits);


template void gemm_4bit_inference<sycl::half>(int m, int n, int k, sycl::half * A,  unsigned char* B,  float *absmax, sycl::half * out,  int lda, int ldb, int ldc, int blocksize);

template void gemm_4bit_inference_naive<sycl::half, 16>(int m, int n, int k, sycl::half * A,  unsigned char* B,  float *absmax, float *datatype, sycl::half * out,  int lda, int ldb, int ldc, int blocksize);
template void gemm_4bit_inference_naive<bf16, 16>(int m, int n, int k, bf16 * A,  unsigned char* B,  float *absmax, float *datatype, bf16 * out,  int lda, int ldb, int ldc, int blocksize);
template void gemm_4bit_inference_naive<float, 32>(int m, int n, int k, float * A,  unsigned char* B,  float *absmax, float *datatype, float * out,  int lda, int ldb, int ldc, int blocksize);

template void spmm_coo_very_sparse_naive<sycl::half, 16>(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, sycl::half *values, sycl::half *B, sycl::half *out, float *dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB);
template void spmm_coo_very_sparse_naive<signed char, 8>(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, sycl::half *values, signed char *B, sycl::half *out, float *dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB);

template int igemmlt<COL_TURING, 32, 0>( int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_TURING, 8, 0>( int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_TURING, 8, 1>( int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_AMPERE, 32, 0>( int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_AMPERE, 8, 0>( int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_AMPERE, 8, 1>( int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);

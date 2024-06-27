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

//===================================gemm_4bit ================


void spmm_coo(int *A_rowidx, int *A_colidx, sycl::half *A_vals, int A_nnz, int A_rows, int A_cols, int B_cols, int ldb, sycl::half *B, int ldc, sycl::half* C, bool transposed_B)
{ 

  try{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  
    dpct::sparse::sparse_matrix_desc_t descA;
    std::shared_ptr<dpct::sparse::dense_matrix_desc> descB, descC;

    float alpha = 1.0f;
    float beta = 0.0f;
    void *dBuffer = NULL;
    size_t bufferSize = 0;

    
    // Create dense matrix C
    
    descC = std::make_shared<dpct::sparse::dense_matrix_desc>(A_rows, B_cols, ldc, C, dpct::library_data_t::real_half, oneapi::mkl::layout::row_major);
    // Create dense matrix B
    if(transposed_B)
    {
      int tmp = A_cols;
      A_cols = B_cols;
      B_cols = tmp;
    }

    
    descB = std::make_shared<dpct::sparse::dense_matrix_desc>(A_cols, B_cols, ldb, B, dpct::library_data_t::real_half, oneapi::mkl::layout::row_major);
    // allocate an external buffer if needed
    
    bufferSize = 0;
    
    dBuffer = (void *)sycl::malloc_device(bufferSize, q_ct1);

    
    dpct::sparse::spmm(q_ct1, oneapi::mkl::transpose::nontrans, transposed_B ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans, &alpha, descA, descB, &beta, descC, dpct::library_data_t::real_float);
    // destroy matrix/vector descriptors
    descA.reset();
    descB.reset();
    descC.reset();
    sycl::free(dBuffer, q_ct1);
    
  }
  catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
      std::exit(1);
  }

}




int main(){

int A_colidx[512];
int A_rowidx[512];
sycl::half A_vals[512];
int A_nnz=1;

int A_rows =1;
int A_cols=1;
int B_cols = A_cols;
int ldb=1;

bool transposed_B = false;
sycl::half C[512];sycl::half B[512];
int ldc =ldb; 

for(int i=0;i<512;i++){ A_rowidx[i]=1;A_colidx[i]=1,A_vals[i]=sycl::half(1.0f);B[i]=sycl::half(1.0f);C[i]=sycl::half(1.0f);}


spmm_coo(A_rowidx, A_colidx, A_vals, A_nnz, A_rows, A_cols, B_cols, ldb, B, ldc, C, transposed_B);

return 0;

}
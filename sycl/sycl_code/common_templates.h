#ifndef COMMON_TEMPLATES_H
#define COMMON_TEMPLATES_H

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

#define NUM_BLOCK 4096

inline constexpr int fill_up_to_nearest_multiple(int value, int multiple)
{
  return value + (value % multiple == 0 ? 0 : (multiple - (value % multiple)));
}

template <typename T, int FUNC> void func(T *A, T *B, T value, long n)
{
  int threads = 512;
  int blocks = n/threads;
  blocks = n % threads == 0 ? blocks : blocks + 1;
  blocks = blocks > 65535 ? 65535 : blocks;
  
  dpct::get_in_order_queue().submit(
    [&](sycl::handler &cgh) {
    
  cgh.parallel_for(
    sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
    [=](sycl::nd_item<3> item_ct1) {
      kfunc<T, FUNC>(A, B, value, n, item_ct1);
    });
  });
  
}
//======================================= transform row to format===============================================

template <int FORMAT, int TRANSPOSE> void transformRowToFormat(char * A, char *out, int rows, int cols)
{
  
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
	int num_blocks = 0;
  int size = NUM_BLOCK;
  
  sycl::buffer<char, 1> buff_A(A,sycl::range<1>(size));
  sycl::buffer<char, 1> buff_out(out,sycl::range<1>(size));
  
  
  int threads = 256;
  int items_per_thread = 8;
  // we load 128 column values per warp
  int tile_cols = 32*items_per_thread;
  int tile_rows = 32;
  int tiledCols = fill_up_to_nearest_multiple(cols, tile_cols);
  int tiledRows = fill_up_to_nearest_multiple(rows, tile_rows);
	int row_tiles = (tiledRows/tile_rows);
	int col_tiles = (tiledCols/tile_cols);
	row_tiles = row_tiles > 0 ? row_tiles : 1;
	col_tiles = col_tiles > 0 ? col_tiles : 1;
  num_blocks = row_tiles * col_tiles;

  int outCols = fill_up_to_nearest_multiple(cols, 32);
  int outRows = fill_up_to_nearest_multiple(rows, 32);
  if(FORMAT == COL_TURING)
  {
    if(TRANSPOSE)
      outRows = fill_up_to_nearest_multiple(cols, 8);
    else
      outRows = fill_up_to_nearest_multiple(rows, 8);
  }
  else if(FORMAT == COL_AMPERE)
  {
    if(TRANSPOSE)
      outRows = fill_up_to_nearest_multiple(cols, 32);
    else
      outRows = fill_up_to_nearest_multiple(rows, 32);
  }
  else
  {
    if(TRANSPOSE)
    {
      outCols = fill_up_to_nearest_multiple(rows, 32);
      outRows = cols;
    }
  }

  
  dpct::get_in_order_queue().submit(
    [&](sycl::handler &cgh) {
     
     sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
     sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
            
      
    //__shared__ vars
      sycl::local_accessor<char, 1> smem_data_acc_ct1(sycl::range<1>(32*33*8), cgh);

      cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)), 
        [=](sycl::nd_item<3> item_ct1) {
          kTransformRowToFormat<256, 8, 32, 32*8, TRANSPOSE, FORMAT>(A, out, rows, cols, tiledCols, outRows, outCols, item_ct1, smem_data_acc_ct1.get_pointer(), dacc_A, dacc_out);
        });
    });
  
}

//===========================extract outliers===========================

template <int FORMAT> void extractOutliers(char * A, int *idx, char *out, int idx_size, int rows, int cols)
{
  int threads = 256;
  // we load 128 column values per warp
  int tiledCols = tiledCols = fill_up_to_nearest_multiple(cols, 32);
  int tiledRows = 0;
  int size = NUM_BLOCK;
  
	int num_blocks = idx_size;

  if(FORMAT == COL_TURING)
  {
      tiledRows = fill_up_to_nearest_multiple(rows, 8);
  }
  else if(FORMAT == COL_AMPERE)
  {
      tiledRows = fill_up_to_nearest_multiple(rows, 32);
	}

  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
  sycl::buffer<char, 1> buff_A(A,sycl::range<1>(size));
  sycl::buffer<char, 1> buff_out(out,sycl::range<1>(size));
  
  
  dpct::get_in_order_queue().submit(
    [&](sycl::handler &cgh) {
    
    
      sycl::accessor dacc_A(buff_A, cgh, sycl::read_write);
     sycl::accessor dacc_out(buff_out, cgh, sycl::read_write);
    
    cgh.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)), 
      [=](sycl::nd_item<3> item_ct1) {
           kExtractOutliers<FORMAT>(A, idx, out, idx_size, rows, cols, tiledRows, tiledCols, item_ct1, dacc_A, dacc_out);
    });
   });
}
#endif // COMMON_TEMPLATES_H
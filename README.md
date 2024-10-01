### Sample BitsandBytes SYCL kernel Playground

This repository hosts piece wise kernels for bitsandbytes - from the consolidated PR: https://github.com/TimDettmers/bitsandbytes/pull/747
It serves as a testing ground of the SYCL implementation and pytorch linkage. 

### Running individual files

Individual kernels can be run using icpx :

```bash
icpx -fsycl <kernel.cpp> -o <kernel_out.o> -ldnnl
```

Running the end to end optimizer for 4/8 quants:

```bash
For compilation on INTEL GPU :
$ export BUILD_SYCL=1
$ cmake -DCMAKE_CXX_COMPILER=icpx -DSYCL_TARGET=INTEL -DCMAKE_C_COMPILER=icx -DCOMPUTE_BACKEND=sycl
$ cmake –build . –config Release

For compilation on NVIDIA GPU :
$ export BUILD_SYCL=1
$ cmake -DCMAKE_CXX_COMPILER=icpx -DSYCL_TARGET=NVIDIA -DCMAKE_C_COMPILER=icx -DCOMPUTE_BACKEND=sycl
$ cmake –build . –config Release

(this will create & link bitsandbytes/libbitsandbytes_sycl.so )
Navigate to ../bitsandbytes_sycl_samples/
$pip install -e.

(this will link with python package builder)
$python setup.py install 
```


Once done, it will create a python package named : python_src_quants (which is bitsandbytes renamed differently)
The next step involves testing on iGPU (pvc) only .
Navigate to /bitsandbytes_sycl_samples folder

```bash

$python tests_pvc/import.py 

This will give the corresponding output on PVC:


2024-07-30 08:33:49,256 - python_src_quants.cextension - WARNING - The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
tensor(26.0896, device='xpu:0', grad_fn=<SumBackward0>) Adam (
Parameter Group 0
    betas: (0.9, 0.999)
    eps: 1e-08
    lr: 0.001
    weight_decay: 0
)
51.25648498535156 51.25648498535156 0.0
SUCCESS!
Installation was successful!

```
For running matmul igemmlt tests , run:

```bash
$python tests_pvc/test_matmulqlt.py

This will give the corresponding output on PVC:

/bitsandbytes_sycl_samples/python_src_quants/libbitsandbytes_sycl.so
Test standard matmul pass
Test igemmlt dim3 matmul pass
Test standard ibmm matmul pass


```

The other tests are being added inside the tests_pvc folder. 


Training Log for Fp32/16 8 bit quantization without igemmlt:

```bash
$python tests_pvc/test_simple_nn.py

This will give the corresponding output on PVC:

/home/majumder/bb_sycl_samples_fresh/bitsandbytes_sycl_samples/python_src_quants/libbitsandbytes_sycl.so
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 2.3408234119415283
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 4.681646823883057
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 7.022470235824585
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 9.363293647766113
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 11.704117059707642
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 14.04494047164917
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 16.3857638835907
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 18.726587295532227
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 21.067410707473755
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 23.408234119415283
Epoch [1/5], Step [10/32], Loss: 2.3408
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 2.3408234119415283
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 4.681646823883057
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 7.022470235824585
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 9.363293647766113
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 11.704117059707642
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 14.04494047164917
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 16.3857638835907
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 18.726587295532227
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 21.067410707473755
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 23.408234119415283
Epoch [1/5], Step [20/32], Loss: 2.3408
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 2.3408234119415283
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 4.681646823883057
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 7.022470235824585
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 9.363293647766113
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 11.704117059707642
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 14.04494047164917
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 16.3857638835907
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 18.726587295532227
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 21.067410707473755
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 23.408234119415283
Epoch [1/5], Step [30/32], Loss: 2.3408
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 2.3408234119415283
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (8, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (8, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 4.681646823883057
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 2.3408234119415283
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 4.681646823883057
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 7.022470235824585
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 9.363293647766113
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 11.704117059707642
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 14.04494047164917
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 16.3857638835907
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 18.726587295532227
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 21.067410707473755
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 23.408234119415283
Epoch [2/5], Step [10/32], Loss: 2.3408
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 2.3408234119415283
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 4.681646823883057
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 7.022470235824585
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 9.363293647766113
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 11.704117059707642
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 14.04494047164917
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 16.3857638835907
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 18.726587295532227
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 21.067410707473755
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 23.408234119415283
Epoch [2/5], Step [20/32], Loss: 2.3408
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 2.3408234119415283
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 4.681646823883057
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 7.022470235824585
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 9.363293647766113
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 11.704117059707642
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 14.04494047164917
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 16.3857638835907
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 18.726587295532227
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 21.067410707473755
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 23.408234119415283
Epoch [2/5], Step [30/32], Loss: 2.3408
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 2.3408234119415283
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (8, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (8, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 4.681646823883057
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 2.3408234119415283
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 4.681646823883057
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 7.022470235824585
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 9.363293647766113
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 11.704117059707642
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 14.04494047164917
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 16.3857638835907
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 18.726587295532227
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 21.067410707473755
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 23.408234119415283
Epoch [3/5], Step [10/32], Loss: 2.3408
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 2.3408234119415283
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 4.681646823883057
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 7.022470235824585
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 9.363293647766113
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 11.704117059707642
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 14.04494047164917
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 16.3857638835907
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 18.726587295532227
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 21.067410707473755
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 23.408234119415283
Epoch [3/5], Step [20/32], Loss: 2.3408
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 2.3408234119415283
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 4.681646823883057
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 7.022470235824585
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 9.363293647766113
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 11.704117059707642
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 14.04494047164917
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 16.3857638835907
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 18.726587295532227
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 21.067410707473755
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 23.408234119415283
Epoch [3/5], Step [30/32], Loss: 2.3408
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 2.3408234119415283
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (8, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (8, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 4.681646823883057
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 2.3408234119415283
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 4.681646823883057
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 7.022470235824585
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 9.363293647766113
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 11.704117059707642
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 14.04494047164917
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 16.3857638835907
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 18.726587295532227
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 21.067410707473755
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 23.408234119415283
Epoch [4/5], Step [10/32], Loss: 2.3408
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 2.3408234119415283
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 4.681646823883057
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 7.022470235824585
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 9.363293647766113
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 11.704117059707642
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 14.04494047164917
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 16.3857638835907
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 18.726587295532227
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 21.067410707473755
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 23.408234119415283
Epoch [4/5], Step [20/32], Loss: 2.3408
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 2.3408234119415283
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 4.681646823883057
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 7.022470235824585
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 9.363293647766113
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 11.704117059707642
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 14.04494047164917
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 16.3857638835907
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 18.726587295532227
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 21.067410707473755
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 23.408234119415283
Epoch [4/5], Step [30/32], Loss: 2.3408
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 2.3408234119415283
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (8, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (8, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 4.681646823883057
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 2.3408234119415283
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 4.681646823883057
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 7.022470235824585
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 9.363293647766113
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 11.704117059707642
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 14.04494047164917
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 16.3857638835907
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 18.726587295532227
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 21.067410707473755
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 23.408234119415283
Epoch [5/5], Step [10/32], Loss: 2.3408
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 2.3408234119415283
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 4.681646823883057
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 7.022470235824585
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 9.363293647766113
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 11.704117059707642
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 14.04494047164917
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 16.3857638835907
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 18.726587295532227
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 21.067410707473755
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 23.408234119415283
Epoch [5/5], Step [20/32], Loss: 2.3408
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 2.3408234119415283
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 4.681646823883057
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 7.022470235824585
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 9.363293647766113
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 11.704117059707642
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 14.04494047164917
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 16.3857638835907
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 18.726587295532227
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 21.067410707473755
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 23.408234119415283
Epoch [5/5], Step [30/32], Loss: 2.3408
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (32, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 2.3408234119415283
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (8, 256)
Initiate Matmul
performing mul
gemmm complete
MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
double quant done
quasnt b
output shape (8, 10)
Initiate Matmul
performing mul
gemmm complete
Loss
Loss: 4.681646823883057
Training complete.
```
Gemm kernels and more tests are under progress

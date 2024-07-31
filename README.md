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
For compilation on INTEL GPU (from tests/ folder):
$ export BUILD_SYCL=1
$ cmake -DCMAKE_CXX_COMPILER=icpx -DSYCL_TARGET=INTEL -DCMAKE_C_COMPILER=icx -DCOMPUTE_BACKEND=sycl
$ cmake –build . –config Release

For compilation on NVIDIA GPU (from tests/ folder):
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
Navigate to ../bitsandbytes_sycl_samples/tests_pvc folder

```bash

$python import.py 

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
The other tests are being added inside the tests_pvc folder. 

Gemm kernels and more tests are under progress

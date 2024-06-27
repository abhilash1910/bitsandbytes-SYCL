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
For compilation (from tests/ folder):
$ export BUILD_SYCL=1
$ cmake -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx -DCOMPUTE_BACKEND=sycl
$ cmake –build . –config Release

(this will create & link bitsandbytes/libbitsandbytes_sycl.so )
$pip install .

To check whether the kernels are correctly linked, please run (also in cextension.py L 91 replace it with your absolute path to libbitsandbytes_sycl.so):
$cd python_src_quants
$ python cextension.py
```
Gemm kernels and more tests are under progress

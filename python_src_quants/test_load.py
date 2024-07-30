import ctypes as ct
import os

# Load dependencies first (example: libsycl.so)
#ct.cdll.LoadLibrary('/home/ftartagl/oneapi/inteloneapi-basekit-hpckit.2023.2.003/compiler/2023.2.0/linux/lib/libsycl.so.6')

# Now load the target library
binary_path = "/home/majumder/bb_sycl_samples/bitsandbytes_sycl_samples/bitsandbytes/libbitsandbytes_sycl.so"
dll = ct.cdll.LoadLibrary(binary_path)
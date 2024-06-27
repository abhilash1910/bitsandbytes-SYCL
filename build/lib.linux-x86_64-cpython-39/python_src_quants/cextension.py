"""
extract factors the build is dependent on:
[X] compute capability
    [ ] TODO: Q - What if we have multiple GPUs of different makes?
- CUDA version
- Software:
    - CPU-only: only CPU quantization functions (no optimizer, no matrix multiple)
    - CuBLAS-LT: full-build 8-bit optimizer
    - no CuBLAS-LT: no 8-bit matrix multiplication (`nomatmul`)

evaluation:
    - if paths faulty, return meaningful error
    - else:
        - determine CUDA version
        - determine capabilities
        - based on that set the default path
"""

import ctypes as ct
import logging
import os
from pathlib import Path

import torch
import intel_extension_for_pytorch

import platform

DYNAMIC_LIBRARY_SUFFIX = {
    "Darwin": ".dylib",
    "Linux": ".cpp.o",
    "Windows": ".dll",
}.get(platform.system(), ".cpp.o")

PACKAGE_DIR = Path(__file__).parent




logger = logging.getLogger(__name__)


def get_sycl_bnb_library_path() -> Path:
    
    library_name = f"libbitsandbytes_sycl"
    library_name = f"{library_name}{DYNAMIC_LIBRARY_SUFFIX}"

    override_value = os.environ.get("BNB_CUDA_VERSION")
    if override_value:
        library_name_stem, _, library_name_ext = library_name.rpartition(".")
        # `library_name_stem` will now be e.g. `libbitsandbytes_cuda118`;
        # let's remove any trailing numbers:
        library_name_stem = library_name_stem.rstrip("0123456789")
        # `library_name_stem` will now be e.g. `libbitsandbytes_cuda`;
        # let's tack the new version number and the original extension back on.
        library_name = f"{library_name_stem}{override_value}.{library_name_ext}"
        logger.warning(
            f"WARNING: BNB_CUDA_VERSION={override_value} environment variable detected; loading {library_name}.\n"
            "This can be used to load a bitsandbytes version that is different from the PyTorch CUDA version.\n"
            "If this was unintended set the BNB_CUDA_VERSION variable to an empty string: export BNB_CUDA_VERSION=\n"
            "If you use the manual override make sure the right libcudart.so is in your LD_LIBRARY_PATH\n"
            "For example by adding the following to your .bashrc: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path_to_cuda_dir/lib64\n",
        )

    return PACKAGE_DIR / library_name

class BNBNativeLibrary:
    _lib: ct.CDLL
    compiled_with_sycl= False

    def __init__(self, lib: ct.CDLL):
        self._lib = lib

    def __getattr__(self, item):
        return getattr(self._lib, item)


class SYCLBNBNativeLibrary(BNBNativeLibrary):
    compiled_with_sycl = True

    def __init__(self, lib: ct.CDLL):
        super().__init__(lib)
        lib.get_context.restype = ct.c_void_p
        lib.get_cusparse.restype = ct.c_void_p
        lib.cget_managed_ptr.restype = ct.c_void_p


def get_native_library() -> BNBNativeLibrary:
    binary_path = PACKAGE_DIR / f"libbitsandbytes_sycl.so"
    sycl_binary_path = binary_path
    """
    if sycl_binary_path.exists():
        binary_path = sycl_binary_path
    else:
        logger.warning("Could not find the bitsandbytes sycl binary at %r", sycl_binary_path)
    """
    logger.debug(f"Loading bitsandbytes native library from: {binary_path}")
    dll = ct.cdll.LoadLibrary(str(binary_path))

    if hasattr(dll, "get_context"):  # only a CUDA-built library exposes this
        return SYCLBNBNativeLibrary(dll)

    logger.warning(
        "The installed version of bitsandbytes was compiled without GPU support. "
        "8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.",
    )
    return BNBNativeLibrary(dll)


try:
    lib = get_native_library()
except Exception as e:
    lib = None
    logger.error(f"Could not load bitsandbytes native library: {e}", exc_info=True)
    if torch.xpu.is_available():
        logger.warning(
            """
SYCL Setup failed despite CUDA being available. Please run the following command to get more information:

python -m bitsandbytes

Inspect the output of the command and see if you can locate SYCL libraries. You might need to add them
to your LD_LIBRARY_PATH. If you suspect a bug, please take the information from python -m bitsandbytes
and open an issue at: https://github.com/TimDettmers/bitsandbytes/issues
""",
        )
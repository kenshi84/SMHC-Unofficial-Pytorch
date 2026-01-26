import glob
import os
import platform
import sys

from rich.console import Console
from setuptools import setup

ABS_ROOT = os.path.dirname(os.path.abspath(__file__))
WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "0") == "1"
LINE_INFO = os.getenv("LINE_INFO", "0") == "1"


def get_cpp_ext():
    from torch.utils.cpp_extension import BuildExtension

    return BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)


def get_openmp_args():
    from torch.__config__ import parallel_info

    info = parallel_info()
    # check if PyTorch was built with OpenMP support
    has_openmp = "backend: OpenMP" in info and "OpenMP not found" not in info

    # skip OpenMP on macOS by default (often requires external libomp)
    if has_openmp and sys.platform != "darwin":
        if sys.platform == "win32":
            omp_flags = ["/openmp"]
        else:
            omp_flags = ["-fopenmp"]

        # return flags for the C++ compiler and the NVCC driver
        return {
            "cxx": omp_flags + ["-DAT_PARALLEL_OPENMP"],
            "nvcc": [f"-Xcompiler={','.join(omp_flags)}, -O3", "-DAT_PARALLEL_OPENMP"],
            "link": omp_flags,
        }

    Console().print(
        "[yellow]smhc: no support for OpenMP found, compiling without OpenMP ...[/yellow]"
    )
    return {"cxx": [], "nvcc": [], "link": []}


def get_extensions():
    from torch.utils.cpp_extension import CUDAExtension

    extensions_dir = os.path.join("src", "smhc", "cuda", "csrc")
    sources = glob.glob(os.path.join(extensions_dir, "*.cu")) + glob.glob(
        os.path.join(extensions_dir, "*.cpp")
    )

    # remove generated 'hip' files, in case of rebuilds
    sources = [path for path in sources if "hip" not in path]
    undef_macros = []
    define_macros = []

    extra_compile_args = {"cxx": ["-O3"]}
    extra_link_args = [] if WITH_SYMBOLS else ["-s"]

    # Compile for mac arm64
    if sys.platform == "darwin" and platform.machine() == "arm64":
        extra_compile_args["cxx"] += ["-arch", "arm64"]
        extra_link_args += ["-arch", "arm64"]

    nvcc_flags = os.getenv("NVCC_FLAGS", "")
    nvcc_flags = [] if nvcc_flags == "" else nvcc_flags.split(" ")
    nvcc_flags += ["-O3", "--use_fast_math"]
    if LINE_INFO:
        nvcc_flags += ["-lineinfo"]

    extra_compile_args["nvcc"] = nvcc_flags

    omp_args = get_openmp_args()
    extra_compile_args["cxx"] += omp_args["cxx"]
    extra_compile_args["nvcc"] += omp_args["nvcc"]
    extra_link_args += omp_args["link"]

    if sys.platform == "win32":
        extra_compile_args["nvcc"] += ["-DWIN32_LEAN_AND_MEAN"]

    extension = CUDAExtension(
        f"smhc.csrc",
        sources,
        include_dirs=[
            os.path.join(ABS_ROOT, extensions_dir, "third_party", "cudaKDTree"),
            os.path.join(ABS_ROOT, extensions_dir, "third_party", "eigen"),
            os.path.join(ABS_ROOT, extensions_dir, "third_party", "cyCodeBase"),
        ],
        define_macros=define_macros,
        undef_macros=undef_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )

    return [extension]


setup(
    ext_modules=get_extensions(),
    cmdclass={"build_ext": get_cpp_ext()},
)

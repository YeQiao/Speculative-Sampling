"""Build script for CPU SSM kernels."""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

src_dir = os.path.join(os.path.dirname(__file__), "spec_mamba", "cpu_kernels")

setup(
    name="cpu_ssm_ops",
    ext_modules=[
        CppExtension(
            name="spec_mamba.cpu_kernels._cpu_ssm_ops",
            sources=[os.path.join(src_dir, "ssm_ops.cpp")],
            extra_compile_args=[
                "-O3",
                "-march=native",
                "-mavx512f",
                "-mavx512bw",
                "-mavx512dq",
                "-mavx512vl",
                "-mfma",
                "-fopenmp",
                "-std=c++17",
            ],
            extra_link_args=["-lgomp"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)

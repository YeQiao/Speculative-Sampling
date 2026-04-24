"""CPU kernel build and import utilities."""

import os
import torch
from torch.utils.cpp_extension import load

_module = None

def get_cpu_ssm_ops():
    """Lazy-compile and return the C++ SSM ops module."""
    global _module
    if _module is not None:
        return _module

    src_dir = os.path.dirname(os.path.abspath(__file__))
    src_file = os.path.join(src_dir, "ssm_ops.cpp")

    _module = load(
        name="_cpu_ssm_ops",
        sources=[src_file],
        extra_cflags=[
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
        extra_ldflags=["-lgomp"],
        verbose=False,
    )
    return _module

"""CPU kernel build and import utilities."""

import os
import torch
from torch.utils.cpp_extension import load

_module = None
_fused_module = None

_COMMON_CFLAGS = [
    "-O3",
    "-march=native",
    "-mavx512f",
    "-mavx512bw",
    "-mavx512dq",
    "-mavx512vl",
    "-mfma",
    "-fopenmp",
    "-std=c++17",
]
_COMMON_LDFLAGS = ["-lgomp"]

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
        extra_cflags=_COMMON_CFLAGS,
        extra_ldflags=_COMMON_LDFLAGS,
        verbose=False,
    )
    return _module


def get_fused_forward():
    """Lazy-compile and return the fused Mamba2 forward module."""
    global _fused_module
    if _fused_module is not None:
        return _fused_module

    src_dir = os.path.dirname(os.path.abspath(__file__))
    src_file = os.path.join(src_dir, "fused_forward.cpp")

    _fused_module = load(
        name="_fused_mamba2_forward",
        sources=[src_file],
        extra_cflags=_COMMON_CFLAGS + ["-mavx512bf16", "-mavx512vnni", "-ffast-math"],
        extra_ldflags=_COMMON_LDFLAGS,
        verbose=False,
    )
    return _fused_module

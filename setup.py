from setuptools import setup
import os
import sys
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Ensure CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This package requires CUDA.")

# Define the CUDA extension
cuda_extension = CUDAExtension(
    name='interpolator.interpolator',
    sources=[
        os.path.join('interpolator', 'python_bindings.cpp'),
        os.path.join('interpolator', 'interpolator_forw.cu')
    ],
    extra_compile_args={
        'cxx': ['-O3'],
        'nvcc': ['-O3', '-std=c++14']
    },
    include_dirs=[os.path.join('interpolator')]
)

setup(
    name='interpolator',
    version='0.1.1',
    packages=['interpolator'],
    ext_modules=[cuda_extension],
    cmdclass={
        'build_ext': BuildExtension
    }
) 
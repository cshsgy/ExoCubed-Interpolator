from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os

setup(
    name='interpolator',
    ext_modules=[
        CUDAExtension(
            name='interpolator._C',  # Note the ._C to make it a proper extension
            sources=['python_bindings.cpp', 'interpolator.cu'],
            include_dirs=[
                torch.utils.cpp_extension.include_paths(),
                os.path.dirname(os.path.abspath(__file__))
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-std=c++14']
            }
        )
    ],
    packages=['interpolator'],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=['torch']
) 
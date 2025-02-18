from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='interpolator',
    ext_modules=[
        CUDAExtension('interpolator', [
            'python_bindings.cpp',
            'interp_forward.cu',
            # 'interp_backward.cu',  # Uncomment when backward pass is implemented
        ],
        extra_compile_args={
            'cxx': ['-O2'],
            'nvcc': ['-O2']
        })
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=['torch']
) 
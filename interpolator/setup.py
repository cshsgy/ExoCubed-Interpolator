from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='interpolator',
    ext_modules=[
        CUDAExtension('interpolator', [
            'python_bindings.cpp',
            'interpolator_forw.cu'
            # 'interpolator_back.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=['torch']
) 
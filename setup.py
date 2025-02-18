from setuptools import setup

# Initialize empty extension modules and cmdclass
ext_modules = []
cmdclass = {}

# Conditionally add CUDA extension
try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    ext_modules = [
        CUDAExtension('interpolator', [
            'interpolator/python_bindings.cpp',
            'interpolator/interpolator_forw.cu'
            # 'interpolator_back.cu',
        ])
    ]
    cmdclass = {
        'build_ext': BuildExtension
    }
except ImportError:
    pass

setup(
    name='interpolator',
    packages=['interpolator'],
    install_requires=['torch', 'numpy'],  # Added numpy as it's required by PyTorch
    setup_requires=['torch'],
    ext_modules=ext_modules,
    cmdclass=cmdclass
) 
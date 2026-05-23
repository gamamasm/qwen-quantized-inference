from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='w8a16_qwen_gemm',
    ext_modules=[
        CUDAExtension(
            name='w8a16_qwen_gemm',
            sources=['w8a16_qwen_gemm.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)

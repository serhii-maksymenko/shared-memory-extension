from setuptools import setup, Extension
import numpy as np

module = Extension(
    'shared_memory_extension',
    sources=['src/shared_memory_extension.cpp'],
    include_dirs=[np.get_include()],
    extra_link_args=["-lrt"]
)


setup(
    name='shared_memory_extension',
    version='1.0',
    ext_modules=[module]
)

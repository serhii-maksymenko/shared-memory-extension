from setuptools import setup, Extension
import numpy

module = Extension(
    'shared_memory_extension',
    sources=['shared_memory_extension.cpp'],
    include_dirs=[numpy.get_include()],
    extra_link_args=["-lrt"]
)

setup(
    name='shared_memory_extension',
    version='1.0',
    ext_modules=[module]
)

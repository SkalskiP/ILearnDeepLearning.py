# ------------------------------------------------------------------------------
# File was originally part of Stamford CS231N course: https://cs231n.github.io/
# ------------------------------------------------------------------------------

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
  Extension('fast_conv_cython', ['fast_conv_cython.pyx'],
    include_dirs = [numpy.get_include()]),
]

setup(
    ext_modules = cythonize(extensions),
)
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import sys

ext_modules = [
    Extension(
        "FasterCoco",
        sources=["bindings.cpp", "fft_convolver.cpp"],
        include_dirs=[pybind11.get_include(), "."],
        language="c++"
    )
]

setup(
    name="FasterCoco",
    version="0.1",
    author="david",
    description="FFT-based 2D Convolver using Pybind11",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
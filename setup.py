import platform
import sys

import numpy
from setuptools import Extension, setup

with open("README.md", "r") as fh:
    long_description = fh.read()


class LazyCythonize(list):
    def __init__(self, callback):
        self._list, self.callback = None, callback

    def c_list(self):
        if self._list is None:
            self._list = self.callback()
        return self._list

    def __iter__(self):
        for e in self.c_list():
            yield e

    def __getitem__(self, ii):
        return self.c_list()[ii]

    def __len__(self):
        return len(self.c_list())


def extensions():
    from Cython.Build import cythonize

    include_dirs = [
        "fcbkwrapper/src/core/include",
        numpy.get_include(),
    ]

    is_windows = (platform.system() == "Windows") or (sys.platform == "win32")

    define_macros = [("NDEBUG", "1"), ("FCBK_ENABLE_OB_CHECKS", "0")]
    if is_windows:
        # MSVC-style flags
        extra_compile_args = [
            "/std:c++17",  # C++ version 17
            "/EHsc",  # Compiler assumes exceptions can only occur at throw (faster code)
            "/O2",  # Favor speed over size when compiling
            "/DNDEBUG",  # No debug checking (faster code)
        ]
        extra_link_args = []
    else:
        # Unix-like (Linux) flags as requested
        extra_compile_args = [
            "-std=c++17",
            "-g",
            "-O3",
            "-march=native",
            "-mtune=native",
            "-DNDEBUG",
            "-m64",
        ]
        extra_link_args = ["-m64"]

    maxflow_module = Extension(
        "fcbkwrapper._fcbk",
        ["fcbkwrapper/src/_fcbk.pyx"],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    )
    return cythonize([maxflow_module])


setup(
    name="fcbkwrapper",
    version="1.0.0",
    author="Christian Mikkelstrup",
    author_email="cmomi@dtu.dk",
    description="A thin wrapper for Python of the fcBK algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["fcbkwrapper"],
    install_requires=["numpy"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    ext_modules=LazyCythonize(extensions),
    setup_requires=["Cython", "numpy"],
)

"""
Build script for the _iq_native extension.

Integration into an existing setuptools-based package:
  1. Copy the `_iq_native/` directory into your package (e.g. next to your
     top-level `__init__.py`), and `maxlfq.py` alongside it (adjust the
     `from . import _iq_native` import if you place it elsewhere).
  2. Merge the `ext_modules=[...]` block below into your own setup.py's
     setup() call (or keep this as a standalone setup.py if your package
     doesn't have one yet).
  3. Add "pybind11>=2.11" to your build-system requires in pyproject.toml
     (see pyproject.toml.snippet in this folder).
  4. `pip install -e .` to build in place.

Eigen is header-only and is vendored under `_iq_native/thirdparty/eigen/`
(trimmed to the dense modules this code actually uses -- see the LICENSE.txt
there). This means the build needs no system C++ dependency on Linux,
macOS, or Windows. Set the EIGEN_INCLUDE env var to override with your own
Eigen install instead.
"""
import os
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

_HERE = os.path.dirname(os.path.abspath(__file__))
EIGEN_INCLUDE = os.environ.get(
    "EIGEN_INCLUDE", os.path.join(_HERE, "jumplion", "_iq_native", "thirdparty", "eigen")
)

is_msvc = sys.platform == "win32"

if is_msvc:
    # cxx_std=17 below already adds /std:c++17; don't pass GCC-style flags here.
    extra_compile_args = ["/O2", "/openmp"]
    extra_link_args = []
else:
    extra_compile_args = ["-O3"]
    extra_link_args = []
    # OpenMP is optional -- the code works single-threaded without it.
    # Comment these two lines out if you don't want the OpenMP build dependency
    # (e.g. plain clang on macOS without libomp installed).
    if sys.platform != "darwin":
        extra_compile_args.append("-fopenmp")
        extra_link_args.append("-fopenmp")

ext_modules = [
    Pybind11Extension(
        "jumplion._iq_native",  # <-- rename to match where you place it
        ["jumplion/_iq_native/src/bindings.cpp"],
        include_dirs=[EIGEN_INCLUDE],
        cxx_std=17,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

if __name__ == "__main__":
    setup(
        name="jumplion",  # only needed if used standalone
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
    )
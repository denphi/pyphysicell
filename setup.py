"""Build the kernels into the wheel.

The library loads its kernels through ctypes, not as a Python extension module,
so what matters here is only that a shared library lands inside the package.
setuptools' Extension machinery is used for that because it already knows how
to drive MSVC, clang and gcc; the resulting file has no PyInit_ symbol and is
never imported, only dlopen'ed by `pyphysicell._kernels`.

OpenMP is deliberately *not* enabled in the wheels. The kernels carry their own
std::thread pool, so a wheel built this way needs no libomp beside it on any
platform -- which is the difference between `pip install` working and not.
"""
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

FLAGS = {
    "msvc": ["/O2", "/std:c++17", "/EHsc"],
    "unix": ["-O3", "-std=c++17", "-fvisibility=hidden"],
}


class BuildKernels(build_ext):
    def build_extensions(self):
        flags = FLAGS.get(self.compiler.compiler_type, FLAGS["unix"])
        for ext in self.extensions:
            ext.extra_compile_args = list(flags)
        super().build_extensions()


setup(
    cmdclass={"build_ext": BuildKernels},
    ext_modules=[Extension("pyphysicell._pckernels", ["python/pyphysicell/kernels.cpp"])],
)

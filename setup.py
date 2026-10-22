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
import os

from setuptools import Extension, setup
from wheel.bdist_wheel import bdist_wheel
from setuptools.command.build_ext import build_ext

FLAGS = {
    "msvc": ["/O2", "/std:c++17", "/EHsc"],
    "unix": ["-O3", "-std=c++17", "-fvisibility=hidden"],
}


class BuildKernels(build_ext):
    """Build the kernels, with two Windows-specific corrections.

    setuptools assumes every Extension is an importable module and adds
    /EXPORT:PyInit_<name> to the MSVC link line. These kernels have no
    PyInit_ -- they are a plain shared library loaded with ctypes -- so the
    link fails with an unresolved external. `get_export_symbols` returning
    nothing is the documented way out; __declspec(dllexport) in the source
    already exports what we call.

    That failure used to be swallowed so an sdist install could continue
    without a compiler. It should not be: a wheel whose kernels did not build
    is broken, and the silence cost a CI round trip in which a 32-bit MinGW
    fallback DLL reached 64-bit Python (WinError 193). Set
    PYPHYSICELL_OPTIONAL_KERNELS=1 to install anyway and compile on first
    import.
    """

    def get_export_symbols(self, ext):
        return []

    def get_ext_filename(self, name):
        # drop the cpython-3XX tag: the file is found by glob, not imported
        base = super().get_ext_filename(name)
        root, ext = os.path.splitext(base)
        return root.split(".")[0] + ext

    def build_extensions(self):
        flags = FLAGS.get(self.compiler.compiler_type, FLAGS["unix"])
        for ext in self.extensions:
            ext.extra_compile_args = list(flags)
        try:
            super().build_extensions()
        except Exception as exc:                                  # noqa: BLE001
            if not os.environ.get("PYPHYSICELL_OPTIONAL_KERNELS"):
                raise
            print(f"\n*** pyphysicell: could not prebuild the kernels ({exc}).\n"
                  f"*** Installing anyway; they will be compiled on first import.\n")


class PlainTag(bdist_wheel):
    """Tag the wheel py3-none-<platform>, not cp3XX-cp3XX-<platform>.

    The kernels call no CPython API -- they are a plain shared library that
    ctypes dlopens -- so one build serves every interpreter on a platform.
    That turns a 4-versions x 7-platforms matrix into 7 wheels and covers
    Python versions that did not exist when the wheel was built.
    """

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False          # platform-specific, but not ABI-specific

    def get_tag(self):
        _py, _abi, plat = super().get_tag()
        return "py3", "none", plat


setup(
    cmdclass={"build_ext": BuildKernels, "bdist_wheel": PlainTag},
    ext_modules=[Extension("pyphysicell._pckernels", ["python/pyphysicell/kernels.cpp"])],
)

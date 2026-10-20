"""Bind the C++ computation kernels.

A wheel ships the kernels already compiled, next to this file; there is nothing
to build and no compiler needed.  Installed from source, or run from a checkout,
they are compiled on first import into a cache directory keyed by the hash of
the source and the compiler flags -- still no build step for the user.

OpenMP is used when the toolchain provides it; otherwise the same kernels run
on a std::thread fan-out (Apple clang ships no OpenMP runtime, and the wheels
are built this way deliberately so they carry no libomp dependency).
PHYSICELL_THREADS caps the pool, which is what an ensemble wants.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import sys
import sysconfig
import tempfile
from ctypes import POINTER, c_char_p, c_double, c_int, c_long, c_ubyte, c_void_p
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
# the kernel source ships inside the package, so an installed copy can rebuild it
_SRC = next(p for p in (_HERE / "kernels.cpp", _HERE.parents[1] / "cpp" / "kernels.cpp")
            if p.exists())

_BASE_FLAGS = ["-O3", "-std=c++17", "-fPIC", "-shared", "-fvisibility=hidden"]
# Tried in order; the first that compiles wins.
_FLAG_SETS = [
    ("openmp-gcc", ["-fopenmp"]),
    ("openmp-clang", ["-Xpreprocessor", "-fopenmp", "-lomp"]),
    ("threads", []),
]


def _cache_dir() -> Path:
    env = os.environ.get("PHYSICELL_CACHE")
    base = Path(env) if env else Path(tempfile.gettempdir()) / "pyphysicell-cache"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _compilers() -> list[str]:
    cands = [os.environ.get("CXX"), sysconfig.get_config_var("CXX"), "g++", "clang++"]
    out, seen = [], set()
    for c in cands:
        if not c:
            continue
        c = c.split()[0]
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _prebuilt() -> Path | None:
    """The shared library a wheel ships, if this is an installed wheel."""
    for pat in ("_pckernels*.so", "_pckernels*.dylib", "_pckernels*.pyd", "_pckernels*.dll"):
        for found in sorted(_HERE.glob(pat)):
            return found
    return None


def _build() -> tuple[Path, str]:
    src = _SRC.read_bytes()
    suffix = {"darwin": ".dylib", "win32": ".dll"}.get(sys.platform, ".so")
    errors = []
    for tag, extra in _FLAG_SETS:
        for cxx in _compilers():
            key = hashlib.sha256(src + f"{cxx}|{tag}".encode()).hexdigest()[:16]
            lib = _cache_dir() / f"libpckernels-{key}{suffix}"
            if lib.exists():
                return lib, tag
            cmd = [cxx, *_BASE_FLAGS, *extra, str(_SRC), "-o", str(lib)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0 and lib.exists():
                return lib, tag
            errors.append(f"{cxx} [{tag}]: {proc.stderr.strip().splitlines()[-1:] or ''}")
    raise RuntimeError("could not build the kernels:\n  " + "\n  ".join(errors))


_D = POINTER(c_double)
_L = POINTER(c_long)
_U = POINTER(c_ubyte)


def _load():
    shipped = _prebuilt()
    if shipped is not None:
        lib_path, tag = shipped, "prebuilt"
    else:
        lib_path, tag = _build()
    lib = ctypes.CDLL(str(lib_path))

    lib.pc_has_openmp.restype = c_int
    lib.pc_max_threads.restype = c_int

    sigs = {
        "pc_thomas_sweep": [_D, c_long, c_long, c_long, c_long, c_int, _D, _D, _D],
        "pc_apply_dirichlet": [_D, c_long, _L, _D, _U, c_long],
        "pc_gradients": [_D, c_long, c_long, c_long, c_long, c_double, c_double, c_double, _D],
        "pc_secretion_uptake": [_D, c_long, _L, _D, _D, _D, _D, _D, c_long, c_double, c_double, _D],
        "pc_voxel_of": [_D, c_long, c_double, c_double, c_double, c_double, c_double, c_double,
                        c_long, c_long, c_long, _L],
        "pc_build_cell_list": [_D, c_long, c_double, c_double, c_double, c_double,
                               c_long, c_long, c_long, _L, _L],
        "pc_forces": [_D, _D, _D, _D, _D, _L, _D, c_long, _U, c_long,
                      c_double, c_double, c_double, c_double, c_long, c_long, c_long,
                      _L, _L, c_int, _D, _D, _L, _L, c_long],
        "pc_update_positions": [_D, _D, _D, _U, c_long, c_double, c_int],
        "pc_sample_at_cells": [_D, _D, c_long, _L, c_long, _D, _D],
        "pc_fibre_push_cells": [_D, _D, _D, _U, c_long, _D, _D, _D, _D, _D, c_long,
                                c_double, c_double, c_double, c_double, c_long, c_long, c_long,
                                _L, _L, c_int, _D, _D, _L],
        "pc_cells_push_fibres": [_D, _D, _D, _D, _D, c_long, _D, _D, _D, _D, _D, _U, _U, _L,
                                 c_long, c_double, c_double, c_double, c_double,
                                 c_long, c_long, c_long, _L, _L, c_int, _D, _D, _D],
    }
    for name, argtypes in sigs.items():
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = None
    return lib, lib_path, tag


_LIB, LIB_PATH, BUILD_TAG = _load()

HAS_OPENMP = bool(_LIB.pc_has_openmp())
MAX_THREADS = int(_LIB.pc_max_threads())
BACKEND = "OpenMP" if HAS_OPENMP else "std::thread"


def _d(a: np.ndarray):
    assert a.dtype == np.float64 and a.flags.c_contiguous, "expected C-contiguous float64"
    return a.ctypes.data_as(_D)


def _l(a: np.ndarray):
    assert a.dtype == np.int64 and a.flags.c_contiguous, "expected C-contiguous int64"
    return a.ctypes.data_as(_L)


def _u(a: np.ndarray):
    assert a.dtype == np.uint8 and a.flags.c_contiguous, "expected C-contiguous uint8"
    return a.ctypes.data_as(_U)


_NULL_D = ctypes.cast(None, _D)
_NULL_L = ctypes.cast(None, _L)


def thomas_sweep(rho, axis, denom, cc, c1):
    nz, ny, nx, ns = rho.shape
    _LIB.pc_thomas_sweep(_d(rho), nx, ny, nz, ns, int(axis), _d(denom), _d(cc), _d(c1))


def apply_dirichlet(rho, idx, val, mask):
    ns = rho.shape[-1]
    _LIB.pc_apply_dirichlet(_d(rho), ns, _l(idx), _d(val), _u(mask), idx.size)


def gradients(rho, dx, dy, dz, out):
    nz, ny, nx, ns = rho.shape
    _LIB.pc_gradients(_d(rho), nx, ny, nz, ns, dx, dy, dz, _d(out))


def secretion_uptake(rho, cell_voxel, cell_volume, S, U, T, net_export, voxel_volume, dt,
                     internalized=None):
    ns = rho.shape[-1]
    _LIB.pc_secretion_uptake(
        _d(rho), ns, _l(cell_voxel), _d(cell_volume), _d(S), _d(U), _d(T), _d(net_export),
        cell_voxel.size, voxel_volume, dt,
        _d(internalized) if internalized is not None else _NULL_D,
    )


def voxel_of(pos, origin, spacing, shape, out):
    (x0, y0, z0), (dx, dy, dz), (nx, ny, nz) = origin, spacing, shape
    _LIB.pc_voxel_of(_d(pos), pos.shape[0], x0, y0, z0, dx, dy, dz, nx, ny, nz, _l(out))


def build_cell_list(pos, origin, h, grid_shape, head, next_):
    x0, y0, z0 = origin
    gx, gy, gz = grid_shape
    _LIB.pc_build_cell_list(_d(pos), pos.shape[0], x0, y0, z0, h, gx, gy, gz, _l(head), _l(next_))


def forces(pos, radius, rel_max_adh, adhesion, repulsion, type_, affinity, movable,
           origin, h, grid_shape, head, next_, is2d, velocity, pressure, neighbors,
           neighbor_idx=None, cap=0):
    x0, y0, z0 = origin
    gx, gy, gz = grid_shape
    _LIB.pc_forces(_d(pos), _d(radius), _d(rel_max_adh), _d(adhesion), _d(repulsion), _l(type_),
                   _d(affinity), affinity.shape[1], _u(movable), pos.shape[0],
                   x0, y0, z0, h, gx, gy, gz, _l(head), _l(next_), int(is2d),
                   _d(velocity), _d(pressure), _l(neighbors),
                   _l(neighbor_idx) if neighbor_idx is not None else _NULL_L, int(cap))


def update_positions(pos, vel, prev_vel, movable, dt, is2d):
    _LIB.pc_update_positions(_d(pos), _d(vel), _d(prev_vel), _u(movable), pos.shape[0], dt, int(is2d))


def sample_at_cells(rho, grad, cell_voxel, out_rho, out_grad=None):
    ns = rho.shape[-1]
    _LIB.pc_sample_at_cells(_d(rho), _d(grad) if grad is not None else _NULL_D, ns,
                            _l(cell_voxel), cell_voxel.size, _d(out_rho),
                            _d(out_grad) if out_grad is not None else _NULL_D)


def fibre_push_cells(cpos, cradius, crepulsion, cmovable, fpos, fdir, fhalf, frad, frep,
                     origin, h, grid_shape, head, next_, is2d, velocity, pressure, contacts):
    x0, y0, z0 = origin
    gx, gy, gz = grid_shape
    _LIB.pc_fibre_push_cells(_d(cpos), _d(cradius), _d(crepulsion), _u(cmovable), cpos.shape[0],
                             _d(fpos), _d(fdir), _d(fhalf), _d(frad), _d(frep), fpos.shape[0],
                             x0, y0, z0, h, gx, gy, gz, _l(head), _l(next_), int(is2d),
                             _d(velocity), _d(pressure), _l(contacts))


def cells_push_fibres(cpos, cradius, crepulsion, cspeed, csticky, fpos, fdir, fhalf, frad, frep,
                      fpush, frotate, fcrosslinks, origin, h, grid_shape, head, next_, is2d,
                      fvelocity, fangle, fpressure):
    x0, y0, z0 = origin
    gx, gy, gz = grid_shape
    _LIB.pc_cells_push_fibres(_d(cpos), _d(cradius), _d(crepulsion), _d(cspeed), _d(csticky),
                              cpos.shape[0], _d(fpos), _d(fdir), _d(fhalf), _d(frad), _d(frep),
                              _u(fpush), _u(frotate), _l(fcrosslinks), fpos.shape[0],
                              x0, y0, z0, h, gx, gy, gz, _l(head), _l(next_), int(is2d),
                              _d(fvelocity), _d(fangle), _d(fpressure))

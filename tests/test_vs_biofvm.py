"""Golden test: the pyphysicell diffusion kernel against BioFVM itself.

Builds the reference oracle (tests/biofvm_reference.cpp) against the vendored
PhysiCell checkout, runs the same case through our kernel, and compares.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from pyphysicell.model import Domain, Model, Substrate  # noqa: E402
from pyphysicell.state import Microenvironment  # noqa: E402


def build_case(two_d=True):
    z = (-10.0, 10.0) if two_d else (-200.0, 200.0)
    m = Model(domain=Domain(-200, 200, -200, 200, z[0], z[1], 20, 20, 20, use_2D=two_d))
    ox = Substrate("oxygen", diffusion_coefficient=100000.0, decay_rate=0.1)
    db = Substrate("debris", diffusion_coefficient=1000.0, decay_rate=0.01)
    for face in ("xmin", "xmax"):
        ox.dirichlet_faces[face] = (True, 38.0)
    m.substrates = [ox, db]
    me = Microenvironment(m)
    x, y, zc = me.centers
    X = x[None, None, :]
    Y = y[None, :, None]
    Z = zc[:, None, None]
    me.density[..., 0] = 10.0 + 0.01 * X + 0.02 * Y + 0.003 * Z
    n = np.arange(me.nx * me.ny * me.nz).reshape(me.nz, me.ny, me.nx)
    me.density[..., 1] = np.where(n % 7 == 0, 5.0, 0.0)
    me.apply_dirichlet()
    return me


def run_reference(mode, nsteps, out):
    exe = ROOT / "tests" / "biofvm_ref"
    assert exe.exists(), "build tests/biofvm_ref first (see tests/README)"
    subprocess.run([str(exe), mode, str(nsteps), str(out)], check=True, capture_output=True)
    raw = np.fromfile(out, dtype=np.uint8)
    nv, nd = np.frombuffer(raw[:16], dtype=np.int64)
    vals = np.frombuffer(raw[16:], dtype=np.float64).reshape(int(nv), int(nd))
    return vals


def compare(mode, nsteps, tol=1e-12):
    me = build_case(two_d=(mode == "2d"))
    for _ in range(nsteps):
        me.step(0.01)
    out = Path(tempfile.gettempdir()) / f"pyphysicell-ref_{mode}.bin"
    ref = run_reference(mode, nsteps, out)
    ours = me.density.reshape(-1, me.ns)
    err = np.abs(ours - ref)
    rel = err / np.maximum(np.abs(ref), 1e-30)
    return ours, ref, err.max(), rel.max()


def test_2d_matches_biofvm():
    _, _, amax, rmax = compare("2d", 100)
    assert rmax < 1e-12, f"2D mismatch: max abs {amax:.3e}, max rel {rmax:.3e}"


def test_3d_matches_biofvm():
    _, _, amax, rmax = compare("3d", 50)
    assert rmax < 1e-12, f"3D mismatch: max abs {amax:.3e}, max rel {rmax:.3e}"


if __name__ == "__main__":
    for mode, steps in (("2d", 100), ("3d", 50)):
        ours, ref, amax, rmax = compare(mode, steps)
        print(f"{mode}: {ours.shape[0]} voxels x {ours.shape[1]} substrates, {steps} steps")
        print(f"     max |ours-BioFVM| = {amax:.3e}   max relative = {rmax:.3e}")
        print(f"     sample ours={ours[123]}  biofvm={ref[123]}")

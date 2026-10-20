"""Unit tests for the computation kernels: numerics against independent maths."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from pyphysicell import _kernels as K  # noqa: E402
from pyphysicell.model import Domain, Model, Substrate  # noqa: E402
from pyphysicell.state import Cells, Microenvironment  # noqa: E402


def test_pure_decay_matches_analytic():
    m = Model(domain=Domain(-100, 100, -100, 100, -10, 10, 20, 20, 20, True),
              substrates=[Substrate("s", diffusion_coefficient=0.0, decay_rate=0.1,
                                    initial_condition=1.0)])
    me = Microenvironment(m)
    for _ in range(1000):
        me.step(0.01)
    assert abs(me.density.mean() - np.exp(-1.0)) / np.exp(-1.0) < 1e-3


def test_diffusion_conserves_mass_without_decay():
    m = Model(domain=Domain(-200, 200, -200, 200, -10, 10, 20, 20, 20, True),
              substrates=[Substrate("s", diffusion_coefficient=1000.0, decay_rate=0.0)])
    me = Microenvironment(m)
    me.density[0, 10, 10, 0] = 100.0
    before = me.density.sum()
    for _ in range(500):
        me.step(0.01)
    assert abs(me.density.sum() - before) / before < 1e-12


def _reference_force(xi, xj, ri, rj, adh_i, adh_j, rep_i, rep_j, rmax_i, rmax_j, aff_ij, aff_ji):
    """PhysiCell's Cell::add_potentials, written out directly."""
    disp = xi - xj
    d = max(float(np.linalg.norm(disp)), 1e-5)
    R = ri + rj
    temp_r = 0.0
    press = 0.0
    if d <= R:
        temp_r = (1.0 - d / R) ** 2
        press = temp_r / 0.027288820670331
    temp_r *= np.sqrt(rep_i * rep_j)
    S = rmax_i * ri + rmax_j * rj
    if d < S:
        temp_a = (1.0 - d / S) ** 2 * np.sqrt((adh_i * aff_ij) * (adh_j * aff_ji))
        temp_r -= temp_a
    if abs(temp_r) < 1e-16:
        return np.zeros(3), press
    return (temp_r / d) * disp, press


def test_force_law_matches_physicell_formula():
    rng = np.random.default_rng(0)
    for _ in range(20):
        c = Cells(n_substrates=1, capacity=4)
        idx = c.add(2)
        sep = rng.uniform(2.0, 40.0)
        c.position[idx] = [[0.0, 0.0, 0.0], [sep, 0.0, 0.0]]
        c.radius[idx] = [8.4, 9.1]
        c.adhesion[idx] = [0.4, 0.8]
        c.repulsion[idx] = [10.0, 12.0]
        c.rel_max_adhesion[idx] = [1.25, 1.5]
        c.type_idx[idx] = [0, 1]
        c.movable[idx] = 1
        affinity = np.array([[1.0, 0.7], [0.7, 1.0]])

        h = 100.0
        head = np.full(8, -1, dtype=np.int64)
        nxt = np.full(2, -1, dtype=np.int64)
        pos = np.ascontiguousarray(c.position[:2])
        K.build_cell_list(pos, (-100.0, -100.0, -100.0), h, (2, 2, 2), head, nxt)
        vel = np.zeros((2, 3))
        press = np.zeros(2)
        nbrs = np.zeros(2, dtype=np.int64)
        K.forces(pos, np.ascontiguousarray(c.radius[:2]),
                 np.ascontiguousarray(c.rel_max_adhesion[:2]),
                 np.ascontiguousarray(c.adhesion[:2]), np.ascontiguousarray(c.repulsion[:2]),
                 np.ascontiguousarray(c.type_idx[:2]), affinity,
                 np.ascontiguousarray(c.movable[:2]), (-100.0, -100.0, -100.0), h, (2, 2, 2),
                 head, nxt, 1, vel, press, nbrs)

        expect, p0 = _reference_force(pos[0], pos[1], 8.4, 9.1, 0.4, 0.8, 10.0, 12.0,
                                      1.25, 1.5, 0.7, 0.7)
        assert np.allclose(vel[0], expect, rtol=0, atol=1e-13), (sep, vel[0], expect)
        assert abs(press[0] - p0) < 1e-13
        assert np.allclose(vel[1], -expect, rtol=0, atol=1e-13)  # symmetric pair


def test_adams_bashforth_matches_physicell():
    c = Cells(n_substrates=1, capacity=2)
    idx = c.add(1)
    c.position[idx] = [1.0, 2.0, 0.0]
    c.velocity[idx] = [0.5, -0.25, 0.0]
    c.prev_velocity[idx] = [0.1, 0.1, 0.0]
    c.movable[idx] = 1
    dt = 0.1
    expect = np.array([1.0, 2.0, 0.0]) + 1.5 * dt * np.array([0.5, -0.25, 0.0]) \
        - 0.5 * dt * np.array([0.1, 0.1, 0.0])
    K.update_positions(c.position[:1], c.velocity[:1], c.prev_velocity[:1],
                       np.ascontiguousarray(c.movable[:1]), dt, 1)
    assert np.allclose(c.position[0], expect, atol=1e-15)
    assert np.allclose(c.prev_velocity[0], [0.5, -0.25, 0.0])


def test_secretion_uptake_matches_biofvm_formula():
    m = Model(domain=Domain(-20, 20, -20, 20, -10, 10, 20, 20, 20, True),
              substrates=[Substrate("s", diffusion_coefficient=0.0, decay_rate=0.0,
                                    initial_condition=5.0)])
    me = Microenvironment(m)
    c = Cells(n_substrates=1, capacity=2)
    idx = c.add(1)
    c.position[idx] = [0.0, 0.0, 0.0]
    c.volume_total[idx] = 2494.0
    c.secretion_rate[idx, 0] = 0.3
    c.secretion_target[idx, 0] = 10.0
    c.uptake_rate[idx, 0] = 0.7
    K.voxel_of(c.position[:1], me.origin, (me.dx, me.dy, me.dz), (me.nx, me.ny, me.nz), c.voxel[:1])
    dt = 0.01
    rho0 = me.density.reshape(-1, 1)[c.voxel[0], 0]
    f = dt * 2494.0 / me.voxel_volume
    expect = (rho0 + f * 0.3 * 10.0) / (1.0 + f * (0.3 + 0.7))
    K.secretion_uptake(me.density, c.voxel[:1], np.ascontiguousarray(c.volume_total[:1]),
                       np.ascontiguousarray(c.secretion_rate[:1]),
                       np.ascontiguousarray(c.uptake_rate[:1]),
                       np.ascontiguousarray(c.secretion_target[:1]),
                       np.ascontiguousarray(c.net_export_rate[:1]),
                       me.voxel_volume, dt, np.ascontiguousarray(c.internalized[:1]))
    got = me.density.reshape(-1, 1)[c.voxel[0], 0]
    assert abs(got - expect) < 1e-15


def test_rule_hill_matches_physicell():
    from pyphysicell.model import Rule
    r = Rule("t", "oxygen", "increases", "cycle entry", max_response=0.0033,
             half_max=21.5, hill_power=4)
    s = np.array([0.0, 21.5, 1e6])
    out = r.response(s, base_value=0.0)
    assert abs(out[0] - 0.0) < 1e-15
    assert abs(out[1] - 0.0033 / 2) < 1e-15        # half-max gives half the response
    assert abs(out[2] - 0.0033) < 1e-9             # saturates at max_response

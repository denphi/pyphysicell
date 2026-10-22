"""PhysiMeSS behaviour: the parts the sample project's own config turns off.

physimess ships with fibre_pushing, fibre_rotation and fibre_degradation all
at 0.0, so running that project exercises none of them. These build the
minimal situation for each one instead.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import pyphysicell as pc                                            # noqa: E402
from pyphysicell import fibres as F                                 # noqa: E402


def _model(**custom):
    m = pc.Model(
        domain=pc.Domain(-200, 200, -200, 200, -10, 10, dx=20, dy=20, dz=20, use_2D=True),
        time=pc.Time(max_time=60.0),
        cell_types=[pc.CellType("worker")],
    )
    ct = m.cell_type("worker")
    ct.motility.enabled = True
    ct.motility.speed = 1.0
    ct.motility.persistence_time = 5.0
    ct.custom_data.update(F.PARAM_DEFAULTS)
    ct.custom_data.update(custom)
    return m


def _sim(m, cell_xy=(-30.0, 0.0), fibre_dir=(0.0, 1.0, 0.0)):
    sim = pc.Simulation(m, seed=0)
    sim.add_cells(np.array([[cell_xy[0], cell_xy[1], 0.0]]), "worker")
    sim.fibres = F.Fibres()
    sim.fibres.add([[0.0, 0.0, 0.0]], [fibre_dir], half_length=40.0, radius=2.0)
    return sim


def test_free_fibre_is_pushed_only_when_the_cell_says_so():
    """fibre_pushing lives on the cell, not the fibre (add_potentials_from_cell)."""
    moved = {}
    for flag in (0.0, 1.0):
        sim = _sim(_model(fibre_pushing=flag))
        sim.cells.position[0] = (-10.0, 0.0, 0.0)        # overlapping the fibre
        start = sim.fibres.position[0].copy()
        for _ in range(20):
            F.step(sim, 0.1)
        moved[flag] = float(np.linalg.norm(sim.fibres.position[0] - start))
    assert moved[0.0] == pytest.approx(0.0, abs=1e-12)
    assert moved[1.0] > 1e-6


def test_free_fibre_turns_when_rotation_is_on():
    turned = {}
    for flag in (0.0, 1.0):
        sim = _sim(_model(fibre_rotation=flag))
        sim.cells.position[0] = (-10.0, 5.0, 0.0)
        start = sim.fibres.direction[0].copy()
        for _ in range(20):
            F.step(sim, 0.1)
        turned[flag] = float(np.linalg.norm(sim.fibres.direction[0] - start))
    assert turned[0.0] == pytest.approx(0.0, abs=1e-12)
    assert turned[1.0] > 1e-6


def test_two_crosslinks_hold_a_fibre_still():
    """PhysiMeSS returns early at X_crosslink_count >= 2."""
    sim = _sim(_model(fibre_pushing=1.0, fibre_rotation=1.0))
    sim.cells.position[0] = (-10.0, 0.0, 0.0)
    sim.fibres.crosslinks[0] = 2
    start = sim.fibres.position[0].copy()
    direction = sim.fibres.direction[0].copy()
    for _ in range(20):
        F.step(sim, 0.1)
        sim.fibres.crosslinks[0] = 2                     # keep it pinned
    assert np.allclose(sim.fibres.position[0], start)
    assert np.allclose(sim.fibres.direction[0], direction)


def test_velocity_dependent_adhesion_changes_the_cell_path():
    """vel_adhesion pulls a cell along the fibre it is touching."""
    ends = {}
    for adh in (0.0, 2.0):
        sim = _sim(_model(vel_adhesion=adh, cell_velocity_max=5.0))
        sim.cells.position[0] = (-10.0, 0.0, 0.0)
        sim.cells.prev_velocity[0] = (0.0, 1.0, 0.0)     # already moving along it
        sim.cells.velocity[0] = 0.0
        F.step(sim, 0.1)
        ends[adh] = sim.cells.velocity[0].copy()
    assert ends[0.0][1] == pytest.approx(0.0, abs=1e-12)
    assert ends[2.0][1] > 1e-6                           # pulled along +y


def test_a_stuck_cell_degrades_the_fibre_it_pushes_into():
    m = _model(fibre_degradation=1.0, fibre_deg_rate=1.0, fibre_stuck=1.0)
    sim = _sim(m)
    sim.cells.position[0] = (-10.0, 0.0, 0.0)
    sim.cells.motility_vector[0] = (1.0, 0.0, 0.0)       # driving into the fibre
    sim.cells.prev_velocity[0] = 0.0                     # i.e. not moving: stuck
    assert sim.fibres.n == 1
    for _ in range(5):
        F.step(sim, 0.1)
    assert sim.fibres.n == 0, "a stuck cell at full degradation rate should cut it"


def test_degradation_stays_off_by_default():
    sim = _sim(_model())
    sim.cells.position[0] = (-10.0, 0.0, 0.0)
    sim.cells.motility_vector[0] = (1.0, 0.0, 0.0)
    sim.cells.prev_velocity[0] = 0.0
    for _ in range(20):
        F.step(sim, 0.1)
    assert sim.fibres.n == 1


def test_stuck_cell_gets_one_forced_step():
    sim = _sim(_model())
    sim.cells.prev_velocity[0] = 0.0
    for _ in range(F.STUCK_THRESHOLD):
        F._update_stuck(sim)
    assert sim._fibre_counters["stuck"][0] == F.STUCK_THRESHOLD
    F.forced_motility(sim, 0.1)
    assert sim._fibre_counters["stuck"][0] == 0
    F.forced_motility(sim, 0.1)
    assert sim._fibre_counters["unstuck"][0] == 0        # exactly one forced step


def test_placement_keeps_fibres_inside_the_domain():
    m = _model()
    ct = pc.CellType("ecm")
    ct.custom_data.update(F.PARAM_DEFAULTS)
    rng = np.random.default_rng(0)
    centres = rng.uniform(-200, 200, size=(300, 3))
    centres[:, 2] = 0.0
    kept, u, half, radius, dropped = F.place(m, ct, centres, rng, "ecm")
    ends = np.stack([kept - half[:, None] * u, kept + half[:, None] * u], axis=1)
    assert kept.shape[0] + dropped == 300
    assert (ends[:, :, 0] >= -200).all() and (ends[:, :, 0] <= 200).all()
    assert (ends[:, :, 1] >= -200).all() and (ends[:, :, 1] <= 200).all()
    assert half.mean() == pytest.approx(37.5, abs=1e-9)   # fibre_length / 2


def test_crosslink_geometry():
    class S:
        pass
    sim = S()
    sim.fibres = F.Fibres()
    sim.fibres.add([[0, 0, 0]], [[1, 0, 0]], 20.0, 2.0)
    sim.fibres.add([[0, 0, 0]], [[0, 1, 0]], 20.0, 2.0)   # crosses the first
    sim.fibres.add([[500, 500, 0]], [[1, 0, 0]], 20.0, 2.0)
    F.crosslink(sim)
    assert sim.fibres.view("crosslinks").tolist() == [1, 1, 0]
    assert np.allclose(sim.fibres.view("crosslink_point")[0], [0.0, 0.0, 0.0])

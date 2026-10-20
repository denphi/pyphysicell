"""Regressions for the defects found in review.

Each test fails against the code as it was before the fix.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PC = ROOT.parent / "PhysiCell"
sys.path.insert(0, str(ROOT / "python"))

import pyphysicell as pc  # noqa: E402
from pyphysicell import phenotype as ph  # noqa: E402
from pyphysicell.state import Cells  # noqa: E402


def test_phase_durations_close_the_ring():
    """A 4-duration cycle is a 4-phase ring, not a 5-phase dead end."""
    model = pc.load(PC / "sample_projects" / "template")
    ct = model.cell_types[0]
    ends = {j for (_i, j) in ct.cycle.transition_rates}
    starts = {i for (i, _j) in ct.cycle.transition_rates}
    assert 0 in ends, "the last phase must link back to phase 0"
    assert starts == ends, f"ring is not closed: starts={starts} ends={ends}"


def test_cells_keep_their_size_across_divisions():
    """Division halves the volume and the targets; the growth phase doubles the
    targets again, so a cycling population does not shrink away."""
    model = pc.load(PC / "sample_projects" / "rules_sample")
    ct = model.cell_type("malignant epithelial cell")
    sim = pc.Simulation(model, seed=1)
    sim.cells.remove(np.ones(sim.cells.n, dtype=bool))
    idx = sim.add_cells([[0.0, 0.0, 0.0]], "malignant epithelial cell")
    c = sim.cells
    c.cycle_entry_rate[idx] = 1e9          # force a division at every phenotype step
    for _ in range(40):
        divides = ph.advance_cycle(c, sim.rng, model.time.dt_phenotype, sim.ring_length,
                                   sim.cycle_fixed, sim.cycle_rates, sim.growth_phase)
        sim._divide(divides)
        ph.update_volume(c, model.time.dt_phenotype)
    v = c.view("volume_total")
    assert v.min() > 0.25 * ct.volume.total, f"cells shrank away: min volume {v.min():.0f}"
    assert v.max() < 4.0 * ct.volume.total, f"cells ballooned: max volume {v.max():.0f}"


def test_recycled_slots_are_cleared():
    c = Cells(n_substrates=2, capacity=4)
    first = c.add(2)
    c.elapsed_time_in_phase[first] = 123.0
    c.internalized[first] = 7.0
    c.prev_velocity[first] = 5.0
    mask = np.zeros(c.n, dtype=bool)
    mask[:] = True
    c.remove(mask)
    fresh = c.add(1)
    assert c.elapsed_time_in_phase[fresh] == 0.0
    assert np.all(c.internalized[fresh] == 0.0)
    assert np.all(c.prev_velocity[fresh] == 0.0)


def test_division_splits_internalized_substrates():
    model = pc.load(PC / "sample_projects" / "rules_sample")
    sim = pc.Simulation(model, seed=0)
    sim.cells.remove(np.ones(sim.cells.n, dtype=bool))
    idx = sim.add_cells([[0.0, 0.0, 0.0]], "malignant epithelial cell")
    sim.cells.internalized[idx, 0] = 10.0
    sim._divide(np.asarray(idx))
    total = sim.cells.internalized[: sim.cells.n, 0].sum()
    assert abs(total - 10.0) < 1e-12, f"internalized mass changed: {total}"


def test_rules_respect_apply_to_dead():
    """A rule flagged for dead cells still fires on a corpse; one that is not
    flagged must leave the behavior as start_death left it."""
    model = pc.load(PC / "sample_projects" / "rules_sample")
    sim = pc.Simulation(model, seed=0)
    c = sim.cells
    mal = np.flatnonzero(c.view("type_idx") == sim.type_index["malignant epithelial cell"])
    dead = mal[:10]
    ph.start_death(c, dead, 0, sim.death_params)   # apoptosis: zeroes cycle entry and secretion
    assert np.all(c.cycle_entry_rate[dead] == 0.0)
    sim._sample_fields()
    sim.ruleset.apply(c, sim.type_index, 0.0)

    # "oxygen increases cycle entry" is not flagged for dead cells (column 8 = 0)
    assert np.all(c.cycle_entry_rate[dead] == 0.0), "a live-only rule wrote to a dead cell"
    live = mal[10:]
    assert c.cycle_entry_rate[live].min() > 0.0, "the same rule must still drive live cells"

    # "apoptotic increases apoptotic debris secretion" IS flagged for dead cells
    debris = model.substrate_names.index("apoptotic debris")
    assert np.all(c.secretion_rate[dead, debris] > 0.0), \
        "a rule flagged for dead cells did not fire on an apoptotic cell"
    assert np.all(c.secretion_rate[live, debris] == 0.0)


def test_chained_runs_do_not_duplicate_a_time_point():
    model = pc.load(PC / "sample_projects" / "rules_sample")
    model.time.max_time = 1.0
    sim = pc.Simulation(model, seed=0)
    sim.run(until=0.5, record_every=0.25)
    sim.run(until=1.0, record_every=0.25)
    times = np.asarray(sim.scalars.times)
    assert len(times) == len(np.unique(np.round(times, 9))), f"duplicate times: {times}"


def test_simple_chemotaxis_is_applied():
    model = pc.load(PC / "sample_projects" / "biorobots")
    sim = pc.Simulation(model, seed=0)
    assert np.abs(sim.chemotaxis_sensitivity).sum() > 0, \
        "a project using <chemotaxis> ended up with no chemotactic sensitivity"
    assert sim.needs_gradients


def test_death_phase_rates_are_read():
    """Death phases given as rates must not fall back to the hardcoded defaults."""
    from pyphysicell.io_xml import read_xml
    import xml.etree.ElementTree as ET
    import tempfile

    src = PC / "sample_projects" / "rules_sample" / "config" / "PhysiCell_settings.xml"
    tree = ET.parse(src)
    root = tree.getroot()
    cd = root.find("cell_definitions/cell_definition")
    apop = cd.find("phenotype/death/model[@code='100']")
    for child in list(apop.findall("phase_durations")):
        apop.remove(child)
    rates = ET.SubElement(apop, "phase_transition_rates")
    r = ET.SubElement(rates, "rate")
    r.set("start_index", "0")
    r.set("fixed_duration", "true")
    r.text = "0.01"            # 100 minutes, not the 516 default
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "PhysiCell_settings.xml"
        tree.write(out)
        m = read_xml(out)
    assert abs(m.cell_types[0].deaths["apoptosis"].phase_durations[0] - 100.0) < 1e-9


def test_cells_csv_type_column_may_be_a_float():
    """mechano writes the type index as 1.000000000000000000e+00."""
    m = pc.load(PC / "sample_projects" / "mechano")
    assert len(m.initial_cells) > 0
    names = {n for *_xyz, n in m.initial_cells}
    assert names <= set(m.cell_type_names), f"unparsed type column: {names}"
    sim = pc.Simulation(m, seed=0)
    assert sim.cells.n == len(m.initial_cells)


def test_unknown_initial_cell_types_are_not_dropped_silently():
    import pytest

    m = pc.load(PC / "sample_projects" / "mechano")
    m.initial_cells = m.initial_cells[:5] + [(0.0, 0.0, 0.0, "not a cell type")]
    with pytest.raises(ValueError, match="not define"):
        pc.Simulation(m, seed=0)


@pytest.mark.parametrize("project", ["heterogeneity", "rules_sample", "interactions",
                                     "biorobots", "worm"])
def test_state_stays_finite(project):
    """A run must not produce NaN or runaway positions.

    This caught a real defect: when adhesive affinity became per cell, the
    binding still passed the array's first dimension as the type count, so the
    force kernel read far out of bounds and positions became NaN within one
    phenotype step.
    """
    import pyphysicell as pc

    model = pc.load(PC / "sample_projects" / project)
    sim = pc.Simulation(model, seed=0)
    if project in pc.ports.available():
        pc.ports.install(sim)
    elif not sim.cells.n:
        pc.initial.apply(sim)
    sim.run(until=30)
    d = model.domain
    pos = sim.cells.view("position")
    assert np.isfinite(pos).all(), "positions went non-finite"
    assert np.isfinite(sim.cells.view("velocity")).all()
    assert np.isfinite(sim.microenvironment.density).all(), "the field went non-finite"
    reach = 3 * max(d.x_max - d.x_min, d.y_max - d.y_min)
    assert np.abs(pos).max() < reach, "cells flew out of the domain"


def test_basement_membrane_confines_cells():
    """pyphysicell.membrane: PhysiCell leaves this to the project and never sets
    its own `contact with basement membrane` signal; here both work."""
    import pyphysicell as pc

    model = pc.load(PC / "sample_projects" / "heterogeneity")
    sim = pc.Simulation(model, seed=0)
    pc.ports.install(sim)
    sim.membrane = pc.membrane.Sphere(radius=120.0)
    sim.run(until=60)
    assert "contact with basement membrane" in sim.dictionary.signals
    r = np.linalg.norm(sim.cells.view("position"), axis=1)
    touching = sim.cells.view("contact_bm")
    assert touching.sum() > 0, "no cell ever reached the membrane"
    assert np.isfinite(r).all()


def test_ki67_advanced_divides_on_exit_from_the_premitotic_phase():
    """PhysiCell's Ki67 (advanced) divides leaving phase 1, not on the return to
    phase 0: the postmitotic phase comes *after* division.  A ring heuristic
    gets this one model wrong."""
    import pyphysicell as pc
    from pyphysicell import cycles, phenotype as ph

    assert cycles.division_phase(0, 3) == 1          # Ki67 advanced
    assert cycles.division_phase(6, 4) == 3          # flow cytometry separated
    assert cycles.division_phase(5, 1) == 0          # live

    model = pc.load(PC / "sample_projects" / "rules_sample")
    ct = model.cell_types[0]
    ct.cycle.code = 0                                 # make it Ki67 advanced
    ct.cycle.transition_rates = {(0, 1): 1e9, (1, 2): 1e9, (2, 0): 1e9}
    ct.cycle.fixed_duration = {(0, 1): False, (1, 2): False, (2, 0): False}
    model.initial_cells = [(0.0, 0.0, 0.0, ct.name)]
    sim = pc.Simulation(model, seed=0)
    c = sim.cells
    seen = []
    for _ in range(6):
        before = c.phase[0]
        divides = ph.advance_cycle(c, sim.rng, 6.0, sim.ring_length, sim.cycle_fixed,
                                   sim.cycle_rates, sim.growth_phase, sim.division_phase)
        if divides.size:
            seen.append(int(before))
        sim._divide(divides)
    assert seen, "the cell never divided"
    assert set(seen) == {1}, f"divided on exit from phases {set(seen)}, expected phase 1"


def test_cycle_falls_back_to_the_standard_rates():
    """A <cycle> with no rates in the XML still cycles: PhysiCell starts from
    the standard model."""
    import tempfile
    import xml.etree.ElementTree as ET

    import pyphysicell as pc

    src = PC / "sample_projects" / "rules_sample" / "config" / "PhysiCell_settings.xml"
    tree = ET.parse(src)
    cd = tree.getroot().find("cell_definitions/cell_definition")
    cycle = cd.find("phenotype/cycle")
    for child in list(cycle):
        cycle.remove(child)                            # strip every rate
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "PhysiCell_settings.xml"
        tree.write(out)
        model = pc.read_xml(out)
    rates = model.cell_types[0].cycle.transition_rates
    assert rates, "no rates were inherited from the standard model"
    assert any(r > 0 for r in rates.values())

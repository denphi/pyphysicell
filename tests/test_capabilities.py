"""The things that are only possible because Python owns the state.

Each of these is impossible, or means editing C++ and re-running from t=0, in
the XML-and-executable workflow.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PC = ROOT.parent / "PhysiCell"
sys.path.insert(0, str(ROOT / "python"))

import pyphysicell as pc  # noqa: E402

PROJECT = PC / "sample_projects" / "rules_sample"


def small():
    """rules_sample, trimmed so the tests are quick -- keeping every cell type
    (the file is grouped by type, so a plain head() would be macrophages only)."""
    m = pc.load(PROJECT)
    m.initial_cells = m.initial_cells[::11]
    return m


def test_two_simulations_coexist_in_one_process():
    a, b = pc.Simulation(small(), seed=0), pc.Simulation(small(), seed=1)
    a.run(until=6)
    assert b.t == 0.0 and a.t > 0.0, "the two simulations share state"
    b.run(until=12)
    assert b.t > a.t


def test_branch_is_an_independent_future():
    sim = pc.Simulation(small(), seed=0)
    sim.run(until=6)
    left, right = sim.branch(), sim.branch()

    # a perturbation in one branch must not be visible in the other, or in the parent
    right.microenvironment.density[...] = 0.0
    assert left.microenvironment.density.max() > 0.0
    assert sim.microenvironment.density.max() > 0.0

    # and the two futures must then diverge: ablate half of one branch
    doomed = np.zeros(right.cells.n, dtype=bool)
    doomed[::2] = True
    right.cells.remove(doomed)
    left.run(until=18)
    right.run(until=18)
    assert right.cells.n < left.cells.n
    assert abs(sim.t - 6.0) < 1e-6, "branching advanced the parent"
    assert sim.cells.n > right.cells.n, "the parent lost cells to its branch"


def test_checkpoint_round_trip(tmp_path):
    sim = pc.Simulation(small(), seed=0)
    sim.run(until=6)
    path = sim.save(tmp_path / "ckpt.pkl")
    back = pc.Simulation.load(path)
    assert back.t == sim.t and back.cells.n == sim.cells.n
    assert np.allclose(back.cells.positions, sim.cells.positions)
    back.run(until=12)
    assert back.t > sim.t


def test_hook_sees_arrays_and_writes_behaviors():
    sim = pc.Simulation(small(), seed=0)
    calls = []

    @sim.hook("phenotype", cell_type="M0 macrophage")
    def freeze_the_left(ctx):
        calls.append(ctx.n)
        ctx.behaviors["migration speed"] = np.where(ctx.position[:, 0] < 0, 0.0, 1.0)

    sim.run(until=18)
    assert calls, "the hook never ran"
    assert all(n > 1 for n in calls), "a vectorized hook must get the whole population"
    c = sim.cells
    mine = c.view("type_idx") == sim.type_index["M0 macrophage"]
    left = mine & (c.view("position")[:, 0] < 0)
    right = mine & (c.view("position")[:, 0] >= 0)
    assert c.view("motility_speed")[left].max() == 0.0
    assert c.view("motility_speed")[right].min() == 1.0


def test_hook_is_called_once_per_step_not_once_per_cell():
    sim = pc.Simulation(small(), seed=0)
    calls = []
    sim.hook("phenotype")(lambda ctx: calls.append(ctx.n))
    steps = 5
    sim.run(until=steps * sim.model.time.dt_phenotype)
    assert len(calls) <= steps + 1, f"{len(calls)} calls for {steps} phenotype steps"
    assert calls[0] == sim.cells.n or calls[0] > 1


def test_stop_when_ends_the_run_early():
    sim = pc.Simulation(small(), seed=0)
    sim.run(until=10_000, stop_when=lambda s: s.t >= 30)
    assert 30 <= sim.t < 10_000


def test_timers_fire_on_schedule():
    sim = pc.Simulation(small(), seed=0)
    fired = []
    sim.every(6.0, lambda s: fired.append(round(s.t, 3)))
    sim.run(until=30)
    assert len(fired) >= 4, fired


def test_sweep_runs_many_models_in_one_process():
    m = small()

    before = m.cell_type("malignant epithelial cell").motility.speed

    def change(model, value):
        model.cell_type("malignant epithelial cell").motility.speed = value

    df, runs = pc.sweep(m, change, [before + 5.0, before + 9.0], until=12, replicates=2)
    assert len(df) == 4 and len(runs) == 4
    assert df["seed"].nunique() == 4, "replicates must differ by seed"
    assert {"cells", "spread_um", "mean_pressure", "wall_s"} <= set(df.columns)
    assert m.cell_type("malignant epithelial cell").motility.speed == before, \
        "the sweep mutated the caller's model"
    # each run really used its own model
    speeds = {r.sim.model.cell_type("malignant epithelial cell").motility.speed for r in runs}
    assert speeds == {before + 5.0, before + 9.0}


def test_custom_data_is_per_cell_and_addressable():
    """PhysiCell exposes <custom_data> as custom:<name>, readable as a signal and
    writable as a behavior; heterogeneity's whole model is one such variable."""
    m = pc.load(PC / "sample_projects" / "heterogeneity")
    sim = pc.Simulation(m, seed=0)
    pc.initial.apply(sim)
    assert "custom:oncoprotein" in sim.dictionary.signals
    assert "custom:oncoprotein" in sim.dictionary.behaviors

    onco = sim.cells.custom_column("oncoprotein")
    assert np.allclose(onco, m.cell_types[0].custom_data["oncoprotein"])

    # heterogeneity is created in C++ by drawing per cell; here it is an array
    rng = np.random.default_rng(0)
    onco[:] = rng.normal(1.0, 0.25, onco.size)
    seen = []

    @sim.hook("phenotype")
    def proliferate_with_oncoprotein(ctx):
        seen.append(ctx.signals["custom:oncoprotein"].std())
        ctx.behaviors["cycle entry"] = 0.001 * np.clip(ctx.signals["custom:oncoprotein"], 0, None)

    sim.run(until=12)
    assert seen and seen[0] > 0.1, "the hook did not see the per-cell variation"
    assert sim.cells.view("cycle_entry_rate").std() > 0, "the behavior write did not land"


def test_conserved_custom_data_is_halved_at_division():
    m = pc.load(PC / "sample_projects" / "heterogeneity")
    ct = m.cell_types[0]
    ct.custom_data["payload"] = 8.0
    ct.custom_conserved["payload"] = True          # conserved: split between daughters
    ct.custom_data["mark"] = 3.0
    ct.custom_conserved["mark"] = False            # not conserved: inherited whole
    m.initial_cells = [(0.0, 0.0, 0.0, ct.name)]
    sim = pc.Simulation(m, seed=0)
    sim._divide(np.array([0]))
    assert sim.cells.n == 2
    assert np.allclose(sim.cells.custom_column("payload"), [4.0, 4.0])
    assert np.allclose(sim.cells.custom_column("mark"), [3.0, 3.0])


def test_every_per_cell_array_is_inherited_at_division():
    """A new per-cell array must not be forgotten by divide(); the field list is shared."""
    from pyphysicell.state import Cells

    m = small()
    sim = pc.Simulation(m, seed=0)
    c = sim.cells
    # asymmetric_division_prob is excluded on purpose: a non-zero value is meant to
    # change the daughter's type, which re-seeds every type-dependent array
    names = [n for n in (*Cells._VEC3, *Cells._SCALAR_F, *Cells._SCALAR_I, *Cells._SCALAR_B,
                         *Cells._ALL_FIELDS)
             if n not in ("ID", "position", "prev_velocity", "asymmetric_division_prob")]
    for name in names:                       # make every array distinctive
        arr = getattr(c, name)
        arr[0] = 7 if arr.dtype != np.uint8 else 1
    sim._divide(np.array([0]))
    child = sim.cells.n - 1
    for name in names:
        arr = getattr(sim.cells, name)
        if name in ("volume_total", "volume_fluid", "volume_nuclear", "volume_cytoplasmic",
                    "volume_nuclear_solid", "volume_cytoplasmic_solid", "radius",
                    "target_solid_nuclear", "target_solid_cytoplasmic", "rupture_volume",
                    "internalized", "elapsed_time_in_phase", "custom",
                    "asymmetric_division_prob"):
            continue                          # these are deliberately rescaled or reset
        assert np.all(arr[child] == arr[0]), f"{name} was not inherited by the daughter"


def test_interactions_run_phagocytosis_attack_and_damage():
    """standard_cell_cell_interactions: a macrophage eats, a T cell damages."""
    # at full density: contact is what drives these, so a thinned population
    # simply never touches
    m = pc.load(PROJECT)
    sim = pc.Simulation(m, seed=0)
    sim.run(until=30)
    c = sim.cells
    assert (c.view("damage") > 0).any(), "nothing was attacked"
    assert (c.view("total_attack_time") > 0).any()
    assert sim._deaths > 0


def test_transformation_changes_a_cell_type():
    m = pc.load(PROJECT)
    m.initial_cells = [(0.0, 0.0, 0.0, "M0 macrophage")] * 20
    m.cell_type("M0 macrophage").interactions.transformation_rates["M1 macrophage"] = 1.0
    sim = pc.Simulation(m, seed=0)
    sim.run(until=12)
    names = [m.cell_type_names[i] for i in sim.cells.view("type_idx")]
    assert "M1 macrophage" in names, "no cell transformed"


def test_asymmetric_division_sets_the_daughter_type():
    m = pc.load(PC / "sample_projects" / "asymmetric_division")
    sim = pc.Simulation(m, seed=0)
    pc.initial.apply(sim)
    sim.run(until=600, record_every=300)
    names = {m.cell_type_names[i] for i in sim.cells.view("type_idx")}
    assert "progenitor_1" in names, "a stem cell never produced a progenitor"


def test_every_hook_slot_fires():
    from pyphysicell.simulation import HOOK_SLOTS

    m = pc.load(PROJECT)
    m.initial_cells = m.initial_cells[::11]
    sim = pc.Simulation(m, seed=0)
    seen = {slot: 0 for slot in HOOK_SLOTS}
    for slot in HOOK_SLOTS:
        sim.hook(slot)(lambda ctx, slot=slot: seen.__setitem__(slot, seen[slot] + 1))
    # "contact" is over *attached* pairs, so give it something to act on
    sim.cells.attach(0, 1)
    sim.run(until=60)
    for slot in ("phenotype", "custom_rule", "touch", "step", "migration_bias", "contact"):
        assert seen[slot] > 0, f"the {slot!r} hook never fired"

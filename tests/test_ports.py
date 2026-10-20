"""Every ported project installs and runs, and does what its model is about."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PC = ROOT.parent / "PhysiCell"
sys.path.insert(0, str(ROOT / "python"))

import pyphysicell as pc  # noqa: E402

PORTS = pc.ports.available()


def load(name):
    for pat in ("sample_projects", "sample_projects_intracellular/*"):
        for d in PC.glob(f"{pat}/{name}"):
            return pc.load(d)
    raise FileNotFoundError(name)


@pytest.mark.parametrize("name", PORTS)
def test_port_installs_and_runs(name):
    model = load(name)
    if name in ("cancer_immune",):
        model.user_parameters["tumor_radius"] = 80.0      # keep it quick
    sim = pc.Simulation(model, seed=0)
    note = pc.ports.install(sim)
    assert note and sim.cells.n > 0, f"{name}: nothing was seeded"
    # a port installs hooks, or (physimess) a different data model: fibres
    assert sim._hooks or getattr(sim, "fibres", None), f"{name}: the port did nothing"
    sim.run(until=30, record_every=30)
    assert sim.t >= 30


def test_biorobots_workers_pick_up_cargo():
    sim = pc.Simulation(load("biorobots"), seed=0)
    pc.ports.install(sim)
    sim.run(until=120)
    assert (sim.cells.view("n_attached") > 0).any(), "no worker ever attached to cargo"


def test_heterogeneity_gives_every_cell_its_own_oncoprotein():
    sim = pc.Simulation(load("heterogeneity"), seed=0)
    pc.ports.install(sim)
    onco = sim.cells.custom_column("oncoprotein")
    assert onco.std() > 0.1, "the population is not heterogeneous"
    sim.run(until=60)
    rates = sim.cells.view("cycle_entry_rate")
    assert rates.std() > 0, "oncoprotein did not reach the cycle rate"


def test_pred_prey_predators_eat_and_starve():
    sim = pc.Simulation(load("pred_prey_farmer"), seed=0)
    pc.ports.install(sim)
    energy_before = sim.cells.custom_column("energy").copy()
    sim.run(until=120)
    assert sim._deaths > 0, "nothing was eaten or starved"
    assert not np.allclose(sim.cells.custom_column("energy")[: energy_before.size],
                           energy_before), "predator energy never changed"


def test_celltypes3_cross_regulation_differs_between_types():
    sim = pc.Simulation(load("celltypes3"), seed=0)
    pc.ports.install(sim)
    sim.run(until=60)
    c = sim.cells
    rates = {name: c.view("cycle_entry_rate")[c.view("type_idx") == i].mean()
             for name, i in sim.type_index.items() if (c.view("type_idx") == i).any()}
    assert len({round(v, 12) for v in rates.values()}) > 1, f"all types identical: {rates}"

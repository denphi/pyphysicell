"""Ports of the sample projects' `custom_modules/custom.cpp`, in Python.

Each module here reproduces one project's custom functions as hooks, and its
`setup_tissue` as an initial condition.  They exist to show that the custom C++
of the sample corpus -- about a thousand lines in total -- is ordinary
arithmetic over cell state, and needs no compiler once the state is arrays.

    sim = pc.Simulation(model)
    pc.ports.install(sim)        # finds the port for this model by name
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path


def available() -> list[str]:
    here = Path(__file__).parent
    return sorted(m.name for m in pkgutil.iter_modules([str(here)]) if not m.name.startswith("_"))


def get(name: str):
    if name not in available():
        raise KeyError(f"no port for {name!r}; have {available()}")
    return importlib.import_module(f"{__name__}.{name}")


def install(sim, name: str | None = None, seed_cells: bool = True) -> str:
    """Install the port for this simulation's model. Returns what it did."""
    name = name or sim.model.name
    mod = get(name)
    notes = []
    if seed_cells and sim.cells.n == 0:
        if hasattr(mod, "seed"):
            notes.append(mod.seed(sim))
        else:
            from .. import initial
            notes.append(initial.apply(sim))
    notes.append(mod.install(sim))
    return " | ".join(n for n in notes if n)

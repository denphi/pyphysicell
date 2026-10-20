"""Parameter sweeps and replicates, in one process.

PhysiCell keeps its state in process globals, so a sweep there is one
subprocess (and one output folder) per point.  Here a model is an object and a
simulation is an object, so N of them are just N objects: no files, no
executables, no recompilation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .initial import apply as _apply_initial
from .model import Model
from .simulation import Simulation


@dataclass
class Run:
    value: object
    replicate: int
    seed: int
    scalars: object
    sim: Simulation


def _summarise(sim, value, replicate, seed) -> dict:
    c = sim.cells
    pos = c.view("position")
    spread = (float(np.sqrt(((pos - pos.mean(axis=0)) ** 2).sum(axis=1).mean()))
              if c.n else float("nan"))
    row = {"value": value, "replicate": replicate, "seed": seed, "t": sim.t,
           "cells": int(c.n), "births": sim._births, "deaths": sim._deaths,
           "dead": int(np.count_nonzero(c.view("dead") == 1)),
           # observables that respond even when nothing divides
           "spread_um": round(spread, 2),
           "mean_pressure": round(float(c.view("pressure").mean()), 4) if c.n else float("nan"),
           "mean_neighbors": round(float(c.view("neighbors").mean()), 2) if c.n else float("nan"),
           "wall_s": round(sim.wall_time, 2)}
    for name, i in sim.type_index.items():
        row[f"n_{name}"] = int(np.count_nonzero((c.view("type_idx") == i) & (c.view("dead") == 0)))
    for s, name in enumerate(sim.model.substrate_names):
        row[f"mean_{name}"] = float(sim.microenvironment.density[..., s].mean())
    return row


def sweep(model: Model, change, values, until: float, replicates: int = 1,
          record_every: float | None = None, seed0: int = 0, hooks=None,
          progress: bool = False):
    """Run `model` once per value (and per replicate) and collect the results.

    `change(model, value)` edits a fresh copy of the model for each point.
    Returns (DataFrame of final metrics, list of Run).  Pass `hooks` as a list
    of (slot, cell_type, fn) to install the same Python hook in every run.
    """
    import pandas as pd

    rows, runs = [], []
    seed = seed0
    for value in values:
        for rep in range(replicates):
            m = copy.deepcopy(model)
            change(m, value)
            sim = Simulation(m, seed=seed)
            if not sim.cells.n:
                _apply_initial(sim)
            for slot, cell_type, fn in (hooks or []):
                sim.hook(slot, cell_type)(fn)
            scalars = sim.run(until=until, record_every=record_every or until / 6.0)
            rows.append(_summarise(sim, value, rep, seed))
            runs.append(Run(value, rep, seed, scalars, sim))
            if progress:
                print(f"  {value!r:>12}  rep {rep}  ->  {sim.cells.n:6d} cells "
                      f"({sim.wall_time:.1f} s)", flush=True)
            seed += 1
    return pd.DataFrame(rows), runs


def plot_sweep(df, y: str, ax=None, label: str = ""):
    """Mean +/- spread over replicates, against the swept value."""
    import matplotlib.pyplot as plt

    ax = ax or plt.subplots(figsize=(6, 4))[1]
    g = df.groupby("value")[y]
    mean, lo, hi = g.mean(), g.min(), g.max()
    x = np.asarray(mean.index, dtype=float)
    ax.plot(x, mean.values, "o-", lw=2, label=label or y)
    if df["replicate"].max() > 0:
        ax.fill_between(x, lo.values, hi.values, alpha=0.2)
    ax.set_xlabel("swept value")
    ax.set_ylabel(y)
    ax.grid(alpha=0.3)
    return ax

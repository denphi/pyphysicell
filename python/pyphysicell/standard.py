"""PhysiCell's standard non-default cell functions, as vectorized hooks.

These are the functions sample projects reach for in `custom.cpp` before
writing anything of their own, chiefly `update_cell_and_death_parameters_O2_based`.
Porting them once keeps the project ports short.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CellParameters:
    """PhysiCell's Cell_Parameters defaults (PhysiCell_cell.cpp)."""

    o2_hypoxic_threshold: float = 15.0
    o2_hypoxic_response: float = 8.0
    o2_hypoxic_saturation: float = 4.0
    o2_necrosis_threshold: float = 5.0
    o2_necrosis_max: float = 2.5
    o2_proliferation_threshold: float = 5.0
    o2_proliferation_saturation: float = 160.0
    o2_reference: float = 160.0
    max_necrosis_rate: float = 1.0 / (6.0 * 60.0)


def o2_based_phenotype(ctx, params: CellParameters | None = None, substrate: str = "oxygen"):
    """`update_cell_and_death_parameters_O2_based`.

    Cycle entry scales linearly with oxygen between the proliferation threshold
    and saturation; necrosis ramps from zero at the necrosis threshold to the
    maximum rate at `o2_necrosis_max`.
    """
    p = params or CellParameters()
    o2 = ctx.signals[substrate]
    live = ~ctx.dead

    span = p.o2_proliferation_saturation - p.o2_proliferation_threshold
    mult = np.clip((o2 - p.o2_proliferation_threshold) / span, 0.0, 1.0)
    base = ctx.base("cycle entry")
    ctx.behaviors["cycle entry"] = np.where(live, mult * base, 0.0)

    nspan = p.o2_necrosis_threshold - p.o2_necrosis_max
    nmult = np.clip((p.o2_necrosis_threshold - o2) / nspan, 0.0, 1.0)
    ctx.behaviors["necrosis"] = np.where(live, p.max_necrosis_rate * nmult, 0.0)
    return o2


def hill_response(s, half_max, hill_power):
    """PhysiCell's Hill_response_function, vectorized."""
    s = np.maximum(np.asarray(s, dtype=float), 0.0)
    t = np.power(s / half_max, hill_power)
    return t / (1.0 + t)


def linear_response(s, s_min, s_max):
    """linear_response_function: 0 below s_min, 1 above s_max."""
    return np.clip((np.asarray(s, float) - s_min) / max(s_max - s_min, 1e-15), 0.0, 1.0)


def decreasing_linear_response(s, s_min, s_max):
    """decreasing_linear_response_function: 1 below s_min, 0 above s_max."""
    return 1.0 - linear_response(s, s_min, s_max)

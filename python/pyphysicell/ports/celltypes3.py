"""Port of sample_projects/celltypes3/custom_modules/custom.cpp.

Three cell types, each promoting or inhibiting the others' cycling, death,
speed and signalling through a shared up/down Hill response.  The three C++
phenotype functions are identical bar their parameter prefix, so this is one
function parameterised by cell type.
"""

from __future__ import annotations

import numpy as np


def up_down_signal(base, maximum, effects, half_max, hill_power):
    """`up_down_signal::compute_effect_hill`, vectorized.

    effects: list of (values, "promotes" | "inhibits" | anything else).
    """
    up = None
    down = None
    for values, kind in effects:
        if kind == "promotes":
            up = values if up is None else up + values
        elif kind == "inhibits":
            down = values if down is None else down + values
    denom = half_max ** hill_power
    if up is None:
        UP = 0.0
    else:
        t = np.power(np.maximum(up, 0.0), hill_power)
        UP = t / (denom + t)
    if down is None:
        DOWN = 1.0
    else:
        t = np.power(np.maximum(down, 0.0), hill_power)
        DOWN = denom / (denom + t)
    return (base + (maximum - base) * UP) * DOWN


def install(sim) -> str:
    p = sim.model.user_parameters
    c = sim.cells
    names = sim.model.substrate_names
    half_max = float(p.get("half_max", 0.5))
    hill_power = float(p.get("hill_power", 4.0))

    def make(letter: str):
        sub = f"signal {letter}"
        n_sub = names.index(sub) if sub in names else None

        def phenotype(ctx):
            live = ~ctx.dead
            A, B, C = (ctx.signals[f"signal {x}"] for x in "ABC")
            R = ctx.signals["resource"]
            press = ctx.signals["pressure"]

            # necrosis ramps up as the resource runs out
            thresh = float(p.get(f"{letter}_necrosis_threshold", 0.1))
            base_nec = ctx.base("necrosis")
            ctx.behaviors["necrosis"] = np.where(R < thresh,
                                                 base_nec * (1.0 - R / max(thresh, 1e-15)), 0.0)

            def rule(kind, extra=()):
                return [(A, p.get(f"{letter}_{kind}_A", "neutral")),
                        (B, p.get(f"{letter}_{kind}_B", "neutral")),
                        (C, p.get(f"{letter}_{kind}_C", "neutral")), *extra]

            cycle = up_down_signal(float(p.get(f"{letter}_base_cycle", 0.0)) * R,
                                   float(p.get(f"{letter}_max_cycle", 0.0)),
                                   rule("cycle"), half_max, hill_power)
            cycle = np.where(press > float(p.get(f"{letter}_cycle_pressure_threshold", 1.0)),
                             0.0, cycle)
            ctx.behaviors["cycle entry"] = np.where(live, cycle, 0.0)

            # NB: the original passes C where R is meant in the death and speed
            # rules ("{letter}_death_R" applied to C); kept for fidelity
            death = up_down_signal(float(p.get(f"{letter}_base_death", 0.0)),
                                   float(p.get(f"{letter}_max_death", 0.0)),
                                   rule("death", [(C, p.get(f"{letter}_death_R", "neutral"))]),
                                   half_max, hill_power)
            death = np.where(press > float(p.get(f"{letter}_apoptosis_pressure_threshold", 1.0)),
                             10.0, death)
            ctx.behaviors["apoptosis"] = np.where(live, death, 0.0)

            speed = up_down_signal(float(p.get(f"{letter}_base_speed", 0.0)),
                                   float(p.get(f"{letter}_max_speed", 0.0)),
                                   rule("speed", [(C, p.get(f"{letter}_speed_R", "neutral"))]),
                                   half_max, hill_power)
            ctx.behaviors["migration speed"] = np.where(live, speed, 0.0)

            if n_sub is not None:
                secretion = up_down_signal(
                    float(p.get(f"{letter}_base_secretion", 0.0)),
                    float(p.get(f"{letter}_max_secretion", 0.0)),
                    rule("signal", [(R, p.get(f"{letter}_signal_R", "neutral"))]),
                    half_max, hill_power)
                c.secretion_rate[ctx.idx, n_sub] = np.where(live, secretion, 0.0)

            # the dead stop everything
            dead_idx = ctx.idx[~live]
            if dead_idx.size:
                c.secretion_rate[dead_idx] = 0.0
                c.uptake_rate[dead_idx] = 0.0
                c.is_motile[dead_idx] = 0

        return phenotype

    installed = []
    for letter in "ABC":
        if letter in sim.type_index:
            sim.hook("phenotype", cell_type=letter)(make(letter))
            installed.append(letter)
    return f"up/down Hill cross-regulation for cell types {', '.join(installed)}"

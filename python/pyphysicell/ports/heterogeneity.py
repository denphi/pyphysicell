"""Port of sample_projects/heterogeneity/custom_modules/custom.cpp.

The model: a tumour whose cells carry an inherited `oncoprotein` drawn from a
normal distribution, which multiplies their oxygen-driven proliferation.
"""

from __future__ import annotations

import numpy as np

from .. import initial
from ..standard import o2_based_phenotype


def seed(sim) -> str:
    """`setup_tissue`: random cells of every type, then a hex-packed tumour disc
    whose cells get oncoprotein ~ N(mean, sd), clipped."""
    m = sim.model
    p = m.user_parameters
    n_each = int(p.get("number_of_cells", 0))
    for name in m.cell_type_names:
        if n_each:
            pos = initial.uniform_random(m, n_each, name, sim.rng)
            sim.add_cells([[x, y, z] for x, y, z, _ in pos], name)

    disc = initial.hex_disc(m, float(p.get("tumor_radius", 250.0)), "cancer cell")
    idx = sim.add_cells([[x, y, z] for x, y, z, _ in disc], "cancer cell")

    onco = sim.cells.custom[:, sim.cells.custom_names.index("oncoprotein")]
    draw = sim.rng.normal(float(p.get("oncoprotein_mean", 1.0)),
                          float(p.get("oncoprotein_sd", 0.25)), idx.size)
    onco[idx] = np.clip(draw, float(p.get("oncoprotein_min", 0.0)),
                        float(p.get("oncoprotein_max", 2.0)))
    return (f"{len(disc)} tumour cells in a disc of radius "
            f"{p.get('tumor_radius', 250.0):g} um, oncoprotein mean "
            f"{onco[idx].mean():.3f} sd {onco[idx].std():.3f}")


def install(sim) -> str:
    @sim.hook("phenotype", cell_type="cancer cell")
    def tumor_cell_phenotype_with_oncoprotein(ctx):
        o2_based_phenotype(ctx)                       # the standard O2 response
        ctx.behaviors["cycle entry"] = (ctx.behaviors["cycle entry"]
                                        * ctx.signals["custom:oncoprotein"])
    return "oncoprotein-scaled oxygen-driven proliferation"

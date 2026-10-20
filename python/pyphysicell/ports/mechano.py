"""Port of sample_projects/mechano/custom_modules/custom.cpp."""

from __future__ import annotations

import numpy as np


def install(sim) -> str:
    @sim.hook("phenotype", cell_type="cancer")
    def cancer_phenotype_function(ctx):
        # proliferation stops under pressure, and everything dies after t = 10000
        base = ctx.base("cycle entry")
        ctx.behaviors["cycle entry"] = np.where(ctx.signals["pressure"] > 0.75, 0.0, base)
        if ctx.t > 10000:
            ctx.behaviors["apoptosis"] = 9e99
            # the original also sets "cell detachment rate"; attachments are not
            # implemented here, so that part has no effect either way
    return "pressure-arrested proliferation, apoptosis after t=10000 min"

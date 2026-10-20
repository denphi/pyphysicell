"""Port of sample_projects/biorobots/custom_modules/custom.cpp.

Worker cells chemotax toward the director signal, pick up cargo cells by
attaching to them, then switch to following the cargo signal and release their
load where the director signal is strong.
"""

from __future__ import annotations

import numpy as np


def install(sim) -> str:
    p = sim.model.user_parameters
    k_elastic = float(p.get("elastic_coefficient", 0.05))
    threshold = float(p.get("drop_threshold", 0.4))
    bias_attached = float(p.get("attached_worker_migration_bias", 1.0))
    bias_unattached = float(p.get("unattached_worker_migration_bias", 0.5))

    @sim.hook("phenotype", cell_type="cargo cell")
    def cargo_cell_rule(ctx):
        ctx.behaviors["cell-cell adhesion elastic constant"] = k_elastic

    @sim.hook("phenotype", cell_type="worker cell")
    def worker_cell_rule(ctx):
        c = sim.cells
        ctx.behaviors["cell-cell adhesion elastic constant"] = k_elastic
        drop = ctx.signals["director signal"] > threshold
        for k in np.flatnonzero(drop):
            i = int(ctx.idx[k])
            for other in c.attached[i][c.attached[i] >= 0]:
                o = int(other)
                c.custom[o, c.custom_names.index("receptor")] = 0.0
                c.cycle_entry_rate[o] = 0.0
            c.detach_all(i)
        ctx.behaviors["chemotactic response to director signal"] = np.where(drop, 0.0,
            ctx.behaviors["chemotactic response to director signal"])
        ctx.behaviors["chemotactic response to cargo signal"] = np.where(drop, 1.0,
            ctx.behaviors["chemotactic response to cargo signal"])
        ctx.behaviors["migration bias"] = np.where(drop, bias_unattached,
                                                   ctx.behaviors["migration bias"])

    @sim.hook("touch", cell_type="worker cell")
    def pick_up_cargo(ctx):
        """The original scans its mechanics voxel; here it is the touching pairs."""
        c = sim.cells
        receptor = c.custom_names.index("receptor")
        free = c.n_attached[ctx.i] == 0
        has_receptor = c.custom[ctx.j, receptor] > 0.5
        for k in np.flatnonzero(free & has_receptor):
            i, j = int(ctx.i[k]), int(ctx.j[k])
            if c.n_attached[i] != 0:
                continue
            c.attach(i, j)
            c.custom[j, receptor] = 0.0
            for sub in ("director signal", "cargo signal"):
                if sub in sim.model.substrate_names:
                    c.secretion_rate[j, sim.model.substrate_names.index(sub)] = 0.0
            si = sim.model.substrate_names
            if "director signal" in si:
                c.chemotaxis_sensitivity[i, si.index("director signal")] = 1.0
            if "cargo signal" in si:
                c.chemotaxis_sensitivity[i, si.index("cargo signal")] = 0.0
            c.migration_bias[i] = bias_attached
    return "workers attach cargo, follow the director signal, release on contact"

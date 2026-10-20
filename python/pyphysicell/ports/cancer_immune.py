"""Port of sample_projects/cancer_immune/custom_modules/cancer_immune_3D.cpp.

Tumour cells proliferate with oxygen and their inherited oncoprotein, and
secrete an immunostimulatory factor; immune cells chemotax up that factor,
attach to tumour cells, and kill them at a rate set by the oncoprotein.
"""

from __future__ import annotations

import numpy as np

from .. import initial
from ..standard import o2_based_phenotype


def seed(sim) -> str:
    """`setup_tissue`: a tumour spheroid with oncoprotein drawn per cell."""
    m = sim.model
    p = m.user_parameters
    blob = initial.hex_sphere(m, float(p.get("tumor_radius", 250.0)), "cancer cell")
    idx = sim.add_cells([[x, y, z] for x, y, z, _ in blob], "cancer cell")
    onco = sim.cells.custom[:, sim.cells.custom_names.index("oncoprotein")]
    onco[idx] = np.clip(sim.rng.normal(float(p.get("oncoprotein_mean", 1.0)),
                                       float(p.get("oncoprotein_sd", 0.25)), idx.size),
                        float(p.get("oncoprotein_min", 0.0)),
                        float(p.get("oncoprotein_max", 2.0)))
    return (f"{len(blob)} tumour cells in a spheroid of radius "
            f"{p.get('tumor_radius', 250.0):g} um, oncoprotein mean {onco[idx].mean():.3f}")


def install(sim) -> str:
    c = sim.cells
    names = sim.model.substrate_names
    n_factor = names.index("immunostimulatory factor") if \
        "immunostimulatory factor" in names else None
    onco_i = c.custom_names.index("oncoprotein") if "oncoprotein" in c.custom_names else None

    def cd(name, idx, default=0.0):
        return (c.custom[idx, c.custom_names.index(name)] if name in c.custom_names
                else np.full(len(idx), default))

    @sim.hook("phenotype", cell_type="cancer cell")
    def tumor_cell_phenotype_with_and_immune_stimulation(ctx):
        if n_factor is not None:
            c.secretion_rate[ctx.idx, n_factor] = 10.0
        o2_based_phenotype(ctx)
        if onco_i is not None:
            ctx.behaviors["cycle entry"] = (ctx.behaviors["cycle entry"]
                                            * c.custom[ctx.idx, onco_i])

    @sim.hook("custom_rule", cell_type="immune cell")
    def immune_cell_rule(ctx):
        live = ~ctx.dead
        attached = c.n_attached[ctx.idx] > 0
        lifetime = cd("attachment_lifetime", ctx.idx, 60.0)
        kill_rate = cd("kill_rate", ctx.idx, 1.0 / 15.0)
        sat = cd("oncoprotein_saturation", ctx.idx, 2.0)
        thresh = cd("oncoprotein_threshold", ctx.idx, 0.5)

        for k in np.flatnonzero(attached & live):
            i = int(ctx.idx[k])
            target = int(c.attached[i][c.attached[i] >= 0][0])
            detach = False
            if onco_i is not None and c.dead[target] == 0:
                onco = c.custom[target, onco_i]
                if onco >= thresh[k]:
                    scale = min((onco - thresh[k]) / max(sat[k] - thresh[k], 1e-15), 1.0)
                    if ctx.rng.random() < kill_rate[k] * scale * ctx.dt:
                        from .. import phenotype as ph
                        ph.start_death(c, np.array([target]), 0, sim.death_params)
                        detach = True
            if ctx.rng.random() < ctx.dt / (lifetime[k] + 1e-15):
                detach = True
            if detach:
                c.detach(i, target)
                c.is_motile[i] = 1
        c.is_motile[ctx.idx[live & ~attached]] = 1
        c.is_motile[ctx.idx[live & attached]] = 0

    @sim.hook("touch", cell_type="immune cell")
    def immune_cell_check_neighbors_for_attachment(ctx):
        if onco_i is None:
            return
        free = c.n_attached[ctx.i] == 0
        tumor = c.type_idx[ctx.j] == sim.type_index["cancer cell"]
        worth_it = c.custom[ctx.j, onco_i] > cd("oncoprotein_threshold", ctx.i, 0.5)
        alive = c.dead[ctx.j] == 0
        for k in np.flatnonzero(free & tumor & worth_it & alive):
            i, j = int(ctx.i[k]), int(ctx.j[k])
            if c.n_attached[i] == 0 and c.attach(i, j):
                c.is_motile[i] = 0

    @sim.hook("contact", cell_type="immune cell")
    def adhesion_contact_function(ctx):
        """Break the bond when it is stretched past the detachment distance."""
        disp = c.position[ctx.j] - c.position[ctx.i]
        limit = c.radius[ctx.i] * c.rel_max_adhesion[ctx.i]
        too_far = (disp ** 2).sum(axis=1) > limit ** 2
        for k in np.flatnonzero(too_far):
            c.detach(int(ctx.i[k]), int(ctx.j[k]))
    return "oncoprotein-driven tumour growth; immune cells attach and kill"

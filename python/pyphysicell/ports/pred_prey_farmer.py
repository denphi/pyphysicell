"""Port of sample_projects/pred_prey_farmer/custom_modules/custom.cpp.

Farmers grow food, prey eat food and breed, predators eat prey and starve.
Every cell steers by a weighted sum of three chemical gradients.
"""

from __future__ import annotations

import numpy as np


def _wrap(sim):
    """wrap_boundaries: the domain is periodic in this model."""
    d, c = sim.model.domain, sim.cells
    n = c.n
    pos = c.position[:n]
    for axis, (lo, hi) in enumerate(((d.x_min, d.x_max), (d.y_min, d.y_max))):
        span = hi - lo
        out_lo = pos[:, axis] < lo
        out_hi = pos[:, axis] > hi
        pos[out_lo, axis] += span
        pos[out_hi, axis] -= span


def install(sim) -> str:
    c = sim.cells
    names = sim.model.substrate_names

    @sim.hook("migration_bias")
    def weighted_motility_function(ctx):
        """Steer by prey_weight*grad(prey) + predator_weight*grad(predator)
        + food_weight*grad(food), each weight per cell."""
        grads = getattr(sim, "_gradients", None)
        if grads is None:
            return
        v = np.zeros((ctx.n, 3))
        for sub, weight in (("prey signal", "prey_weight"),
                            ("predator signal", "predator_weight"),
                            ("food", "food_weight")):
            if sub in names and weight in c.custom_names:
                w = c.custom[ctx.idx, c.custom_names.index(weight)]
                v += w[:, None] * grads[ctx.idx, names.index(sub)]
        ctx.bias_direction = v

    @sim.hook("phenotype", cell_type="prey")
    def prey_phenotype_function(ctx):
        food = ctx.signals["food"]
        starving = food < 0.1
        ctx.kill(starving, model="necrosis")
        ctx.behaviors["cycle entry"] = np.where(starving, 0.0,
                                                ctx.base("cycle entry") * (food - 0.1) / 0.9)

    @sim.hook("phenotype", cell_type="predator")
    def predator_phenotype_function(ctx):
        energy = c.custom_names.index("energy")
        c.custom[ctx.idx, energy] /= (1.0 + ctx.dt * 0.00025)
        ctx.kill(c.custom[ctx.idx, energy] < 0.1, model="necrosis")

    @sim.hook("touch", cell_type="predator")
    def eat_the_prey(ctx):
        prey = sim.type_index["prey"]
        energy = c.custom_names.index("energy")
        d = np.linalg.norm(c.position[ctx.j] - c.position[ctx.i], axis=1)
        reach = c.radius[ctx.i] + c.radius[ctx.j] + 2.0
        catch = (c.type_idx[ctx.j] == prey) & (d <= reach) & (c.dead[ctx.j] == 0)
        for k in np.flatnonzero(catch):
            i, j = int(ctx.i[k]), int(ctx.j[k])
            if c.dead[j] or c.volume_total[j] < 1e-15:
                continue
            from ..interactions import _ingest
            _ingest(sim, np.array([i]), np.array([j]))
            c.custom[i, energy] += 100.0

    @sim.hook("custom_rule")
    def wrap_boundaries(ctx):
        _wrap(sim)
    return "farmers feed prey, predators eat prey and starve; gradient-weighted motility"

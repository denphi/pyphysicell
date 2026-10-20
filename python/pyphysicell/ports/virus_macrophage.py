"""Port of sample_projects/virus_macrophage/custom_modules/custom.cpp.

Epithelial cells replicate internalised virus until they burst; macrophages
digest virus, eat visibly infected cells, and keep away from the domain edge.
"""

from __future__ import annotations

import numpy as np


def install(sim) -> str:
    c = sim.cells
    names = sim.model.substrate_names
    n_virus = names.index("virus") if "virus" in names else None
    n_ifn = names.index("interferon") if "interferon" in names else None

    def cd(name, idx):
        return c.custom[idx, c.custom_names.index(name)]

    @sim.hook("phenotype", cell_type="epithelial cell")
    def epithelial_function(ctx):
        virus = c.internalized[ctx.idx, n_virus]
        burst = virus >= cd("burst_virion_count", ctx.idx)
        ctx.kill(burst)                                   # lyse_cell
        replicating = (~burst) & (virus >= cd("min_virion_count", ctx.idx))
        c.internalized[ctx.idx[replicating], n_virus] += (
            cd("viral_replication_rate", ctx.idx)[replicating] * ctx.dt)
        if n_ifn is not None:
            secreting = virus >= cd("virion_threshold_for_interferon", ctx.idx)
            rate = np.where(secreting, cd("max_interferon_secretion_rate", ctx.idx),
                            c.secretion_rate[ctx.idx, n_ifn])
            c.secretion_rate[ctx.idx, n_ifn] = rate

    @sim.hook("phenotype", cell_type="macrophage")
    def digest_virus(ctx):
        c.internalized[ctx.idx, n_virus] /= (1.0 + ctx.dt * cd("virus_digestion_rate", ctx.idx))

    @sim.hook("touch", cell_type="macrophage")
    def eat_infected_cells(ctx):
        from ..interactions import _ingest

        mine = sim.type_index["macrophage"]
        other_type = c.type_idx[ctx.j] != mine
        d = np.linalg.norm(c.position[ctx.j] - c.position[ctx.i], axis=1)
        reach = 1.1 * (c.radius[ctx.i] + c.radius[ctx.j])
        infected = c.internalized[ctx.j, n_virus] > cd("min_virion_detection_threshold", ctx.i)
        hit = other_type & infected & (d < reach) & (c.volume_total[ctx.j] > 1e-15)
        if hit.any():
            eater, eaten = ctx.i[hit], ctx.j[hit]
            _, first = np.unique(eaten, return_index=True)
            _ingest(sim, eater[first], eaten[first])

    @sim.hook("custom_rule", cell_type="macrophage")
    def avoid_boundaries(ctx):
        d = sim.model.domain
        zone, speed = 25.0, -0.5
        pos = c.position[ctx.idx]
        near = ((pos[:, 0] < d.x_min + zone) | (pos[:, 0] > d.x_max - zone)
                | (pos[:, 1] < d.y_min + zone) | (pos[:, 1] > d.y_max - zone))
        if not d.use_2D:
            near |= (pos[:, 2] < d.z_min + zone) | (pos[:, 2] > d.z_max - zone)
        if near.any():
            c.velocity[ctx.idx[near]] = speed * pos[near]      # head back toward the origin
    return "viral replication and burst, macrophage phagocytosis of infected cells"

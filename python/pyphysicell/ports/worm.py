"""Port of sample_projects/worm/custom_modules/custom.cpp.

Cells attach into a chain and pass a `head` quantity along it; whichever end
holds the most head leads, the other trails, and the middle follows.
"""

from __future__ import annotations

import numpy as np


def install(sim) -> str:
    c = sim.cells
    p = sim.model.user_parameters
    names = sim.model.substrate_names
    n_signal = names.index("signal") if "signal" in names else 0
    head_i = c.custom_names.index("head") if "head" in c.custom_names else None

    def par(key, default):
        return float(p.get(key, default))

    roles = {
        "head": (par("head_migration_direction", 1.0), par("head_migration_speed", 1.0),
                 par("head_migration_bias", 0.5), par("head_migration_persistence", 1.0)),
        "tail": (par("tail_migration_direction", -1.0), par("tail_migration_speed", 1.0),
                 par("tail_migration_bias", 0.5), par("tail_migration_persistence", 1.0)),
        "middle": (par("middle_migration_direction", 1.0), par("middle_migration_speed", 1.0),
                   par("middle_migration_bias", 0.5), par("middle_migration_persistence", 1.0)),
    }

    @sim.hook("touch")
    def attach_into_a_chain(ctx):
        max_att = c.custom[:, c.custom_names.index("max_attachments")] \
            if "max_attachments" in c.custom_names else np.full(c.n, 2.0)
        for k in range(len(ctx.i)):
            i, j = int(ctx.i[k]), int(ctx.j[k])
            if c.n_attached[i] < max_att[i] and c.n_attached[j] < max_att[j]:
                c.attach(i, j)

    @sim.hook("custom_rule")
    def custom_function(ctx):
        n_att = c.n_attached[ctx.idx]
        if head_i is None:
            return
        head = c.custom[ctx.idx, head_i]
        # one attachment: the end with more head leads, the other trails
        lead = np.zeros(ctx.n, dtype=bool)
        for k in np.flatnonzero(n_att == 1):
            i = int(ctx.idx[k])
            other = int(c.attached[i][c.attached[i] >= 0][0])
            lead[k] = head[k] > c.custom[other, head_i]
        # the original resets head to head_initial for a cell with one attachment
        if "head_initial" in c.custom_names:
            ends = ctx.idx[n_att == 1]
            c.custom[ends, head_i] = c.custom[ends, c.custom_names.index("head_initial")]
        role = np.where(n_att > 1, "middle", np.where(lead, "head", "tail"))
        for name, (direction, speed, bias, persistence) in roles.items():
            sel = ctx.idx[role == name]
            if sel.size == 0:
                continue
            c.motility_speed[sel] = speed
            c.migration_bias[sel] = bias
            c.persistence_time[sel] = max(persistence, 1e-12)
            c.chemotaxis_sensitivity[sel, n_signal] = direction
            c.secretion_rate[sel, n_signal] = 1.0 if name == "middle" else 100.0
        c.is_motile[ctx.idx] = 1

    @sim.hook("contact")
    def transfer_head_along_the_chain(ctx):
        if head_i is None:
            return
        rate = (c.custom[ctx.i, c.custom_names.index("transfer_rate")]
                if "transfer_rate" in c.custom_names else 0.0)
        mine, theirs = c.custom[ctx.i, head_i], c.custom[ctx.j, head_i]
        move = np.where(mine > theirs, ctx.dt * rate * (mine - theirs), 0.0)
        np.add.at(c.custom[:, head_i], ctx.i, -move)
        np.add.at(c.custom[:, head_i], ctx.j, move)
    return "cells chain up and pass 'head' along; ends lead and trail"

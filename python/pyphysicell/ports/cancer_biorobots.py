"""Port of sample_projects/cancer_biorobots/custom_modules/custom.cpp.

Worker robots collect cargo cells, carry them down the oxygen gradient into
the hypoxic core, and release them there; the cargo then releases a drug that
damages tumour cells until they die.
"""

from __future__ import annotations

import numpy as np

from .. import initial
from ..standard import o2_based_phenotype


def seed(sim) -> str:
    """`setup_tissue` + `introduce_biorobots`: a tumour disc, then robots
    injected in a strip at the right-hand edge."""
    m = sim.model
    p = m.user_parameters
    disc = initial.hex_disc(m, float(p.get("tumor_radius", 200.0)), "cancer cell")
    sim.add_cells([[x, y, z] for x, y, z, _ in disc], "cancer cell")

    n_inject = int(p.get("number_of_injected_cells", 500))
    worker_fraction = float(p.get("worker_fraction", 0.10))
    d = m.domain
    xs = (d.x_max - 150.0) + 100.0 * sim.rng.random(n_inject)
    ys = (d.y_min + 50.0) + (d.y_max - d.y_min - 100.0) * sim.rng.random(n_inject)
    is_worker = sim.rng.random(n_inject) <= worker_fraction
    for name, mask in (("worker cell", is_worker), ("cargo cell", ~is_worker)):
        if mask.any():
            sim.add_cells([[x, y, 0.0] for x, y in zip(xs[mask], ys[mask])], name)
    return (f"{len(disc)} tumour cells, {int(is_worker.sum())} workers and "
            f"{int((~is_worker).sum())} cargo injected at the right edge")


def install(sim) -> str:
    c = sim.cells
    names = sim.model.substrate_names

    def sub(name):
        return names.index(name) if name in names else None

    n_chemo, n_drug = sub("chemoattractant"), sub("therapeutic")
    n_o2 = sub("oxygen")

    def cd(name, idx, default=0.0):
        return (c.custom[idx, c.custom_names.index(name)] if name in c.custom_names
                else np.full(len(idx), default))

    @sim.hook("phenotype", cell_type="cancer cell")
    def tumor_cell_phenotype_with_therapy(ctx):
        o2_based_phenotype(ctx)
        drug = ctx.signals["therapeutic"] if n_drug is not None else np.zeros(ctx.n)
        damage_rate = cd("damage_rate", ctx.idx)
        repair_rate = cd("repair_rate", ctx.idx)
        death_rate = cd("drug_death_rate", ctx.idx)
        max_damage = damage_rate / (1e-16 + repair_rate)

        damage = c.damage[ctx.idx]
        damage = (damage + ctx.dt * drug * damage_rate) / (1.0 + ctx.dt * repair_rate)
        c.damage[ctx.idx] = damage
        p_die = ctx.dt * damage * death_rate / np.maximum(max_damage, 1e-16)
        killed = (ctx.rng.random(ctx.n) <= p_die) & ~ctx.dead
        ctx.behaviors["apoptosis"] = np.where(killed, 9e99, ctx.behaviors["apoptosis"])

    @sim.hook("phenotype", cell_type="cargo cell")
    def cargo_cell_phenotype_rule(ctx):
        receptor = cd("receptor", ctx.idx)
        attached = c.n_attached[ctx.idx] > 0
        o2 = ctx.signals["oxygen"] if n_o2 is not None else np.zeros(ctx.n)
        drop_above = cd("cargo_release_o2_threshold", ctx.idx)

        # unattached: advertise if it still has a receptor, otherwise leak drug
        free_signalling = (~attached) & (receptor > 0.1)
        free_releasing = (~attached) & (receptor <= 0.1)
        # attached and in the hypoxic core: release the drug and let go
        releasing = attached & (o2 <= drop_above)
        holding = attached & (o2 > drop_above)

        if n_chemo is not None:
            c.secretion_rate[ctx.idx, n_chemo] = np.where(free_signalling, 10.0, 0.0)
        if n_drug is not None:
            c.secretion_rate[ctx.idx, n_drug] = np.where(free_releasing | releasing, 10.0, 0.0)
        if "receptor" in c.custom_names:
            r = c.custom_names.index("receptor")
            c.custom[ctx.idx[releasing | holding], r] = 0.0
        for k in np.flatnonzero(releasing):
            c.detach_all(int(ctx.idx[k]))

    @sim.hook("custom_rule", cell_type="cargo cell")
    def cargo_cell_rule(ctx):
        carried = c.n_attached[ctx.idx] > 0
        c.motility_speed[ctx.idx[carried]] = 0.0

    @sim.hook("custom_rule", cell_type="worker cell")
    def worker_cell_rule(ctx):
        attached = c.n_attached[ctx.idx] > 0
        detection = cd("motility_shutdown_detection_threshold", ctx.idx)
        bias_att = cd("attached_worker_migration_bias", ctx.idx, 1.0)
        bias_free = cd("unattached_worker_migration_bias", ctx.idx, 0.5)
        chemo = ctx.signals["chemoattractant"] if n_chemo is not None else np.zeros(ctx.n)

        c.migration_bias[ctx.idx] = np.where(attached, bias_att, bias_free)
        if n_o2 is not None:
            c.chemotaxis_sensitivity[ctx.idx, n_o2] = np.where(attached, -1.0, 0.0)
        if n_chemo is not None:
            c.chemotaxis_sensitivity[ctx.idx, n_chemo] = np.where(attached, 0.0, 1.0)
        quiet = (~attached) & (chemo < detection)
        c.motility_speed[ctx.idx[quiet]] = 0.0

    @sim.hook("touch", cell_type="worker cell")
    def pick_up_cargo(ctx):
        if "receptor" not in c.custom_names:
            return
        r = c.custom_names.index("receptor")
        free = c.n_attached[ctx.i] == 0
        wanted = c.custom[ctx.j, r] > 0.5
        for k in np.flatnonzero(free & wanted):
            i, j = int(ctx.i[k]), int(ctx.j[k])
            if c.n_attached[i] == 0:
                c.attach(i, j)
    return "workers ferry drug-loaded cargo into the hypoxic core; drug damages the tumour"

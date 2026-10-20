"""Port of sample_projects/interactions/custom_modules/custom.cpp.

Seven cell types in one ecology: bacteria consume resource and make toxin and
quorum factor, stem cells differentiate on contact, differentiated cells are
poisoned by toxin, and macrophages, T cells and neutrophils follow debris and
pro-inflammatory signal.  Every response is a Hill or linear function of a
local substrate or a neighbour count, which is why this ports almost line for
line into array arithmetic.
"""

from __future__ import annotations

import numpy as np

from ..standard import decreasing_linear_response, hill_response, linear_response

HILL = 1.5


def install(sim) -> str:
    c = sim.cells
    names = sim.model.substrate_names
    m = sim.model

    def n_of(sub):
        return names.index(sub) if sub in names else None

    nR, nTox, nDebris = n_of("resource"), n_of("toxin"), n_of("debris")
    nQuorum, nPIF = n_of("quorum"), n_of("pro-inflammatory")

    def cd(ct_name, key, default):
        return float(m.cell_type(ct_name).custom_data.get(key, default))

    def dead_leaks_debris(ctx):
        """Every type in this model exports its own volume as debris when dead."""
        dead = ctx.idx[ctx.dead]
        if dead.size and nDebris is not None:
            c.net_export_rate[dead, nDebris] = c.volume_total[dead]
            c.secretion_rate[dead] = 0.0
        return ctx.idx[~ctx.dead]

    @sim.hook("phenotype", cell_type="bacteria")
    def bacteria_phenotype(ctx):
        live = dead_leaks_debris(ctx)
        if live.size == 0:
            return
        R = c.sampled[live, nR]
        Q = c.sampled[live, nQuorum] if nQuorum is not None else 0.0
        base_cycle = m.cell_type("bacteria").cycle.entry_rate
        c.cycle_entry_rate[live] = (base_cycle * 10.0) * linear_response(
            R, cd("bacteria", "cycling_entry_threshold_resource", 0.15), 1.0)
        c.necrosis_rate[live] = 0.0028 * decreasing_linear_response(
            R, cd("bacteria", "necrosis_saturation_resource", 0.075),
            cd("bacteria", "necrosis_threshold_resource", 0.15))
        base_speed = m.cell_type("bacteria").motility.speed
        c.motility_speed[live] = base_speed + (0.0 - base_speed) * hill_response(
            R, cd("bacteria", "migration_speed_halfmax", 0.25), HILL)
        c.migration_bias[live] = base_speed + (1.0 - base_speed) * hill_response(
            np.asarray(Q) + R, cd("bacteria", "migration_bias_halfmax", 0.1), HILL)
        base_apop = m.cell_type("bacteria").deaths["apoptosis"].death_rate
        max_apop = base_apop * cd("bacteria", "relative_max_damage_death", 100.0)
        c.apoptosis_rate[live] = base_apop + (max_apop - base_apop) * hill_response(
            c.damage[live], cd("bacteria", "damage_halfmax", 1.0), HILL)

    @sim.hook("phenotype", cell_type="stem")
    def stem_cell_phenotype(ctx):
        live = dead_leaks_debris(ctx)
        if live.size == 0:
            return
        R = c.sampled[live, nR]
        n_stem = c.contact_count[live, sim.type_index["stem"]]
        n_diff = c.contact_count[live, sim.type_index["differentiated"]]
        rate = float(m.user_parameters.get("max_stem_differentiation", 0.0075))
        c.transformation_rate[live, sim.type_index["differentiated"]] = rate * hill_response(
            n_stem, cd("stem", "differentiation_contact_halfmax", 0.1), HILL)
        base_cycle = m.cell_type("stem").cycle.entry_rate
        c.cycle_entry_rate[live] = base_cycle * (1.0 - hill_response(
            n_diff, cd("stem", "cycling_contact_halfmax", 0.1), HILL))
        c.necrosis_rate[live] = 0.0028 * decreasing_linear_response(
            R, cd("stem", "necrosis_saturation_resource", 0.075),
            cd("stem", "necrosis_threshold_resource", 0.15))

    @sim.hook("phenotype", cell_type="differentiated")
    def differentiated_cell_phenotype(ctx):
        live = dead_leaks_debris(ctx)
        if live.size == 0:
            return
        R = c.sampled[live, nR]
        toxin = c.sampled[live, nTox] if nTox is not None else 0.0
        base_cycle = m.cell_type("differentiated").cycle.entry_rate
        c.cycle_entry_rate[live] = base_cycle * (1.0 - hill_response(
            c.pressure[live], cd("differentiated", "cycling_pressure_halfmax", 0.5), HILL))
        c.necrosis_rate[live] = 0.0028 * decreasing_linear_response(
            R, cd("differentiated", "necrosis_saturation_resource", 0.075),
            cd("differentiated", "necrosis_threshold_resource", 0.15))
        base_apop = m.cell_type("differentiated").deaths["apoptosis"].death_rate
        max_apop = base_apop * cd("differentiated", "relative_max_toxicity", 100.0)
        c.apoptosis_rate[live] = base_apop + (max_apop - base_apop) * hill_response(
            toxin, cd("differentiated", "toxicity_halfmax", 0.2), HILL)

    def follows(signal_sub, type_name, max_bias=0.75):
        def phenotype(ctx):
            live = dead_leaks_debris(ctx)
            if live.size == 0 or signal_sub is None:
                return
            sig = c.sampled[live, signal_sub]
            base = m.cell_type(type_name).motility.migration_bias
            c.migration_bias[live] = base + (max_bias - base) * hill_response(
                sig, cd(type_name, "migration_bias_halfmax", 0.25), HILL)
        return phenotype

    installed = ["bacteria", "stem", "differentiated"]
    for type_name in ("CD8+ T cell", "neutrophil"):
        if type_name in sim.type_index:
            sim.hook("phenotype", cell_type=type_name)(follows(nPIF, type_name))
            installed.append(type_name)

    if "macrophage" in sim.type_index:
        @sim.hook("phenotype", cell_type="macrophage")
        def macrophage_phenotype(ctx):
            live = dead_leaks_debris(ctx)
            if live.size == 0:
                return
            debris = c.sampled[live, nDebris] if nDebris is not None else 0.0
            Q = c.sampled[live, nQuorum] if nQuorum is not None else 0.0
            w_debris = cd("macrophage", "bias_debris_sensitivity", 1.0)
            w_quorum = cd("macrophage", "bias_quorum_sensitivity", 1.0)
            base_bias = m.cell_type("macrophage").motility.migration_bias
            c.migration_bias[live] = base_bias + (0.75 - base_bias) * hill_response(
                w_debris * debris + w_quorum * Q,
                cd("macrophage", "migration_bias_halfmax", 0.05), HILL)
            base_speed = m.cell_type("macrophage").motility.speed
            c.motility_speed[live] = base_speed + (0.1 * base_speed - base_speed) * hill_response(
                w_debris * debris + w_quorum * Q,
                cd("macrophage", "migration_speed_halfmax", 0.1), HILL)
        installed.append("macrophage")
    return f"resource/toxin/quorum ecology for {', '.join(installed)}"

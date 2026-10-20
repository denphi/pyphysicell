"""Volume, cycle and death, vectorized over the population.

The formulas are PhysiCell's: `standard_volume_update_function`,
`Cycle_Model::advance_model` and the standard apoptosis / necrosis models.
"""

from __future__ import annotations

import numpy as np

FOUR_THIRDS_PI = 4.0 / 3.0 * np.pi


def radius_from_volume(v: np.ndarray) -> np.ndarray:
    return np.cbrt(np.maximum(v, 1e-16) / FOUR_THIRDS_PI)


def init_volume(cells, idx: np.ndarray, vol):
    """Seed a cell's volume state from a cell definition's <volume> block."""
    total = vol.total
    fluid = vol.fluid_fraction * total
    nuclear_fluid = (vol.nuclear / (total + 1e-16)) * fluid
    nuclear_solid = vol.nuclear - nuclear_fluid
    cytoplasmic = total - vol.nuclear
    cytoplasmic_fluid = fluid - nuclear_fluid
    cytoplasmic_solid = cytoplasmic - cytoplasmic_fluid

    cells.volume_total[idx] = total
    cells.volume_fluid[idx] = fluid
    cells.volume_nuclear[idx] = vol.nuclear
    cells.volume_cytoplasmic[idx] = cytoplasmic
    cells.volume_nuclear_solid[idx] = nuclear_solid
    cells.volume_cytoplasmic_solid[idx] = cytoplasmic_solid
    cells.fluid_fraction[idx] = vol.fluid_fraction
    cells.calcified_fraction[idx] = vol.calcified_fraction
    cells.target_solid_nuclear[idx] = nuclear_solid
    cells.target_solid_cytoplasmic[idx] = cytoplasmic_solid
    cells.target_fluid_fraction[idx] = vol.fluid_fraction
    cells.target_cyto_to_nuclear[idx] = cytoplasmic_solid / (1e-16 + nuclear_solid)
    cells.fluid_change_rate[idx] = vol.fluid_change_rate
    cells.cyto_biomass_rate[idx] = vol.cytoplasmic_biomass_change_rate
    cells.nuclear_biomass_rate[idx] = vol.nuclear_biomass_change_rate
    cells.calcification_rate[idx] = vol.calcification_rate
    cells.rupture_volume[idx] = vol.relative_rupture_volume * total
    cells.radius[idx] = radius_from_volume(np.asarray(total))


def update_volume(cells, dt: float):
    n = cells.n
    sl = slice(0, n)
    total = cells.volume_total[sl]
    fluid = cells.volume_fluid[sl]

    fluid += dt * cells.fluid_change_rate[sl] * (cells.target_fluid_fraction[sl] * total - fluid)
    np.maximum(fluid, 0.0, out=fluid)

    nuclear_fluid = (cells.volume_nuclear[sl] / (total + 1e-16)) * fluid
    cytoplasmic_fluid = fluid - nuclear_fluid

    ns = cells.volume_nuclear_solid[sl]
    ns += dt * cells.nuclear_biomass_rate[sl] * (cells.target_solid_nuclear[sl] - ns)
    np.maximum(ns, 0.0, out=ns)

    target_cs = cells.target_cyto_to_nuclear[sl] * cells.target_solid_nuclear[sl]
    cs = cells.volume_cytoplasmic_solid[sl]
    cs += dt * cells.cyto_biomass_rate[sl] * (target_cs - cs)
    np.maximum(cs, 0.0, out=cs)

    cells.volume_nuclear[sl] = ns + nuclear_fluid
    cells.volume_cytoplasmic[sl] = cs + cytoplasmic_fluid
    cf = cells.calcified_fraction[sl]
    cf += dt * cells.calcification_rate[sl] * (1.0 - cf)
    cells.volume_total[sl] = cells.volume_nuclear[sl] + cells.volume_cytoplasmic[sl]
    cells.fluid_fraction[sl] = fluid / (1e-16 + cells.volume_total[sl])
    cells.radius[sl] = radius_from_volume(cells.volume_total[sl])


def halve_volume(cells, idx):
    """PhysiCell's Volume::divide -> multiply_by_ratio(0.5): targets and the
    rupture volume scale with the cell, not only the current volumes."""
    for name in ("volume_total", "volume_fluid", "volume_nuclear", "volume_cytoplasmic",
                 "volume_nuclear_solid", "volume_cytoplasmic_solid",
                 "target_solid_nuclear", "target_solid_cytoplasmic", "rupture_volume"):
        getattr(cells, name)[idx] *= 0.5
    cells.radius[idx] = radius_from_volume(cells.volume_total[idx])


def advance_cycle(cells, rng, dt: float, ring_length: np.ndarray, fixed: np.ndarray,
                  rates: np.ndarray, growth_phase: np.ndarray | None = None,
                  division_phase: np.ndarray | None = None) -> np.ndarray:
    """Advance the cycle ring; returns the indices of cells that divide.

    rates[type, phase]  : transition rate out of that phase (1/min)
    fixed[type, phase]  : True when the transition is a fixed duration
    ring_length[type]   : number of phases; the transition back to phase 0 divides
    """
    n = cells.n
    alive = np.flatnonzero(cells.dead[:n] == 0)
    if alive.size == 0:
        return np.zeros(0, dtype=np.int64)

    t_idx = cells.type_idx[alive]
    phase = cells.phase[alive]
    cells.elapsed_time_in_phase[alive] += dt

    # every phase's exit rate is per cell, because rules and hooks can write
    # "exit from cycle phase N"; phase 0 is also "cycle entry"
    rate = cells.cycle_exit_rate[alive, phase]
    rate = np.where(phase == 0, cells.cycle_entry_rate[alive], rate)
    rate = np.where(rate > 0, rate, rates[t_idx, phase])
    is_fixed = fixed[t_idx, phase]
    elapsed = cells.elapsed_time_in_phase[alive]

    with np.errstate(divide="ignore", invalid="ignore"):
        duration = np.where(rate > 0, 1.0 / np.maximum(rate, 1e-300), np.inf)
    go_fixed = is_fixed & (elapsed > duration - 0.5 * dt)
    go_stoch = (~is_fixed) & (rng.random(alive.size) < rate * dt)
    go = go_fixed | go_stoch
    if not go.any():
        return np.zeros(0, dtype=np.int64)

    moving = alive[go]
    nxt = (cells.phase[moving] + 1) % ring_length[cells.type_idx[moving]]
    if division_phase is None:
        divides = moving[nxt == 0]
    else:
        # division happens on *exit* from the model's division phase
        leaving = cells.phase[moving]
        divides = moving[leaving == division_phase[cells.type_idx[moving]]]
    cells.phase[moving] = nxt
    cells.elapsed_time_in_phase[moving] = 0.0
    if growth_phase is not None:
        # phase entry functions (standard_live_phase_entry_function,
        # S_phase_entry_function, ...) double the solid volume targets
        entering = moving[nxt == growth_phase[cells.type_idx[moving]]]
        cells.target_solid_nuclear[entering] *= 2.0
        cells.target_solid_cytoplasmic[entering] *= 2.0
    return divides


def check_for_death(cells, rng, dt: float, death_params):
    """Stochastic entry into apoptosis or necrosis; returns newly dead indices."""
    n = cells.n
    alive = np.flatnonzero(cells.dead[:n] == 0)
    if alive.size == 0:
        return np.zeros(0, dtype=np.int64)
    r = rng.random((alive.size, 2))
    apop = r[:, 0] < cells.apoptosis_rate[alive] * dt
    necr = (~apop) & (r[:, 1] < cells.necrosis_rate[alive] * dt)
    newly = []
    for model_idx, mask in ((0, apop), (1, necr)):
        idx = alive[mask]
        if idx.size == 0:
            continue
        start_death(cells, idx, model_idx, death_params)
        newly.append(idx)
    return np.concatenate(newly) if newly else np.zeros(0, dtype=np.int64)


def start_death(cells, idx, model_idx: int, death_params):
    """PhysiCell's Cell::start_death: stop secreting, stop moving, shrink or swell."""
    cells.dead[idx] = 1
    cells.death_model[idx] = model_idx
    cells.death_phase[idx] = 0
    cells.time_in_death[idx] = 0.0
    cells.is_motile[idx] = 0
    cells.motility_vector[idx] = 0.0
    cells.secretion_rate[idx] = 0.0
    cells.uptake_rate[idx] *= 0.10
    cells.cycle_entry_rate[idx] = 0.0
    cells.apoptosis_rate[idx] = 0.0
    cells.necrosis_rate[idx] = 0.0

    t_idx = cells.type_idx[idx]
    p = death_params[model_idx]
    cells.fluid_change_rate[idx] = p["unlysed_fluid"][t_idx]
    cells.cyto_biomass_rate[idx] = p["cyto"][t_idx]
    cells.nuclear_biomass_rate[idx] = p["nuclear"][t_idx]
    cells.calcification_rate[idx] = p["calcification"][t_idx]
    if model_idx == 0:  # apoptosis: shrink away
        cells.target_fluid_fraction[idx] = 0.0
        cells.target_solid_cytoplasmic[idx] = 0.0
        cells.target_solid_nuclear[idx] = 0.0
        cells.target_cyto_to_nuclear[idx] = 0.0
    else:               # necrosis: swell until rupture, then lyse
        cells.target_fluid_fraction[idx] = 1.0
        cells.target_solid_cytoplasmic[idx] = 0.0
        cells.target_solid_nuclear[idx] = 0.0
        cells.target_cyto_to_nuclear[idx] = 0.0


def advance_death(cells, dt: float, death_params) -> np.ndarray:
    """Progress dead cells; returns the indices to remove from the simulation."""
    n = cells.n
    dead = np.flatnonzero(cells.dead[:n] == 1)
    if dead.size == 0:
        return np.zeros(0, dtype=np.int64)
    cells.time_in_death[dead] += dt
    t_idx = cells.type_idx[dead]
    model = cells.death_model[dead]

    remove = np.zeros(dead.size, dtype=bool)

    apop = model == 0
    if apop.any():
        dur = death_params[0]["duration0"][t_idx[apop]]
        remove[apop] = cells.time_in_death[dead[apop]] > dur

    necr = model == 1
    if necr.any():
        sub = dead[necr]
        swelling = cells.death_phase[sub] == 0
        ruptured = swelling & (cells.volume_total[sub] > cells.rupture_volume[sub])
        if ruptured.any():
            rid = sub[ruptured]
            cells.death_phase[rid] = 1
            cells.time_in_death[rid] = 0.0
            tt = cells.type_idx[rid]
            cells.fluid_change_rate[rid] = death_params[1]["lysed_fluid"][tt]
            cells.target_fluid_fraction[rid] = 0.0
        lysed = cells.death_phase[sub] == 1
        dur = death_params[1]["duration1"][cells.type_idx[sub]]
        rem = lysed & (cells.time_in_death[sub] > dur)
        remove[necr] = rem
    return dead[remove]

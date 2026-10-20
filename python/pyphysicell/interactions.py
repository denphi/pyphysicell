"""Cell-cell interactions: phagocytosis, attack, fusion, transformation, damage.

A port of PhysiCell's `standard_cell_cell_interactions`,
`standard_cell_transformations` and `Cell_Integrity::advance_damage`, done over
the whole population at once instead of cell by cell.  The neighbour pairs come
from the same force pass that computes mechanics, so no extra search is needed.
"""

from __future__ import annotations

import numpy as np

from . import phenotype as ph


def _first_hit(owner: np.ndarray, hit: np.ndarray, n: int):
    """For each owner, the first pair it draws; at most one action per cell per step."""
    if not hit.any():
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    idx = np.flatnonzero(hit)
    first = np.full(n, -1, dtype=np.int64)
    # later writes lose, so iterate in reverse to keep the first
    first[owner[idx][::-1]] = idx[::-1]
    chosen = first[first >= 0]
    return owner[chosen], chosen


def run(sim, dt: float):
    """One interaction pass over every live cell's neighbours."""
    c = sim.cells
    n = c.n
    if n == 0 or sim.neighbor_idx is None:
        return
    cap = sim.neighbor_idx.shape[1]
    counts = np.minimum(c.view("neighbors"), cap)
    if counts.max() == 0:
        _transform(sim, dt)
        _damage(sim, dt)
        return

    # (i, j) pairs: cell i is touching cell j
    slots = np.arange(cap)[None, :] < counts[:, None]
    i = np.repeat(np.arange(n), counts)
    j = sim.neighbor_idx[:n][slots]
    alive_i = c.dead[i] == 0
    i, j = i[alive_i], j[alive_i]
    if i.size == 0:
        _transform(sim, dt)
        _damage(sim, dt)
        return

    # "attacking" lasts for attack_duration, as PhysiCell keeps an attack target,
    # not for the single step in which the attack started
    c.attack_time_left[:n] = np.maximum(c.attack_time_left[:n] - dt, 0.0)
    c.attacking[:n] = (c.attack_time_left[:n] > 0).astype(float)
    dead_j = c.dead[j] == 1
    apoptotic_j = dead_j & (c.death_model[j] == 0)
    necrotic_j = dead_j & (c.death_model[j] == 1)
    other_dead_j = dead_j & ~(apoptotic_j | necrotic_j)
    type_j = c.type_idx[j]
    type_i = c.type_idx[i]

    # -- phagocytosis ----------------------------------------------------
    p = np.where(apoptotic_j, c.apoptotic_phagocytosis_rate[i],
                 np.where(necrotic_j, c.necrotic_phagocytosis_rate[i],
                          np.where(other_dead_j, c.other_dead_phagocytosis_rate[i],
                                   c.live_phagocytosis_rate[i, type_j])))
    eater, pair = _first_hit(i, sim.rng.random(i.size) < p * dt, n)
    if eater.size:
        eaten = j[pair]
        keep = np.ones(eaten.size, dtype=bool)          # a cell can only be eaten once
        _, first = np.unique(eaten, return_index=True)
        keep[:] = False
        keep[first] = True
        eater, eaten = eater[keep], eaten[keep]
        keep = c.volume_total[eaten] > 1e-15
        _ingest(sim, eater[keep], eaten[keep])

    # -- attack ----------------------------------------------------------
    live_j = ~dead_j
    rate = c.attack_rate[i, type_j] * c.immunogenicity[j, type_i] * live_j
    attacker, pair = _first_hit(i, sim.rng.random(i.size) < rate * dt, n)
    if attacker.size:
        victim = j[pair]
        new_damage = c.attack_damage_rate[attacker] * dt
        np.add.at(c.damage, victim, new_damage)
        np.add.at(c.total_attack_time, victim, dt)
        np.add.at(c.damage_delivered, attacker, new_damage)
        c.attack_time_left[attacker] = np.maximum(c.attack_time_left[attacker],
                                                  c.attack_duration[attacker])
        c.attacking[attacker] = 1.0

    # -- fusion ----------------------------------------------------------
    frate = c.fusion_rate[i, type_j] * live_j
    fuser, pair = _first_hit(i, sim.rng.random(i.size) < frate * dt, n)
    if fuser.size:
        other = j[pair]
        ok = (c.dead[other] == 0) & (fuser != other)
        _ingest(sim, fuser[ok], other[ok], fusion=True)

    _transform(sim, dt)
    _damage(sim, dt)


def _ingest(sim, eater, eaten, fusion: bool = False):
    """PhysiCell's Cell::ingest_cell / fuse_cell: the volume moves, the target goes."""
    if eater.size == 0:
        return
    c = sim.cells
    c.volume_cytoplasmic_solid[eater] += c.volume_cytoplasmic_solid[eaten]
    c.volume_fluid[eater] += c.volume_fluid[eaten]
    if fusion:
        c.volume_nuclear_solid[eater] += c.volume_nuclear_solid[eaten]
        c.target_solid_nuclear[eater] += c.target_solid_nuclear[eaten]
    c.target_solid_cytoplasmic[eater] += c.target_solid_cytoplasmic[eaten]
    # only the transferred fraction moves to the eater; the rest is lost, as
    # BioFVM's fraction_transferred_when_ingested says
    c.internalized[eater] += (c.internalized[eaten]
                              * c.fraction_transferred_when_ingested[eaten])
    c.volume_total[eater] = (c.volume_cytoplasmic_solid[eater] + c.volume_nuclear_solid[eater]
                             + c.volume_fluid[eater])
    c.radius[eater] = ph.radius_from_volume(c.volume_total[eater])

    # the ingested cell is emptied and flagged for removal
    for arr in ("volume_total", "volume_fluid", "volume_cytoplasmic_solid",
                "volume_nuclear_solid", "volume_nuclear", "volume_cytoplasmic"):
        getattr(c, arr)[eaten] = 0.0
    c.internalized[eaten] = 0.0
    c.dead[eaten] = 1
    c.movable[eaten] = 0
    c.is_motile[eaten] = 0
    c.secretion_rate[eaten] = 0.0
    c.uptake_rate[eaten] = 0.0
    sim._pending_removal.extend(int(x) for x in eaten)


def _transform(sim, dt: float):
    """standard_cell_transformations: a live cell becomes another cell type."""
    c = sim.cells
    n = c.n
    live = np.flatnonzero(c.dead[:n] == 0)
    if live.size == 0 or not c.transformation_rate[:n].any():
        return
    p = c.transformation_rate[live] * dt                       # (k, ntypes)
    draw = sim.rng.random(p.shape)
    hit = draw <= p
    rows = np.flatnonzero(hit.any(axis=1))
    for r in rows:
        target = int(np.argmax(hit[r]))
        sim.convert_cell(int(live[r]), target)


def _damage(sim, dt: float):
    """Cell_Integrity::advance_damage: damage accrues and is repaired."""
    c = sim.cells
    n = c.n
    if n == 0:
        return
    sl = slice(0, n)
    d = c.damage[sl]
    d += (c.damage_rate[sl] - c.damage_repair_rate[sl]) * dt
    np.maximum(d, 0.0, out=d)

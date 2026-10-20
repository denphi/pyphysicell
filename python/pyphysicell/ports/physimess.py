"""Port of sample_projects/physimess (the PhysiMeSS addon).

PhysiMeSS extends PhysiCell with fibres: ECM agents that are segments rather
than spheres.  Here they live in `sim.fibres` and are handled by the fibre
kernels, because a different geometry needs different inner loops -- the rest
of the project (cells, motility, rules) is the ordinary machinery.
"""

from __future__ import annotations

import numpy as np

from ..fibres import Fibres, is_fibre_type


def seed(sim) -> str:
    """`setup_tissue`: cells at random, and fibres with random orientations."""
    m = sim.model
    p = m.user_parameters
    cell_types = [t for t in m.cell_type_names if not is_fibre_type(t)]
    fibre_types = [t for t in m.cell_type_names if is_fibre_type(t)]

    # the top-level config carries no cells: PhysiMeSS keeps its scenarios in
    # config/<scenario>/*.csv.  Place a few so the cell-fibre mechanics is visible,
    # and say so.
    n_cells = int(p.get("number_of_cells", 0)) or 50
    substituted = int(p.get("number_of_cells", 0)) == 0
    for name in cell_types:
        if n_cells:
            d = m.domain
            pos = np.column_stack([
                sim.rng.uniform(d.x_min, d.x_max, n_cells),
                sim.rng.uniform(d.y_min, d.y_max, n_cells),
                np.zeros(n_cells) if d.use_2D else sim.rng.uniform(d.z_min, d.z_max, n_cells)])
            sim.add_cells(pos, name)

    sim.fibres = Fibres()
    n_fibres = int(p.get("number_of_fibres", 0))
    if n_fibres and fibre_types:
        name = fibre_types[0]
        ct = m.cell_type(name)
        d = m.domain
        centres = np.column_stack([
            sim.rng.uniform(d.x_min, d.x_max, n_fibres),
            sim.rng.uniform(d.y_min, d.y_max, n_fibres),
            np.zeros(n_fibres) if d.use_2D else sim.rng.uniform(d.z_min, d.z_max, n_fibres)])
        theta = sim.rng.uniform(0.0, 2.0 * np.pi, n_fibres)
        if d.use_2D:
            dirs = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(n_fibres)])
        else:
            dirs = sim.rng.normal(size=(n_fibres, 3))
        length = float(ct.custom_data.get("fibre_length", p.get("fibre_length", 75.0)))
        radius = float(ct.custom_data.get("fibre_radius", p.get("fibre_radius", 2.0)))
        sim.fibres.add(centres, dirs, half_length=0.5 * length, radius=radius,
                       repulsion=ct.mechanics.cell_cell_repulsion_strength,
                       type_idx=sim.type_index[name])
    note = (" [the project's own config has number_of_cells=0; its scenarios live in "
            "config/<scenario>/*.csv, so these cells are a stand-in]" if substituted else "")
    return (f"{sim.cells.n} cells and {sim.fibres.n} fibres "
            f"(length {float(p.get('fibre_length', 75.0)):g} um){note}")


def install(sim) -> str:
    # PhysiMeSS's per-cell switches; defaults match its config
    c = sim.cells
    for key, default in (("fibre_pushing", 1.0), ("fibre_rotation", 1.0),
                         ("fibre_sticky", 1.0)):
        if key in c.custom_names and not c.custom[: c.n, c.custom_names.index(key)].any():
            c.custom[: c.n, c.custom_names.index(key)] = default
    if sim.fibres is not None:
        push = any("fibre_pushing" in ct.custom_data for ct in sim.model.cell_types)
        sim.fibres.push[: sim.fibres.n] = 1 if push or True else 0
    return "cell-fibre mechanics through the fibre kernels"

"""Port of sample_projects/physimess (the PhysiMeSS addon).

PhysiMeSS extends PhysiCell with fibres: ECM agents that are segments rather
than spheres.  Here they live in `sim.fibres` and are handled by the fibre
kernels, because a different geometry needs different inner loops -- the rest
of the project (cells, motility, rules) is the ordinary machinery.
"""

from __future__ import annotations

import numpy as np

from .. import fibres as fibre_mod
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
    dropped = 0
    if n_fibres and fibre_types:
        name = fibre_types[0]
        ct = m.cell_type(name)
        d = m.domain
        centres = np.column_stack([
            sim.rng.uniform(d.x_min, d.x_max, n_fibres),
            sim.rng.uniform(d.y_min, d.y_max, n_fibres),
            np.zeros(n_fibres) if d.use_2D else sim.rng.uniform(d.z_min, d.z_max, n_fibres)])
        # assign_fibre_orientation + check_out_of_bounds: lengths drawn per
        # fibre, orientation isotropic or around fibre_angle, and anything
        # still sticking out of the domain after ten tries is discarded
        centres, dirs, half, radius, dropped = fibre_mod.place(m, ct, centres, sim.rng, name)
        if len(centres):
            sim.fibres.add(centres, dirs, half_length=half, radius=radius,
                           repulsion=ct.mechanics.cell_cell_repulsion_strength,
                           rel_max_adh=ct.mechanics.relative_maximum_adhesion_distance,
                           type_idx=sim.type_index[name])
    note = (" [the project's own config has number_of_cells=0; its scenarios live in "
            "config/<scenario>/*.csv, so these cells are a stand-in]" if substituted else "")
    lost = f", {dropped} dropped for leaving the domain" if dropped else ""
    return (f"{sim.cells.n} cells and {sim.fibres.n} fibres "
            f"(length {float(p.get('fibre_length', 75.0)):g} um{lost}){note}")


def install(sim) -> str:
    """PhysiMeSS reads its switches from the cell doing the pushing.

    They are left exactly as the XML sets them. An earlier version of this port
    defaulted `fibre_pushing` and `fibre_rotation` to 1.0 whenever the column
    was all zeros, which silently turned on behaviour the project's own config
    turns off.
    """
    c = sim.cells
    on = [k for k in ("fibre_pushing", "fibre_rotation", "fibre_degradation")
          if k in c.custom_names and c.n
          and float(c.custom[: c.n, c.custom_names.index(k)].max()) > 0.5]
    return ("cell-fibre mechanics through the fibre kernels"
            + (f"; enabled by the config: {', '.join(on)}" if on else
               "; the config leaves pushing, rotation and degradation off"))

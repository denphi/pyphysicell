"""Initial conditions.

Most sample projects place their cells in C++ (`setup_tissue` in
`custom_modules/custom.cpp`), not in `cells.csv`, so a project often loads with
an empty population.  The placements upstream uses fall into a few patterns:

  1. `number_of_cells` of **every** cell type, uniformly at random in the domain
  2. named counts per type (`number_of_prey`, `number_of_infected_cells`, ...)
  3. a hex-packed disc or sphere of one type (a tumor), radius from a parameter

`auto(model)` recognises these from the project's own `<user_parameters>` and
returns both the cells and a sentence saying what it did, so a notebook can
state plainly that this is a port of the C++ placement, not the C++ itself.
"""

from __future__ import annotations

import re

import numpy as np

from .model import Model


def uniform_random(model: Model, n: int, cell_type: str, rng=None):
    """n cells of one type, uniform in the domain (PhysiCell's common loop)."""
    rng = rng or np.random.default_rng(model.random_seed)
    d = model.domain
    x = d.x_min + rng.random(n) * (d.x_max - d.x_min)
    y = d.y_min + rng.random(n) * (d.y_max - d.y_min)
    z = np.zeros(n) if d.use_2D else d.z_min + rng.random(n) * (d.z_max - d.z_min)
    return [(float(a), float(b), float(c), cell_type) for a, b, c in zip(x, y, z)]


def hex_disc(model: Model, radius: float, cell_type: str, spacing: float | None = None,
             center=(0.0, 0.0, 0.0)):
    """A hex-packed disc, as `setup_tissue` builds a 2-D tumor."""
    ct = model.cell_type(cell_type)
    r_cell = (3.0 * ct.volume.total / (4.0 * np.pi)) ** (1.0 / 3.0)
    s = spacing if spacing is not None else 0.95 * 2.0 * r_cell
    out = []
    dy = s * np.sqrt(3.0) / 2.0
    ny = int(radius / dy)
    for j in range(-ny, ny + 1):
        y = j * dy
        offset = 0.5 * s if (j % 2) else 0.0
        nx = int(np.sqrt(max(radius * radius - y * y, 0.0)) / s) + 1
        for i in range(-nx, nx + 1):
            x = i * s + offset
            if x * x + y * y <= radius * radius:
                out.append((center[0] + x, center[1] + y, center[2], cell_type))
    return out


def hex_sphere(model: Model, radius: float, cell_type: str, spacing: float | None = None,
               center=(0.0, 0.0, 0.0)):
    """A hex-packed sphere, as the 3-D samples build a spheroid."""
    ct = model.cell_type(cell_type)
    r_cell = (3.0 * ct.volume.total / (4.0 * np.pi)) ** (1.0 / 3.0)
    s = spacing if spacing is not None else 0.95 * 2.0 * r_cell
    dy, dz = s * np.sqrt(3.0) / 2.0, s * np.sqrt(3.0) / 2.0
    out = []
    nz = int(radius / dz)
    for k in range(-nz, nz + 1):
        z = k * dz
        ry = np.sqrt(max(radius * radius - z * z, 0.0))
        ny = int(ry / dy)
        for j in range(-ny, ny + 1):
            y = j * dy
            offx = 0.5 * s if ((j + k) % 2) else 0.0
            nx = int(np.sqrt(max(ry * ry - y * y, 0.0)) / s) + 1
            for i in range(-nx, nx + 1):
                x = i * s + offx
                if x * x + y * y + z * z <= radius * radius:
                    out.append((center[0] + x, center[1] + y, center[2] + z, cell_type))
    return out


_COUNT_RE = re.compile(r"^(?:number_of_|num_)(.+?)s?$")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _match_type(token: str, type_names: list[str]) -> str | None:
    """Map a parameter stem such as 'CD8T_cell' onto a cell type name."""
    t = _norm(token)
    for name in type_names:                      # exact, then with/without "cell"
        n = _norm(name)
        if t in (n, n + "s", n.rstrip("s")) or t + "cell" == n or t == n.replace("cell", ""):
            return name
    for name in type_names:                      # containment, longest name first
        if _norm(name) and (_norm(name) in t or t in _norm(name)):
            return name
    return None


def auto(model: Model, rng=None) -> tuple[list, str]:
    """Best-effort port of the project's own initial placement.

    Returns (cells, description).  The description is meant to be printed: it
    says which rule was used and which C++ it stands in for.
    """
    if model.initial_cells:
        return list(model.initial_cells), (
            f"{len(model.initial_cells)} cells from the project's own cells.csv")

    rng = rng or np.random.default_rng(model.random_seed)
    names = model.cell_type_names
    if not names:
        return [], 'this model defines no cell types, so there is nothing to place'

    params = {k.lower(): v for k, v in model.user_parameters.items()
              if isinstance(v, (int, float)) and not isinstance(v, bool)}

    # 2: named counts, e.g. number_of_prey -> "prey"
    named: list[tuple[str, str, int]] = []
    for key, value in params.items():
        m = _COUNT_RE.match(key)
        if not m or key in ("number_of_cells",) or value <= 0:
            continue
        target = _match_type(m.group(1), names)
        if target:
            named.append((key, target, int(value)))
    # counts whose name matches no cell type belong to the types nothing claimed
    # (virus_macrophage: number_of_infected_cells -> "epithelial cell")
    unmatched = [(k, int(v)) for k, v in params.items()
                 if (m := _COUNT_RE.match(k)) and k != "number_of_cells" and v > 0
                 and _match_type(m.group(1), names) is None]
    claimed = {t for _k, t, _c in named}
    spare = [n for n in names if n not in claimed]
    notes = []
    for key, count in unmatched:
        if spare:
            named.append((key, spare[0], count))
            notes.append(f"{key} -> {spare[0]}")

    if named:
        cells = []
        for _key, target, count in named:
            cells += uniform_random(model, count, target, rng)
        detail = ", ".join(f"{c} {t}" for _k, t, c in named)
        # a radius parameter alongside counts means a tumor plus something else
        # (cancer_immune: a spheroid of cancer cells, then immune cells)
        radius = next((float(params[k]) for k in
                       ("tumor_radius", "colony_radius", "initial_radius")
                       if k in params and params[k] > 0), None)
        claimed = {t for _k, t, _c in named}
        rest = [n for n in names if n not in claimed]
        extra = ""
        if radius and rest:
            target = rest[0]
            blob = (hex_disc(model, radius, target) if model.domain.use_2D
                    else hex_sphere(model, radius, target))
            cells += blob
            extra = (f", plus a hex-packed {'disc' if model.domain.use_2D else 'sphere'} of "
                     f"{len(blob)} {target} cells of radius {radius:g} um")
        guessed = f" [matched by name: {'; '.join(notes)}]" if notes else ""
        return cells, (f"placed {len(cells)} cells uniformly at random ({detail}){extra} from the "
                       f"project's own user parameters -- a port of its C++ setup_tissue{guessed}")

    # 1: number_of_cells of every type
    n = int(params.get("number_of_cells", 0))
    if n > 0:
        cells = []
        for name in names:
            cells += uniform_random(model, n, name, rng)
        return cells, (f"placed {n} cells of each of the {len(names)} cell types uniformly at "
                       f"random (number_of_cells={n}) -- a port of its C++ setup_tissue")

    # 3: a disc or sphere of the first type, radius from a parameter if there is one
    radius = None
    for key in ("tumor_radius", "colony_radius", "initial_radius", "max_distance_from_origin"):
        if key in params and params[key] > 0:
            radius = float(params[key])
            break
    d = model.domain
    if radius is None:
        radius = 0.15 * min(d.x_max - d.x_min, d.y_max - d.y_min)
    target = names[0]
    cells = (hex_disc(model, radius, target) if d.use_2D
             else hex_sphere(model, radius, target))
    return cells, (f"placed a hex-packed {'disc' if d.use_2D else 'sphere'} of {len(cells)} "
                   f"{target} cells, radius {radius:g} um -- this project seeds its cells in C++ "
                   f"and gives no count parameter, so this is a stand-in, not its initial state")


def apply(sim, cells=None, description: str | None = None) -> str:
    """Seed a Simulation with `cells` (or `auto`), returning the description."""
    if cells is None:
        cells, description = auto(sim.model)
    by_type: dict[str, list] = {}
    for (x, y, z, name) in cells:
        by_type.setdefault(name, []).append((x, y, z))
    for name, pos in by_type.items():
        if name in sim.type_index:
            sim.add_cells(np.asarray(pos, dtype=float), name)
    return description or f"{len(cells)} cells"

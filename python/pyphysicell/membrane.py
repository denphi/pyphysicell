"""Basement membranes.

PhysiCell leaves the membrane to the project: `calculate_distance_to_membrane`
is a `Cell_Functions` slot that every sample project sets to NULL, and
`state.contact_with_basement_membrane` is marked "not implemented yet"
upstream, so the `contact with basement membrane` signal is always false there.

Here a membrane is an object with a geometry, the force is PhysiCell's
`standard_add_basement_membrane_interactions`, and the contact signal is real.
"""

from __future__ import annotations

import numpy as np


class Membrane:
    """distance_and_direction(pos) -> (distance, unit vectors toward the membrane)."""

    def distance_and_direction(self, pos: np.ndarray):
        raise NotImplementedError


class Sphere(Membrane):
    """A spherical shell the cells stay inside — the usual tissue boundary."""

    def __init__(self, radius: float, center=(0.0, 0.0, 0.0)):
        self.radius = float(radius)
        self.center = np.asarray(center, dtype=float)

    def distance_and_direction(self, pos):
        rel = pos - self.center
        r = np.linalg.norm(rel, axis=1)
        safe = np.maximum(r, 1e-12)
        direction = -rel / safe[:, None]          # inward, as PhysiCell's examples do
        return np.abs(self.radius - r), direction


class Box(Membrane):
    """The domain walls as a membrane: distance to the nearest wall."""

    def __init__(self, bounds):
        self.lo = np.asarray(bounds[0], dtype=float)
        self.hi = np.asarray(bounds[1], dtype=float)

    def distance_and_direction(self, pos):
        to_lo, to_hi = pos - self.lo, self.hi - pos
        stacked = np.concatenate([to_lo, to_hi], axis=1)        # (n, 6)
        which = np.argmin(stacked, axis=1)
        dist = stacked[np.arange(pos.shape[0]), which]
        direction = np.zeros_like(pos)
        for axis in range(3):
            direction[which == axis, axis] = 1.0                # push away from the low wall
            direction[which == axis + 3, axis] = -1.0           # ... and from the high wall
        return np.abs(dist), direction


class Callable(Membrane):
    """Any geometry: fn(pos) -> (distance, unit vectors)."""

    def __init__(self, fn):
        self.fn = fn

    def distance_and_direction(self, pos):
        return self.fn(pos)


def apply(sim):
    """`standard_add_basement_membrane_interactions`, over the population."""
    membrane = getattr(sim, "membrane", None)
    c = sim.cells
    n = c.n
    if membrane is None or n == 0:
        return
    pos = c.position[:n]
    distance, direction = membrane.distance_and_direction(pos)
    # cells that have been ingested have zero volume, hence zero radius
    radius = np.maximum(c.radius[:n], 1e-12)
    max_interactive = np.maximum(c.rel_max_adhesion[:n] * radius, 1e-12)
    alive = (c.volume_total[:n] > 1e-15) & (c.movable[:n] == 1)

    temp_a = np.zeros(n)
    close = distance < max_interactive
    temp_a[close] = (1.0 - distance[close] / max_interactive[close]) ** 2
    temp_a[close] *= -c.cell_BM_adhesion[:n][close]

    temp_r = np.zeros(n)
    inside = distance < radius
    temp_r[inside] = (1.0 - distance[inside] / radius[inside]) ** 2
    temp_r[inside] *= c.cell_BM_repulsion[:n][inside]

    total = (temp_r + temp_a) * alive
    acting = np.abs(total) >= 1e-16
    c.velocity[:n][acting] += total[acting, None] * direction[acting]
    # PhysiCell declares this signal but never sets it ("not implemented yet")
    c.contact_bm[:n] = (inside & alive).astype(float)

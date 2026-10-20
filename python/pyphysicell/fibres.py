"""Fibres: extended agents, as PhysiMeSS models the extracellular matrix.

A cell is a sphere; a fibre is a segment with a centre, an orientation, a
half-length and a radius.  That is a different data model, not a longer
function, which is why it needs its own container and its own kernels rather
than a hook (DESIGN.md ADR-015).

Forces follow PhysiMeSS: the cell-cell repulsion law with the distance taken to
the nearest point on the segment, cells pushing and turning uncrosslinked
fibres, and fibres pushing back on cells.
"""

from __future__ import annotations

import numpy as np

from . import _kernels as K

# PhysiMeSS decides what is a fibre from the cell type's name
FIBRE_WORDS = ("ecm", "matrix", "fiber", "fibre", "rod")


def is_fibre_type(name: str) -> bool:
    low = name.lower()
    return any(w in low for w in FIBRE_WORDS)


class Fibres:
    """Struct-of-arrays store of fibre segments."""

    _VEC3 = ("position", "direction", "velocity")
    _SCALAR_F = ("half_length", "radius", "repulsion", "angle", "pressure")
    _SCALAR_I = ("crosslinks", "type_idx", "ID")
    _SCALAR_B = ("push", "rotate")

    def __init__(self, capacity: int = 256):
        self.n = 0
        self._cap = 0
        self._alloc(capacity)

    def _alloc(self, cap):
        old = self._cap
        for name in self._VEC3:
            self._grow(name, (cap, 3), np.float64, old)
        for name in self._SCALAR_F:
            self._grow(name, (cap,), np.float64, old)
        for name in self._SCALAR_I:
            self._grow(name, (cap,), np.int64, old)
        for name in self._SCALAR_B:
            self._grow(name, (cap,), np.uint8, old)
        self._cap = cap

    def _grow(self, name, shape, dtype, old):
        new = np.zeros(shape, dtype=dtype)
        if old:
            new[:old] = getattr(self, name)[:old]
        setattr(self, name, new)

    def add(self, positions, directions, half_length, radius, repulsion=10.0,
            push=True, rotate=True, type_idx=0):
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        count = positions.shape[0]
        if self.n + count > self._cap:
            self._alloc(max(2 * self._cap, self.n + count))
        idx = np.arange(self.n, self.n + count)
        self.n += count
        d = np.atleast_2d(np.asarray(directions, dtype=float))
        norm = np.linalg.norm(d, axis=1, keepdims=True)
        self.position[idx] = positions
        self.direction[idx] = np.divide(d, norm, out=np.zeros_like(d), where=norm > 1e-16)
        self.half_length[idx] = half_length
        self.radius[idx] = radius
        self.repulsion[idx] = repulsion
        self.push[idx] = 1 if push else 0
        self.rotate[idx] = 1 if rotate else 0
        self.type_idx[idx] = type_idx
        self.ID[idx] = idx
        return idx

    def view(self, name):
        return getattr(self, name)[: self.n]

    @property
    def endpoints(self):
        """(n, 2, 3): the two ends of every fibre, for drawing."""
        c, u, L = self.view("position"), self.view("direction"), self.view("half_length")[:, None]
        return np.stack([c - L * u, c + L * u], axis=1)

    def to_dataframe(self):
        import pandas as pd

        c, u = self.view("position"), self.view("direction")
        return pd.DataFrame({"ID": self.view("ID"), "x": c[:, 0], "y": c[:, 1], "z": c[:, 2],
                             "ux": u[:, 0], "uy": u[:, 1], "uz": u[:, 2],
                             "half_length": self.view("half_length"),
                             "radius": self.view("radius"),
                             "crosslinks": self.view("crosslinks")})


def step(sim, dt: float):
    """One fibre-mechanics pass: fibres push cells, cells push and turn fibres."""
    f = sim.fibres
    c = sim.cells
    if f is None or f.n == 0 or c.n == 0:
        return

    d = sim.model.domain
    reach = float(2.0 * (f.view("half_length") + f.view("radius")).max()
                  + 2.0 * c.view("radius").max())
    h = max(reach, 30.0)
    gx = max(1, int(np.ceil((d.x_max - d.x_min) / h)))
    gy = max(1, int(np.ceil((d.y_max - d.y_min) / h)))
    gz = 1 if d.use_2D else max(1, int(np.ceil((d.z_max - d.z_min) / h)))
    origin = (d.x_min, d.y_min, d.z_min)

    cpos = np.ascontiguousarray(c.position[: c.n])
    fpos = np.ascontiguousarray(f.position[: f.n])

    # fibres -> cells
    fhead = np.full(gx * gy * gz, -1, dtype=np.int64)
    fnext = np.full(f.n, -1, dtype=np.int64)
    K.build_cell_list(fpos, origin, h, (gx, gy, gz), fhead, fnext)
    K.fibre_push_cells(cpos, np.ascontiguousarray(c.radius[: c.n]),
                       np.ascontiguousarray(c.repulsion[: c.n]),
                       np.ascontiguousarray(c.movable[: c.n]),
                       fpos, np.ascontiguousarray(f.direction[: f.n]),
                       np.ascontiguousarray(f.half_length[: f.n]),
                       np.ascontiguousarray(f.radius[: f.n]),
                       np.ascontiguousarray(f.repulsion[: f.n]),
                       origin, h, (gx, gy, gz), fhead, fnext, d.use_2D,
                       c.velocity[: c.n], c.pressure[: c.n], c.neighbors[: c.n])

    # cells -> fibres
    chead = np.full(gx * gy * gz, -1, dtype=np.int64)
    cnext = np.full(c.n, -1, dtype=np.int64)
    K.build_cell_list(cpos, origin, h, (gx, gy, gz), chead, cnext)
    sticky = (c.custom[: c.n, c.custom_names.index("fibre_sticky")]
              if "fibre_sticky" in c.custom_names else np.ones(c.n))
    f.velocity[: f.n] = 0.0
    f.angle[: f.n] = 0.0
    f.pressure[: f.n] = 0.0
    K.cells_push_fibres(cpos, np.ascontiguousarray(c.radius[: c.n]),
                        np.ascontiguousarray(c.repulsion[: c.n]),
                        np.ascontiguousarray(c.motility_speed[: c.n]),
                        np.ascontiguousarray(sticky),
                        fpos, np.ascontiguousarray(f.direction[: f.n]),
                        np.ascontiguousarray(f.half_length[: f.n]),
                        np.ascontiguousarray(f.radius[: f.n]),
                        np.ascontiguousarray(f.repulsion[: f.n]),
                        np.ascontiguousarray(f.push[: f.n]),
                        np.ascontiguousarray(f.rotate[: f.n]),
                        np.ascontiguousarray(f.crosslinks[: f.n]),
                        origin, h, (gx, gy, gz), chead, cnext, d.use_2D,
                        f.velocity[: f.n], f.angle[: f.n], f.pressure[: f.n])

    # integrate: displacement, then rotation about the fibre centre (2-D as PhysiMeSS)
    f.position[: f.n] += dt * f.velocity[: f.n]
    angle = f.angle[: f.n]
    turning = np.flatnonzero(np.abs(angle) > 0)
    if turning.size:
        a = angle[turning]
        u = f.direction[turning]
        cos_a, sin_a = np.cos(a), np.sin(a)
        ux = u[:, 0] * cos_a - u[:, 1] * sin_a
        uy = u[:, 0] * sin_a + u[:, 1] * cos_a
        u[:, 0], u[:, 1] = ux, uy
        n = np.linalg.norm(u, axis=1, keepdims=True)
        f.direction[turning] = np.divide(u, n, out=u, where=n > 1e-16)
    np.clip(f.position[: f.n, 0], d.x_min, d.x_max, out=f.position[: f.n, 0])
    np.clip(f.position[: f.n, 1], d.y_min, d.y_max, out=f.position[: f.n, 1])


def crosslink(sim, distance: float | None = None) -> int:
    """Count fibre-fibre contacts and mark crosslinked fibres (they stop moving)."""
    f = sim.fibres
    if f is None or f.n < 2:
        return 0
    ends = f.endpoints
    f.crosslinks[: f.n] = 0
    tol = distance if distance is not None else float(2.0 * f.view("radius").max())
    for a in range(f.n):
        for b in range(a + 1, f.n):
            d = np.linalg.norm(ends[a][:, None, :] - ends[b][None, :, :], axis=2).min()
            if d < tol:
                f.crosslinks[a] += 1
                f.crosslinks[b] += 1
    return int((f.crosslinks[: f.n] > 0).sum())

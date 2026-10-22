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


# PhysiMeSS reads these from custom data.  The 1.14.2 source asks for
# "fibre_stuck_time" and "fibre_degradation_rate", which no shipped XML
# defines -- the configs call them "fibre_stuck" and "fibre_deg_rate" -- so
# both spellings are accepted and the XML one wins.
PARAM_ALIASES = {
    "fibre_stuck_time": ("fibre_stuck", "fibre_stuck_time"),
    "fibre_degradation_rate": ("fibre_deg_rate", "fibre_degradation_rate"),
}
PARAM_DEFAULTS = {
    "fibre_pushing": 0.0, "fibre_rotation": 0.0, "fibre_sticky": 1.0,
    "fibre_degradation": 0.0, "fibre_degradation_rate": 0.01,
    "fibre_custom_degradation": 0.0,
    "fibre_stuck_time": 10.0, "fibre_stuck_threshold": 0.05,
    "vel_adhesion": 0.0, "vel_contact": 0.0, "cell_velocity_max": 1.0,
    "fibre_length": 75.0, "length_normdist_sd": 0.0, "fibre_radius": 2.0,
    "anisotropic_fibres": 0.0, "fibre_angle": 0.0, "angle_normdist_sd": 0.0,
}

# physimess_update_cell_velocity hard-codes both
STUCK_THRESHOLD = 10
UNSTUCK_THRESHOLD = 1


def param(sim, name: str) -> np.ndarray:
    """A per-cell array for one PhysiMeSS custom-data parameter."""
    c = sim.cells
    for key in PARAM_ALIASES.get(name, (name,)):
        if key in c.custom_names:
            col = c.custom[: c.n, c.custom_names.index(key)]
            if np.any(col):
                return col
    return np.full(c.n, float(PARAM_DEFAULTS.get(name, 0.0)))


def _nearest_on_segment(points, centre, direction, half):
    """PhysiMeSS_Fibre::nearest_point_on_fibre, vectorized.

    Returns `point - nearest_point_on_the_segment`, broadcasting over any
    leading shape.  The projection length upstream reduces to |to_point . u|.
    """
    start = centre - half[..., None] * direction
    to_point = points - start
    full = 2.0 * half
    proj = np.einsum("...i,...i->...", to_point, direction)
    dot = proj * full
    before = (dot < 0.0)[..., None]
    after = (dot > full * full)[..., None]
    end_disp = points - (centre + half[..., None] * direction)
    mid_disp = to_point - np.abs(proj)[..., None] * direction
    return np.where(before, to_point, np.where(after, end_disp, mid_disp))



class Fibres:
    """Struct-of-arrays store of fibre segments."""

    _VEC3 = ("position", "direction", "velocity", "crosslink_point")
    _SCALAR_F = ("half_length", "radius", "repulsion", "angle", "pressure",
                 "rel_max_adh")
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
            push=True, rotate=True, type_idx=0, rel_max_adh=1.25):
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
        self.rel_max_adh[idx] = rel_max_adh
        self.push[idx] = 1 if push else 0
        self.rotate[idx] = 1 if rotate else 0
        self.type_idx[idx] = type_idx
        self.ID[idx] = idx
        return idx

    def remove(self, indices):
        """Delete fibres (degradation, or placement that never succeeded).

        Compacts by moving the tail into the holes, which is what `Cells` does
        for the same reason: the arrays stay dense and the kernels keep taking
        a contiguous slice.
        """
        doomed = np.unique(np.asarray(indices, dtype=np.int64))
        doomed = doomed[(doomed >= 0) & (doomed < self.n)]
        if doomed.size == 0:
            return 0
        keep = np.ones(self.n, dtype=bool)
        keep[doomed] = False
        order = np.flatnonzero(keep)
        for name in (*self._VEC3, *self._SCALAR_F, *self._SCALAR_I, *self._SCALAR_B):
            arr = getattr(self, name)
            arr[: order.size] = arr[order]
        self.n = int(order.size)
        return int(doomed.size)

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
    """One fibre-mechanics pass, following physimess_update_cell_velocity.

    Order matters and is upstream's: count how long each cell has been stuck,
    let fibres act on cells (repulsion plus the velocity-dependent adhesion the
    kernel applies), let cells push and turn fibres, then degrade fibres that a
    stuck cell is pressing into.
    """
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

    # physimess_mechanics recounts crosslinks before the forces each step
    crosslink(sim)
    _update_stuck(sim)

    # fibres -> cells
    fhead = np.full(gx * gy * gz, -1, dtype=np.int64)
    fnext = np.full(f.n, -1, dtype=np.int64)
    K.build_cell_list(fpos, origin, h, (gx, gy, gz), fhead, fnext)
    near_fibre = np.full(c.n, -1, dtype=np.int64)
    near_disp = np.zeros((c.n, 3))
    K.fibre_push_cells(cpos, np.ascontiguousarray(c.radius[: c.n]),
                       np.ascontiguousarray(c.repulsion[: c.n]),
                       np.ascontiguousarray(c.movable[: c.n]),
                       fpos, np.ascontiguousarray(f.direction[: f.n]),
                       np.ascontiguousarray(f.half_length[: f.n]),
                       np.ascontiguousarray(f.radius[: f.n]),
                       np.ascontiguousarray(f.repulsion[: f.n]),
                       origin, h, (gx, gy, gz), fhead, fnext, d.use_2D,
                       np.ascontiguousarray(c.prev_velocity[: c.n]),
                       np.ascontiguousarray(c.rel_max_adhesion[: c.n]),
                       np.ascontiguousarray(f.rel_max_adh[: f.n]),
                       np.ascontiguousarray(param(sim, "vel_adhesion")),
                       np.ascontiguousarray(param(sim, "vel_contact")),
                       np.ascontiguousarray(np.maximum(param(sim, "cell_velocity_max"), 1e-9)),
                       c.velocity[: c.n], c.pressure[: c.n], c.neighbors[: c.n],
                       near_fibre, near_disp)

    # cells -> fibres
    chead = np.full(gx * gy * gz, -1, dtype=np.int64)
    cnext = np.full(c.n, -1, dtype=np.int64)
    K.build_cell_list(cpos, origin, h, (gx, gy, gz), chead, cnext)
    f.velocity[: f.n] = 0.0
    f.angle[: f.n] = 0.0
    f.pressure[: f.n] = 0.0
    K.cells_push_fibres(cpos, np.ascontiguousarray(c.radius[: c.n]),
                        np.ascontiguousarray(c.repulsion[: c.n]),
                        np.ascontiguousarray(c.motility_speed[: c.n]),
                        np.ascontiguousarray(param(sim, "fibre_sticky")),
                        np.ascontiguousarray(param(sim, "fibre_pushing")),
                        np.ascontiguousarray(param(sim, "fibre_rotation")),
                        fpos, np.ascontiguousarray(f.direction[: f.n]),
                        np.ascontiguousarray(f.half_length[: f.n]),
                        np.ascontiguousarray(f.radius[: f.n]),
                        np.ascontiguousarray(f.repulsion[: f.n]),
                        np.ascontiguousarray(f.crosslinks[: f.n]),
                        origin, h, (gx, gy, gz), chead, cnext, d.use_2D,
                        f.velocity[: f.n], f.angle[: f.n], f.pressure[: f.n])

    _integrate(f, d, dt)
    _degrade(sim, near_fibre, near_disp)


def _integrate(f, d, dt):
    """Move and turn the fibres: free ones about their centre, hinged ones
    about the crosslink they are pinned to (PhysiMeSS, 2-D)."""
    xl = f.view("crosslinks")
    free = xl == 0
    f.position[: f.n][free] += dt * f.velocity[: f.n][free]

    angle = f.view("angle")
    turning = np.flatnonzero((np.abs(angle) > 0) & (xl <= 1))
    if turning.size:
        a = angle[turning]
        u = f.direction[turning].copy()
        cos_a, sin_a = np.cos(a), np.sin(a)
        ux = u[:, 0] * cos_a - u[:, 1] * sin_a
        uy = u[:, 0] * sin_a + u[:, 1] * cos_a
        u[:, 0], u[:, 1] = ux, uy
        norm = np.linalg.norm(u, axis=1, keepdims=True)
        f.direction[turning] = np.divide(u, norm, out=u, where=norm > 1e-16)

        hinged = turning[xl[turning] == 1]
        if hinged.size:
            # swing about the crosslink, keeping the distance to it
            link = f.crosslink_point[hinged]
            arm = np.linalg.norm(link[:, :2] - f.position[hinged][:, :2], axis=1)
            f.position[hinged, 0] = link[:, 0] - arm * f.direction[hinged, 0]
            f.position[hinged, 1] = link[:, 1] - arm * f.direction[hinged, 1]

    np.clip(f.position[: f.n, 0], d.x_min, d.x_max, out=f.position[: f.n, 0])
    np.clip(f.position[: f.n, 1], d.y_min, d.y_max, out=f.position[: f.n, 1])


def crosslink(sim, chunk: int = 128) -> int:
    """Find fibre-fibre crosslinks, as PhysiMeSS_Fibre::check_fibre_crosslinks.

    Three tests, all three kept from upstream:

      1. coplanar, parallel and colinear -- the fibre vectors' cross product
         vanishes, the centre-to-centre direction matches the orientation, and
         the centres are closer than the summed half-lengths.
      2. parallel fibres lying on top of one another -- an endpoint of either
         fibre is within the summed radii of the other.
      3. skew fibres -- the scalar triple product is below the summed radii, and
         both line parameters fall in [0, 1] widened by radius/length.

    Two upstream quirks are reproduced rather than corrected, because the point
    of this library is to agree with PhysiCell 1.14.2:

      * test 1 compares the centre-to-centre unit vector to the orientation
        with `==` on doubles, so it effectively never fires;
      * test 2's condition is `a || b || c || (d && e && f)` under C precedence,
        not the `(a||b||c||d) && e && f` the layout suggests.

    Sets `crosslinks` (PhysiMeSS's X_crosslink_count) and `crosslink_point` per
    fibre, which together decide whether a fibre is free, hinged or held.
    """
    f = sim.fibres
    if f is None or f.n < 2:
        return 0
    n = f.n
    pos, u = f.view("position"), f.view("direction")
    L, R = f.view("half_length"), f.view("radius")
    p1 = pos - L[:, None] * u
    p2 = pos + L[:, None] * u
    v = p2 - p1

    counts = np.zeros(n, dtype=np.int64)
    point = np.zeros((n, 3))

    # Only pairs whose centres are closer than their combined reach can touch.
    # Sorting by x makes that window contiguous, so each block of rows compares
    # against a slice rather than the whole population: without it every
    # mechanics step is an n^2 sweep that costs more than the rest of the time
    # loop put together.
    order = np.argsort(pos[:, 0], kind="stable")
    xs = pos[order, 0]
    span = float(2.0 * (L.max() + R.max()))

    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        rows = order[lo:hi]
        jlo = int(np.searchsorted(xs, xs[lo] - span, side="left"))
        jhi = int(np.searchsorted(xs, xs[hi - 1] + span, side="right"))
        cols = order[jlo:jhi]
        c2c_all = pos[cols][None, :, :] - pos[rows][:, None, :]
        dist_all = np.linalg.norm(c2c_all, axis=2)
        reach = (L[rows][:, None] + L[cols][None, :]) + (R[rows][:, None] + R[cols][None, :])
        cand = dist_all <= reach
        cand[rows[:, None] == cols[None, :]] = False
        ri, ci = np.nonzero(cand)
        if ri.size == 0:
            continue
        i = rows[ri]                       # this fibre
        j = cols[ci]                       # the neighbour

        v1, v2 = v[i], v[j]
        p1a = p1[i]
        p1_to_p3 = p1[j] - p1a
        c2c = pos[j] - pos[i]
        co_radius = R[i] + R[j]
        co_length = L[i] + L[j]
        dist = dist_all[ri, ci]

        FCP = np.cross(v1, v2)
        hit = np.zeros(i.size, dtype=bool)
        pt = np.zeros((i.size, 3))

        # (1) colinear parallel -- upstream's exact == makes this near-dead
        with np.errstate(invalid="ignore", divide="ignore"):
            c2c_unit = c2c / dist[:, None]
        aligned = np.all(c2c_unit == u[i], axis=1) | np.all(c2c_unit == -u[i], axis=1)
        t1 = (np.einsum("ij,ij->i", FCP, FCP) == 0) & aligned & (dist <= co_length)
        hit |= t1
        pt = np.where(t1[:, None], pos[i] + L[i][:, None] * c2c_unit, pt)

        # (2) parallel fibres lying on top of one another
        d1 = np.linalg.norm(_nearest_on_segment(p1a, pos[j], u[j], L[j]), axis=1)
        d2 = np.linalg.norm(_nearest_on_segment(p2[i], pos[j], u[j], L[j]), axis=1)
        d3 = np.linalg.norm(_nearest_on_segment(p1[j], pos[i], u[i], L[i]), axis=1)
        d4 = np.linalg.norm(_nearest_on_segment(p2[j], pos[i], u[i], L[i]), axis=1)
        t2 = (d1 <= co_radius) | (d2 <= co_radius) | (d3 <= co_radius) | (
            (d4 <= co_radius) & ~aligned)          # C precedence, as upstream
        new2 = t2 & ~hit
        hit |= t2
        pt = np.where(new2[:, None], p1a, pt)

        # (3) skew fibres that cross
        with np.errstate(invalid="ignore", divide="ignore"):
            vv = np.einsum("ij,ij->i", v1, v1)
            aa = np.einsum("ij,ij->i", v1, p1_to_p3) / vv
            bb = np.einsum("ij,ij->i", v1, v2) / vv
            cvec = bb[:, None] * v1 - v2
            nvec = p1_to_p3 - aa[:, None] * v1
            cc = np.einsum("ij,ij->i", cvec, cvec)
            t_neighbor = np.einsum("ij,ij->i", cvec, nvec) / cc
            t_this = aa + bb * t_neighbor
            tol = co_radius / co_length
        triple = np.abs(np.einsum("ij,ij->i", p1_to_p3, FCP)) < co_radius
        inrange = ((t_neighbor >= -tol) & (t_neighbor <= 1 + tol)
                   & (t_this >= -tol) & (t_this <= 1 + tol))
        t3 = triple & inrange & np.isfinite(t_this) & np.isfinite(t_neighbor)
        hit |= t3
        pt = np.where(t3[:, None], p1a + t_this[:, None] * v1, pt)

        if hit.any():
            np.add.at(counts, i[hit], 1)
            point[i[hit]] = pt[hit]          # the last crosslink wins, as upstream

    f.crosslinks[:n] = counts
    f.crosslink_point[:n] = point
    return int((counts > 0).sum())


# ---------------------------------------------------------------- stuck cells


def _counters(sim, n: int) -> dict:
    """PhysiMeSS_Cell's stuck_counter and unstuck_counter, kept on the sim.

    Slots follow cell indices, so a slot recycled by a division inherits a
    counter.  That is harmless in the migration models PhysiMeSS is for, where
    cells rarely divide, and it keeps fibres from reaching into Cells.
    """
    cnt = getattr(sim, "_fibre_counters", None)
    if cnt is None or cnt["stuck"].size < n:
        grown = {k: np.zeros(max(n, 2 * (cnt[k].size if cnt else 0)), dtype=np.int64)
                 for k in ("stuck", "unstuck")}
        if cnt:
            for k in grown:
                grown[k][: cnt[k].size] = cnt[k]
        cnt = grown
        sim._fibre_counters = cnt
    return cnt


def _update_stuck(sim):
    """A motile cell that barely moved last step is one step closer to stuck."""
    c = sim.cells
    n = c.n
    stuck = _counters(sim, n)["stuck"]
    moved = np.linalg.norm(c.prev_velocity[:n], axis=1) * sim.model.time.dt_mechanics
    motile = c.is_motile[:n] != 0
    stepped = np.where(moved < param(sim, "fibre_stuck_threshold"), stuck[:n] + 1, 0)
    stuck[:n] = np.where(motile, stepped, stuck[:n])


def forced_motility(sim, dt: float):
    """A cell stuck for STUCK_THRESHOLD steps gets one randomly redirected step.

    Upstream builds this from the motility vector of the *previous* step, since
    it replaces the normal update; here the normal update has already run, so
    the random direction is added to this step's vector instead.  The effect --
    one step in a direction that owes nothing to the migration bias -- is the
    same.
    """
    c = sim.cells
    n = c.n
    cnt = _counters(sim, n)
    stuck, unstuck = cnt["stuck"], cnt["unstuck"]

    newly = np.flatnonzero(stuck[:n] == STUCK_THRESHOLD)
    stuck[newly] = 0
    unstuck[newly] = 1

    forced = np.flatnonzero((unstuck[:n] >= 1) & (unstuck[:n] < UNSTUCK_THRESHOLD + 1)
                            & (c.is_motile[:n] != 0))
    if forced.size:
        theta = 2.0 * np.pi * sim.rng.random(forced.size)
        if sim.model.domain.use_2D:
            sin_phi = np.ones(forced.size)
            cos_phi = np.zeros(forced.size)
        else:
            phi = np.pi * sim.rng.random(forced.size)
            sin_phi, cos_phi = np.sin(phi), np.cos(phi)
        rand = np.stack([np.cos(theta) * sin_phi, np.sin(theta) * sin_phi, cos_phi], axis=1)
        vec = c.motility_vector[forced] + rand
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        vec = np.divide(vec, norm, out=np.zeros_like(vec), where=norm > 1e-16)
        c.motility_vector[forced] = vec * c.motility_speed[forced][:, None]

    bumped = np.flatnonzero((unstuck[:n] >= 1) & (unstuck[:n] < UNSTUCK_THRESHOLD + 1))
    unstuck[bumped] += 1
    unstuck[:n] = np.where(unstuck[:n] == UNSTUCK_THRESHOLD + 1, 0, unstuck[:n])


def _degrade(sim, near_fibre, near_disp):
    """PhysiMeSS_Cell::degrade_fibre: a stuck cell pushing into a fibre may cut it.

    Upstream can degrade every fibre a cell touches in one step; this degrades
    the closest, which is the one the cell is pressing hardest into.
    """
    c = sim.cells
    f = sim.fibres
    enabled = param(sim, "fibre_degradation") > 0.5
    if not enabled.any():
        return
    stuck = _counters(sim, c.n)["stuck"][: c.n]
    ready = np.flatnonzero(enabled & (stuck >= param(sim, "fibre_stuck_time"))
                           & (near_fibre >= 0))
    if ready.size == 0:
        return
    disp = near_disp[ready]
    dist = np.maximum(np.linalg.norm(disp, axis=1), 1e-5)
    toward = -disp / dist[:, None]                     # unit vector cell -> fibre
    pushing = np.einsum("ij,ij->i", toward, c.motility_vector[ready]) >= 0
    roll = sim.rng.random(ready.size)
    cut = ready[pushing & (roll <= param(sim, "fibre_degradation_rate")[ready])]
    if cut.size == 0:
        return
    doomed = np.unique(near_fibre[cut])
    f.remove(doomed)
    _counters(sim, c.n)["stuck"][cut] = 0


# ------------------------------------------------------------------ placement


def assign_orientation(model, ct, n: int, rng, type_name: str = ""):
    """PhysiMeSS_Fibre::assign_fibre_orientation.

    Length is drawn per fibre and halved; orientation is uniform on the circle
    (2-D) or the sphere (3-D) unless `anisotropic_fibres` is set, when it is
    drawn around `fibre_angle`.  The `fibre_vertical` / `fibre_horizontal`
    names are special-cased upstream for its maze models, and are here too.
    """
    cd = getattr(ct, "custom_data", {})

    def g(key):
        return float(cd.get(key, PARAM_DEFAULTS[key]))

    half = rng.normal(g("fibre_length"), g("length_normdist_sd"), n) / 2.0
    radius = g("fibre_radius")

    if model.domain.use_2D:
        if g("anisotropic_fibres") > 0.5:
            theta = rng.normal(g("fibre_angle"), g("angle_normdist_sd"), n)
        else:
            theta = 2.0 * np.pi * rng.random(n)
        u = np.stack([np.cos(theta), np.sin(theta), np.zeros(n)], axis=1)
    else:
        v = rng.normal(size=(n, 3))
        u = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-16)

    low = type_name.lower()
    if "vertical" in low:
        u[:] = (0.0, 1.0, 0.0)
    elif "horizontal" in low:
        u[:] = (1.0, 0.0, 0.0)
    return u, half, radius


def _out_of_bounds(model, centres, u, half):
    d = model.domain
    ends = np.stack([centres - half[:, None] * u, centres + half[:, None] * u], axis=1)
    bad = ((ends[:, :, 0] < d.x_min) | (ends[:, :, 0] > d.x_max)
           | (ends[:, :, 1] < d.y_min) | (ends[:, :, 1] > d.y_max))
    if not d.use_2D:
        # upstream compares z against the x bounds here; that is a typo in
        # PhysiMeSS 1.14.2 and reproducing it would drop valid 3-D fibres
        bad |= (ends[:, :, 2] < d.z_min) | (ends[:, :, 2] > d.z_max)
    return bad.any(axis=1)


def place(model, ct, centres, rng, type_name: str = ""):
    """Orient fibres at `centres`, re-drawing any that poke out of the domain.

    PhysiMeSS gives each fibre ten tries and deletes it if its endpoints still
    leave the domain -- except with `anisotropic_fibres`, where the angle is
    the point of the model, so a fibre that does not fit is dropped at once.
    Returns the fibres that fit, plus how many were discarded.
    """
    centres = np.atleast_2d(np.asarray(centres, dtype=float))
    n = centres.shape[0]
    u, half, radius = assign_orientation(model, ct, n, rng, type_name)
    anisotropic = float(getattr(ct, "custom_data", {}).get("anisotropic_fibres", 0.0)) > 0.5

    bad = _out_of_bounds(model, centres, u, half)
    if not anisotropic:
        for _ in range(10):
            if not bad.any():
                break
            redraw, _h, _r = assign_orientation(model, ct, int(bad.sum()), rng, type_name)
            u[bad] = redraw
            bad = _out_of_bounds(model, centres, u, half)

    keep = ~bad
    return centres[keep], u[keep], half[keep], radius, int(bad.sum())

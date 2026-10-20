"""Simulation state: the microenvironment fields and the cell population.

Both are plain NumPy, struct-of-arrays, owned by Python.  The C++ kernels are
called on these buffers; nothing is hidden in C++ between calls, so a notebook
can read or change any of it at any time.
"""

from __future__ import annotations

import numpy as np

from . import _kernels as K
from .model import Model


class Microenvironment:
    """Substrate densities on a uniform Cartesian mesh, BioFVM's LOD solver."""

    def __init__(self, model: Model):
        self.model = model
        d = model.domain
        self.nx, self.ny, self.nz = d.shape
        self.ns = len(model.substrates)
        self.dx, self.dy, self.dz = d.dx, d.dy, d.dz
        self.origin = (d.x_min, d.y_min, d.z_min if not d.use_2D else d.z_min)
        self.is2d = d.use_2D
        self.voxel_volume = d.dx * d.dy * d.dz

        self.density = np.zeros((self.nz, self.ny, self.nx, self.ns), dtype=np.float64)
        for s, sub in enumerate(model.substrates):
            self.density[..., s] = sub.initial_condition
        self.gradient = np.zeros((self.nz, self.ny, self.nx, self.ns, 3), dtype=np.float64)

        self._setup_dirichlet()
        self._setup_thomas()

    # -- geometry ----------------------------------------------------------
    @property
    def centers(self):
        x = self.model.domain.x_min + (np.arange(self.nx) + 0.5) * self.dx
        y = self.model.domain.y_min + (np.arange(self.ny) + 0.5) * self.dy
        z = self.model.domain.z_min + (np.arange(self.nz) + 0.5) * self.dz
        return x, y, z

    def index(self, name: str) -> int:
        return self.model.substrate_names.index(name)

    def __getitem__(self, name: str) -> np.ndarray:
        """A copy of one substrate's field, shaped (nz, ny, nx)."""
        return self.density[..., self.index(name)].copy()

    # -- boundary conditions ----------------------------------------------
    def _setup_dirichlet(self):
        idx, val, mask = [], [], []
        faces = {
            "xmin": [(0, j, k) for k in range(self.nz) for j in range(self.ny)],
            "xmax": [(self.nx - 1, j, k) for k in range(self.nz) for j in range(self.ny)],
            "ymin": [(i, 0, k) for k in range(self.nz) for i in range(self.nx)],
            "ymax": [(i, self.ny - 1, k) for k in range(self.nz) for i in range(self.nx)],
            "zmin": [(i, j, 0) for j in range(self.ny) for i in range(self.nx)],
            "zmax": [(i, j, self.nz - 1) for j in range(self.ny) for i in range(self.nx)],
        }
        if self.is2d:
            faces.pop("zmin"), faces.pop("zmax")
        per_voxel: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for face, voxels in faces.items():
            for s, sub in enumerate(self.model.substrates):
                enabled, value = sub.dirichlet_faces.get(face, (False, 0.0))
                if not enabled:
                    continue
                for (i, j, k) in voxels:
                    n = (k * self.ny + j) * self.nx + i
                    v, mk = per_voxel.setdefault(
                        n, (np.zeros(self.ns), np.zeros(self.ns, dtype=np.uint8)))
                    v[s], mk[s] = value, 1
        for n, (v, mk) in sorted(per_voxel.items()):
            idx.append(n), val.append(v), mask.append(mk)
        self.dirichlet_idx = np.asarray(idx, dtype=np.int64) if idx else np.zeros(0, np.int64)
        self.dirichlet_val = (np.asarray(val, dtype=np.float64) if val
                              else np.zeros((0, self.ns), np.float64))
        self.dirichlet_mask = (np.asarray(mask, dtype=np.uint8) if mask
                               else np.zeros((0, self.ns), np.uint8))
        self.apply_dirichlet()

    def apply_dirichlet(self):
        if self.dirichlet_idx.size:
            K.apply_dirichlet(self.density, self.dirichlet_idx,
                              np.ascontiguousarray(self.dirichlet_val),
                              np.ascontiguousarray(self.dirichlet_mask))

    # -- solver coefficients (BioFVM's construction, kept in Python so the
    #    numerics can be inspected and tested without the kernel) ----------
    def _setup_thomas(self):
        D = np.array([s.diffusion_coefficient for s in self.model.substrates], dtype=np.float64)
        lam = np.array([s.decay_rate for s in self.model.substrates], dtype=np.float64)
        self._D, self._lambda = D, lam
        self._coeffs = {}
        self._dt = None

    def _build_coeffs(self, dt: float):
        ndim = 2 if self.is2d else 3
        c2 = dt * self._lambda / ndim           # decay split across the directions
        self._coeffs = {}
        for axis, (n, h) in enumerate(((self.nx, self.dx), (self.ny, self.dy), (self.nz, self.dz))):
            if n < 1:
                continue
            c1 = dt * self._D / (h * h)
            denom = np.empty((n, self.ns), dtype=np.float64)
            cc = np.empty((n, self.ns), dtype=np.float64)
            denom[:] = 1.0 + 2.0 * c1 + c2
            denom[0] = 1.0 + c1 + c2
            denom[n - 1] = 1.0 + c1 + c2
            if n == 1:
                denom[0] = 1.0 + c2
            cc[:] = -c1
            cc[0] /= denom[0]
            for i in range(1, n):
                denom[i] += c1 * cc[i - 1]
                cc[i] /= denom[i]
            self._coeffs[axis] = (np.ascontiguousarray(denom), np.ascontiguousarray(cc),
                                  np.ascontiguousarray(c1))
        self._dt = dt

    def step(self, dt: float):
        """One diffusion-decay step: LOD split, Thomas per line, per direction."""
        if self._dt != dt:
            self._build_coeffs(dt)
        axes = (0, 1) if self.is2d else (0, 1, 2)
        for axis in axes:
            if (self.nx, self.ny, self.nz)[axis] < 2:
                continue
            self.apply_dirichlet()
            denom, cc, c1 = self._coeffs[axis]
            K.thomas_sweep(self.density, axis, denom, cc, c1)
        self.apply_dirichlet()

    def compute_gradients(self):
        K.gradients(self.density, self.dx, self.dy, self.dz, self.gradient)


class Cells:
    """The population, struct-of-arrays, with capacity that grows by doubling."""

    _VEC3 = ("position", "velocity", "prev_velocity", "motility_vector", "bias_direction")
    _PER_TYPE = ("live_phagocytosis_rate", "attack_rate", "fusion_rate",
                 "transformation_rate", "immunogenicity", "contact_count",
                 "asymmetric_division_prob", "adhesive_affinity")
    _SCALAR_F = (
        "apoptotic_phagocytosis_rate", "necrotic_phagocytosis_rate",
        "other_dead_phagocytosis_rate", "attack_damage_rate", "attack_duration",
        "damage", "damage_rate", "damage_repair_rate", "total_attack_time",
        "damage_delivered", "contact_live", "contact_dead", "attachment_elastic_constant",
        "contact_apoptotic", "contact_necrotic", "contact_other_dead", "attacking",
        "attack_time_left",
        "attachment_rate", "detachment_rate", "max_attachments",
        "chemotaxis_index", "chemotaxis_direction",
        "cell_BM_adhesion", "cell_BM_repulsion", "contact_bm",
        "volume_total", "volume_fluid", "volume_nuclear_solid", "volume_cytoplasmic_solid",
        "volume_nuclear", "volume_cytoplasmic", "fluid_fraction", "calcified_fraction", "radius",
        "target_solid_nuclear", "target_solid_cytoplasmic", "target_fluid_fraction",
        "target_cyto_to_nuclear", "fluid_change_rate", "cyto_biomass_rate", "nuclear_biomass_rate",
        "calcification_rate", "rupture_volume",
        "elapsed_time_in_phase", "cycle_entry_rate", "apoptosis_rate", "necrosis_rate",
        "motility_speed", "migration_bias", "persistence_time",
        "adhesion", "repulsion", "rel_max_adhesion", "pressure", "time_in_death",
    )
    _SCALAR_I = ("type_idx", "phase", "death_model", "death_phase", "voxel", "neighbors",
                 "ID", "n_attached")
    _SCALAR_B = ("dead", "movable", "is_motile")

    MAX_ATTACHMENTS = 12          # PhysiCell's Mechanics::maximum_number_of_attachments
    MAX_PHASES = 8

    def __init__(self, n_substrates: int, capacity: int = 1024, custom_names=(),
                 n_types: int = 1):
        self.ns = n_substrates
        self.nt = max(int(n_types), 1)
        self.custom_names = list(custom_names)
        self.nc = max(len(self.custom_names), 1)
        self.n = 0
        self._cap = 0
        self._next_id = 0
        self._alloc(capacity)

    def _alloc(self, cap: int):
        old = self._cap
        for name in self._VEC3:
            self._grow(name, (cap, 3), np.float64, old)
        for name in self._SCALAR_F:
            self._grow(name, (cap,), np.float64, old)
        for name in self._SCALAR_I:
            self._grow(name, (cap,), np.int64, old)
        for name in self._SCALAR_B:
            self._grow(name, (cap,), np.uint8, old)
        for name in ("secretion_rate", "secretion_target", "uptake_rate", "net_export_rate",
                     "internalized", "chemotaxis_sensitivity", "sampled",
                     "sampled_gradient_mag", "fraction_released_at_death",
                     "fraction_transferred_when_ingested"):
            self._grow(name, (cap, self.ns), np.float64, old)
        self._grow("custom", (cap, self.nc), np.float64, old)
        # one exit rate per cycle phase, so "exit from cycle phase N" is writable
        self._grow("cycle_exit_rate", (cap, self.MAX_PHASES), np.float64, old)
        attached = np.full((cap, self.MAX_ATTACHMENTS), -1, dtype=np.int64)
        if old:
            attached[:old] = self.attached[:old]
        self.attached = attached
        for name in self._PER_TYPE:
            self._grow(name, (cap, self.nt), np.float64, old)
        self._cap = cap

    def _grow(self, name: str, shape, dtype, old_cap: int):
        new = np.zeros(shape, dtype=dtype)
        if old_cap:
            cur = getattr(self, name)
            new[:old_cap] = cur[:old_cap]
        setattr(self, name, new)

    def reserve(self, extra: int):
        if self.n + extra > self._cap:
            self._alloc(max(2 * self._cap, self.n + extra))

    # -- views over the live cells ---------------------------------------
    def view(self, name: str) -> np.ndarray:
        return getattr(self, name)[: self.n]

    def attach(self, i: int, j: int) -> bool:
        """Attach two cells to each other (PhysiCell's attach_cells)."""
        if i == j:
            return False
        ok = True
        for a, b in ((i, j), (j, i)):
            row = self.attached[a]
            if b in row:
                continue
            free = np.flatnonzero(row < 0)
            if free.size == 0:
                ok = False
                continue
            row[free[0]] = b
            self.n_attached[a] += 1
        return ok

    def detach(self, i: int, j: int):
        for a, b in ((i, j), (j, i)):
            row = self.attached[a]
            hit = np.flatnonzero(row == b)
            if hit.size:
                row[hit] = -1
                self.n_attached[a] = int((row >= 0).sum())

    def detach_all(self, i: int):
        for b in self.attached[i][self.attached[i] >= 0]:
            self.detach(i, int(b))

    def attached_pairs(self):
        """(i, j) for every attachment, each pair appearing from both sides."""
        rows = self.attached[: self.n]
        i, slot = np.nonzero(rows >= 0)
        return i, rows[i, slot]

    def custom_column(self, name: str) -> np.ndarray:
        """A view of one custom-data variable over the live cells."""
        return self.custom[: self.n, self.custom_names.index(name)]

    @property
    def positions(self) -> np.ndarray:
        return self.position[: self.n].copy()

    _ALL_FIELDS = (*_PER_TYPE, "cycle_exit_rate", "secretion_rate", "secretion_target", "uptake_rate", "net_export_rate",
                   "internalized", "chemotaxis_sensitivity", "sampled", "sampled_gradient_mag",
                   "custom", "cycle_exit_rate", "fraction_released_at_death",
                   "fraction_transferred_when_ingested", *_PER_TYPE)

    def add(self, count: int = 1) -> np.ndarray:
        self.reserve(count)
        idx = np.arange(self.n, self.n + count)
        # remove() compacts, so these slots may hold a previous cell's state
        for name in (*self._VEC3, *self._SCALAR_F, *self._SCALAR_I, *self._SCALAR_B,
                     *self._ALL_FIELDS):
            getattr(self, name)[idx] = 0
        self.attached[idx] = -1
        self.n += count
        self.ID[idx] = np.arange(self._next_id, self._next_id + count)
        self._next_id += count
        self.movable[idx] = 1
        return idx

    def remove(self, mask: np.ndarray):
        """Compact the arrays, dropping cells where mask is True."""
        if not mask.any():
            return 0
        keep = np.flatnonzero(~mask)
        k = keep.size
        for name in (*self._VEC3, *self._SCALAR_F, *self._SCALAR_I, *self._SCALAR_B,
                     *self._ALL_FIELDS):
            arr = getattr(self, name)
            arr[:k] = arr[keep]
        # attachments are stored as indices, so compaction has to renumber them
        remap = np.full(self.n, -1, dtype=np.int64)
        remap[keep] = np.arange(k)
        att = self.attached[keep]
        valid = att >= 0
        att[valid] = remap[att[valid]]
        att.sort(axis=1)                       # push the -1s (dropped cells) to the front
        att = att[:, ::-1].copy()              # ... then to the back
        self.attached[:k] = att
        self.n_attached[:k] = (att >= 0).sum(axis=1)
        removed = self.n - k
        self.n = k
        return removed

    def to_dataframe(self, type_names=None):
        import pandas as pd

        d = {
            "ID": self.view("ID").copy(),
            "x": self.view("position")[:, 0].copy(),
            "y": self.view("position")[:, 1].copy(),
            "z": self.view("position")[:, 2].copy(),
            "type": self.view("type_idx").copy(),
            "volume": self.view("volume_total").copy(),
            "radius": self.view("radius").copy(),
            "phase": self.view("phase").copy(),
            "dead": self.view("dead").astype(bool),
            "pressure": self.view("pressure").copy(),
        }
        if type_names is not None:
            d["cell_type"] = [type_names[i] for i in d["type"]]
        return pd.DataFrame(d)

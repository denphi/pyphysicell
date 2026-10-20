"""The simulation: Python owns the loop, C++ owns the inner loops.

The structure mirrors PhysiCell's `main.cpp` plus
`Cell_Container::update_all_cells`, with the same three time scales
(diffusion, mechanics, phenotype) and the same ordering inside a step.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field

import numpy as np

from . import _kernels as K
from . import fibres as fibre_mod
from . import membrane as membrane_mod
from . import cycles as cycle_models
from . import interactions as inter
from . import phenotype as ph
from .model import Model
from .rules import Ruleset
from .signals import Dictionary
from .state import Cells, Microenvironment


@dataclass
class Scalars:
    """Tier-1 artifact: cheap aggregates written *during* the run."""

    times: list[float] = field(default_factory=list)
    counts: dict[str, list[int]] = field(default_factory=dict)
    dead: list[int] = field(default_factory=list)
    births: list[int] = field(default_factory=list)
    deaths: list[int] = field(default_factory=list)
    mean_substrate: dict[str, list[float]] = field(default_factory=dict)

    def to_dataframe(self):
        import pandas as pd

        d = {"time": self.times, "dead": self.dead, "births": self.births, "deaths": self.deaths}
        d.update({f"n_{k}": v for k, v in self.counts.items()})
        d.update({f"mean_{k}": v for k, v in self.mean_substrate.items()})
        return pd.DataFrame(d)


class Simulation:
    def __init__(self, model: Model, seed: int | None = None, strict_rules: bool = False):
        self.model = model
        self.rng = np.random.default_rng(model.random_seed if seed is None else seed)
        self.t = 0.0
        self.microenvironment = Microenvironment(model)
        self.dictionary = Dictionary(model)
        self.ruleset = Ruleset(model, self.dictionary, strict=strict_rules)
        self.cells = Cells(n_substrates=len(model.substrates),
                           custom_names=self.dictionary.custom_names,
                           n_types=len(model.cell_types))
        self.neighbor_idx = None
        self.fibres = None          # set when a model has fibre types (pyphysicell.fibres)
        self.membrane = None        # set to a pyphysicell.membrane.* to enable BM forces
        self._pending_removal: list[int] = []
        self.scalars = Scalars()
        self._births = 0
        self._deaths = 0
        self._next_phenotype = 0.0
        self._next_mechanics = 0.0
        self._last_record_t = None
        self._compile_types()
        self._seed_cells()
        self.wall_time = 0.0
        self._hooks: list[tuple[str, int | None, object]] = []
        self._timers: list[tuple[float, float, object]] = []

    # -- static per-type tables -------------------------------------------
    def _compile_types(self):
        m = self.model
        self.type_index = {c.name: i for i, c in enumerate(m.cell_types)}
        nt = len(m.cell_types)
        ns = len(m.substrates)

        max_phases = 1
        for c in m.cell_types:
            ends = [j for (_i, j) in c.cycle.transition_rates]
            starts = [i for (i, _j) in c.cycle.transition_rates]
            max_phases = max(max_phases, (max(starts + ends) + 1) if starts else 1)
        self.ring_length = np.ones(nt, dtype=np.int64)
        self.cycle_rates = np.zeros((nt, max_phases))
        self.cycle_fixed = np.zeros((nt, max_phases), dtype=bool)
        for i, c in enumerate(m.cell_types):
            if not c.cycle.transition_rates:
                continue
            starts = [s for (s, _e) in c.cycle.transition_rates]
            ends = [e for (_s, e) in c.cycle.transition_rates]
            self.ring_length[i] = max(max(starts), max(ends)) + 1
            for (s, _e), r in c.cycle.transition_rates.items():
                self.cycle_rates[i, s] = r
                self.cycle_fixed[i, s] = c.cycle.fixed_duration.get((s, _e), False)

        self.affinity = np.ones((nt, nt), dtype=np.float64)
        for i, c in enumerate(m.cell_types):
            for other, v in c.mechanics.adhesion_affinities.items():
                if other in self.type_index:
                    self.affinity[i, self.type_index[other]] = v

        def per_type(fn):
            return np.array([fn(c) for c in m.cell_types], dtype=np.float64)

        self.death_params = [
            {  # apoptosis
                "unlysed_fluid": per_type(lambda c: _d(c, "apoptosis", "unlysed_fluid_change_rate", 0.05)),
                "lysed_fluid": per_type(lambda c: _d(c, "apoptosis", "lysed_fluid_change_rate", 0.0)),
                "cyto": per_type(lambda c: _d(c, "apoptosis", "cytoplasmic_biomass_change_rate", 1.0 / 60)),
                "nuclear": per_type(lambda c: _d(c, "apoptosis", "nuclear_biomass_change_rate", 0.0058)),
                "calcification": per_type(lambda c: _d(c, "apoptosis", "calcification_rate", 0.0)),
                "duration0": per_type(lambda c: _dur(c, "apoptosis", 0, 516.0)),
            },
            {  # necrosis
                "unlysed_fluid": per_type(lambda c: _d(c, "necrosis", "unlysed_fluid_change_rate", 0.0112)),
                "lysed_fluid": per_type(lambda c: _d(c, "necrosis", "lysed_fluid_change_rate", 0.00083)),
                "cyto": per_type(lambda c: _d(c, "necrosis", "cytoplasmic_biomass_change_rate", 5.33e-5)),
                "nuclear": per_type(lambda c: _d(c, "necrosis", "nuclear_biomass_change_rate", 0.0022)),
                "calcification": per_type(lambda c: _d(c, "necrosis", "calcification_rate", 0.0)),
                "duration1": per_type(lambda c: _dur(c, "necrosis", 1, 86400.0)),
            },
        ]
        names = self.dictionary.custom_names
        self.custom_defaults = np.zeros((nt, max(len(names), 1)))
        self.custom_conserved = np.zeros(max(len(names), 1), dtype=bool)
        for i, c in enumerate(m.cell_types):
            for k, v in c.custom_data.items():
                if k in names:
                    self.custom_defaults[i, names.index(k)] = v
            for k, cons in c.custom_conserved.items():
                if k in names and cons:
                    self.custom_conserved[names.index(k)] = True

        # interaction rates, per (source type, target type)
        self.inter_tables = {k: np.zeros((nt, nt)) for k in
                             ("live_phagocytosis_rate", "attack_rate", "fusion_rate",
                              "transformation_rate", "immunogenicity",
                              "asymmetric_division_prob")}
        self.inter_scalars = {k: np.zeros(nt) for k in
                              ("apoptotic_phagocytosis_rate", "necrotic_phagocytosis_rate",
                               "other_dead_phagocytosis_rate", "attack_damage_rate",
                               "attack_duration", "damage_rate", "damage_repair_rate")}
        for i, c in enumerate(m.cell_types):
            it = c.interactions
            for key, src in (("live_phagocytosis_rate", it.live_phagocytosis_rates),
                             ("attack_rate", it.attack_rates),
                             ("fusion_rate", it.fusion_rates),
                             ("transformation_rate", it.transformation_rates),
                             ("immunogenicity", it.immunogenicities),
                             ("asymmetric_division_prob", c.cycle.asymmetric_division)):
                for other, v in src.items():
                    if other in self.type_index:
                        self.inter_tables[key][i, self.type_index[other]] = v
            for key in self.inter_scalars:
                self.inter_scalars[key][i] = getattr(it, key)
        # PhysiCell's default immunogenicity is 1 unless the XML says otherwise
        for i, c in enumerate(m.cell_types):
            if not c.interactions.immunogenicities:
                self.inter_tables["immunogenicity"][i, :] = 1.0

        self.chemotaxis_sensitivity = np.zeros((nt, ns))
        self.normalize_each_gradient = np.zeros(nt, dtype=bool)
        for i, c in enumerate(m.cell_types):
            if c.motility.advanced_chemotaxis:
                for s, name in enumerate(m.substrate_names):
                    self.chemotaxis_sensitivity[i, s] = c.motility.sensitivities.get(name, 0.0)
                self.normalize_each_gradient[i] = c.motility.normalize_each_gradient
        # simple chemotaxis is a substrate index and a direction, kept apart from
        # the advanced sensitivities so that "chemotactic response to X" reads
        # what PhysiCell reads: the advanced array only
        self.simple_chemotaxis = np.full(nt, -1, dtype=np.int64)
        self.simple_direction = np.zeros(nt)
        for i, c in enumerate(m.cell_types):
            if (not c.motility.advanced_chemotaxis and c.motility.chemotaxis_enabled
                    and c.motility.chemotaxis_substrate in m.substrate_names):
                self.simple_chemotaxis[i] = m.substrate_names.index(
                    c.motility.chemotaxis_substrate)
                self.simple_direction[i] = c.motility.chemotaxis_direction

        # which phase divides and which doubles the volume targets come from the
        # standard model, because they are not derivable from the XML: Ki67
        # (advanced) divides on exit from phase 1, not on the return to phase 0
        self.growth_phase = np.array(
            [cycle_models.entry_phase(c.cycle.code, int(self.ring_length[i]))
             for i, c in enumerate(m.cell_types)], dtype=np.int64)
        self.division_phase = np.array(
            [cycle_models.division_phase(c.cycle.code, int(self.ring_length[i]))
             for i, c in enumerate(m.cell_types)], dtype=np.int64)
        # PhysiCell computes gradients whenever <calculate_gradients> is on,
        # whether or not anything reads them, so a rule or hook asking for
        # "<substrate> gradient" sees the same value it would there
        self.needs_gradients = bool(m.calculate_gradients)

    # -- population -------------------------------------------------------
    def add_cells(self, positions, cell_type: str):
        positions = np.atleast_2d(np.asarray(positions, dtype=np.float64))
        idx = self.cells.add(positions.shape[0])
        ti = self.type_index[cell_type]
        ct = self.model.cell_type(cell_type)
        c = self.cells
        c.position[idx] = positions
        c.type_idx[idx] = ti
        c.phase[idx] = 0
        c.death_model[idx] = -1
        c.dead[idx] = 0
        c.movable[idx] = 1
        ph.init_volume(c, idx, ct.volume)
        c.cycle_entry_rate[idx] = ct.cycle.entry_rate
        c.apoptosis_rate[idx] = ct.deaths["apoptosis"].death_rate if "apoptosis" in ct.deaths else 0.0
        c.necrosis_rate[idx] = ct.deaths["necrosis"].death_rate if "necrosis" in ct.deaths else 0.0
        c.motility_speed[idx] = ct.motility.speed
        c.migration_bias[idx] = ct.motility.migration_bias
        c.persistence_time[idx] = max(ct.motility.persistence_time, 1e-12)
        c.is_motile[idx] = 1 if ct.motility.enabled else 0
        c.adhesion[idx] = ct.mechanics.cell_cell_adhesion_strength
        c.adhesive_affinity[idx] = self.affinity[ti]
        c.attachment_rate[idx] = ct.mechanics.attachment_rate
        c.detachment_rate[idx] = ct.mechanics.detachment_rate
        c.max_attachments[idx] = min(ct.mechanics.maximum_number_of_attachments,
                                     Cells.MAX_ATTACHMENTS)
        c.cell_BM_adhesion[idx] = getattr(ct.mechanics, "cell_BM_adhesion_strength", 4.0)
        c.cell_BM_repulsion[idx] = getattr(ct.mechanics, "cell_BM_repulsion_strength", 10.0)
        c.cycle_exit_rate[idx] = 0.0
        for (ph_i, _ph_j), rate in ct.cycle.transition_rates.items():
            if ph_i < Cells.MAX_PHASES:
                c.cycle_exit_rate[idx, ph_i] = rate
        c.repulsion[idx] = ct.mechanics.cell_cell_repulsion_strength
        c.rel_max_adhesion[idx] = ct.mechanics.relative_maximum_adhesion_distance
        c.attachment_elastic_constant[idx] = ct.mechanics.attachment_elastic_constant
        c.custom[idx] = self.custom_defaults[ti]
        # BioFVM's defaults: nothing released at death, everything transferred
        # when a cell is eaten
        c.fraction_released_at_death[idx] = 0.0
        c.fraction_transferred_when_ingested[idx] = 1.0
        for key, table in self.inter_tables.items():
            getattr(c, key)[idx] = table[ti]
        for key, vec in self.inter_scalars.items():
            getattr(c, key)[idx] = vec[ti]
        c.damage[idx] = 0.0
        c.total_attack_time[idx] = 0.0
        c.damage_delivered[idx] = 0.0
        for s, name in enumerate(self.model.substrate_names):
            e = ct.secretion.get(name)
            if e:
                c.secretion_rate[idx, s] = e.secretion_rate
                c.secretion_target[idx, s] = e.secretion_target
                c.uptake_rate[idx, s] = e.uptake_rate
                c.net_export_rate[idx, s] = e.net_export_rate
            c.chemotaxis_sensitivity[idx, s] = self.chemotaxis_sensitivity[ti, s]
        c.chemotaxis_index[idx] = self.simple_chemotaxis[ti]
        c.chemotaxis_direction[idx] = self.simple_direction[ti]
        self._apply_distributions(ct, idx)
        return idx

    def _apply_distributions(self, ct, idx):
        """<initial_parameter_distributions>: draw a behavior per cell at setup."""
        for dist in getattr(ct, "distributions", []):
            if not dist.enabled or not self.dictionary.supports_behavior(dist.behavior):
                continue
            p = dist.parameters
            n = len(idx)
            kind = dist.kind
            if kind in ("uniform",):
                values = self.rng.uniform(p.get("min", 0.0), p.get("max", 1.0), n)
            elif kind in ("normal", "gaussian"):
                values = self.rng.normal(p.get("mu", 0.0), p.get("sigma", 1.0), n)
            elif kind in ("log_normal", "lognormal"):
                values = self.rng.lognormal(p.get("mu", 0.0), p.get("sigma", 1.0), n)
            elif kind in ("log10_normal", "log10normal"):
                values = 10.0 ** self.rng.normal(p.get("mu", 0.0), p.get("sigma", 1.0), n)
            else:
                continue
            lo, hi = p.get("lower_bound"), p.get("upper_bound")
            if lo is not None or hi is not None:
                values = np.clip(values, lo if lo is not None else -np.inf,
                                 hi if hi is not None else np.inf)
            self.dictionary.write_behavior(dist.behavior, self.cells, idx, values)

    def _seed_cells(self):
        unknown: dict[str, int] = {}
        placed = 0
        for (x, y, z, name) in self.model.initial_cells:
            if name in self.type_index:
                self.add_cells([[x, y, z]], name)
                placed += 1
            else:
                unknown[name] = unknown.get(name, 0) + 1
        if unknown:
            # never drop cells quietly: a mistyped or mis-parsed type in cells.csv
            # would otherwise produce an empty, plausible-looking simulation
            detail = ", ".join(f"{n} x {k!r}" for k, n in sorted(unknown.items()))
            raise ValueError(
                f"{sum(unknown.values())} initial cells name a cell type this model does "
                f"not define ({detail}); known types are {self.model.cell_type_names}")

    # -- extension points -------------------------------------------------
    def hook(self, slot: str = "phenotype", cell_type: str | None = None):
        """Register a vectorized hook: Python in the loop, arrays not objects.

        Slots, mirroring PhysiCell's Cell_Functions:
          "phenotype"      update_phenotype      -- once per phenotype step
          "custom_rule"    custom_cell_rule      -- once per mechanics step
          "migration_bias" update_migration_bias -- write ctx.bias_direction
          "contact"        contact_function      -- over attached pairs (ctx.i, ctx.j)
          "touch"          -- over touching pairs found by the force pass
          "division"       cell_division_function-- over (parent, daughter) pairs
          "step"           -- once per diffusion step

            @sim.hook("phenotype", cell_type="macrophage")
            def polarise(ctx):
                ctx.behaviors["migration speed"] = 0.2 + 40 * ctx.signals["necrotic debris"]

        Called once per step for the cells of that type (every type if None).
        """
        if slot not in HOOK_SLOTS:
            raise ValueError(f"unknown hook slot {slot!r}; use one of {sorted(HOOK_SLOTS)}")

        def register(fn):
            ti = self.type_index[cell_type] if cell_type is not None else None
            self._hooks.append((slot, ti, fn))
            return fn
        return register

    def every(self, interval: float, fn):
        """Call fn(sim) every `interval` simulated minutes -- treatment protocols,
        measurements, anything that is a schedule rather than a cell behavior."""
        self._timers.append([float(interval), self.t, fn])
        return fn

    def _run_hooks(self, slot: str, dt: float):
        if not self._hooks:
            return
        n = self.cells.n
        if not n:
            return
        types = self.cells.view("type_idx")
        for hslot, ti, fn in self._hooks:
            if hslot != slot:
                continue
            idx = np.arange(n) if ti is None else np.flatnonzero(types == ti)
            if idx.size:
                fn(HookContext(self, idx, dt))

    def _run_contact_hooks(self, dt: float):
        """contact_function over attached pairs, and 'touch' over neighbours.

        PhysiCell calls contact_function for attached cells only; the touching
        pairs found by the force pass are offered separately."""
        c, n = self.cells, self.cells.n
        if n == 0:
            return
        for slot in ("contact", "touch"):
            hooks = [h for h in self._hooks if h[0] == slot]
            if not hooks:
                continue
            if slot == "contact":
                i, j = c.attached_pairs()
            else:
                if self.neighbor_idx is None:
                    continue
                cap = self.neighbor_idx.shape[1]
                counts = np.minimum(c.view("neighbors"), cap)
                if counts.max() == 0:
                    continue
                slots = np.arange(cap)[None, :] < counts[:, None]
                i = np.repeat(np.arange(n), counts)
                j = self.neighbor_idx[:n][slots]
            if i.size == 0:
                continue
            self._dispatch_pairs(hooks, i, j, dt)

    def _dynamic_attachments(self, dt: float):
        """PhysiCell's dynamic_spring_attachments: cells attach to and detach from
        their neighbours at per-cell rates."""
        c, n = self.cells, self.cells.n
        if n == 0 or self.neighbor_idx is None:
            return
        if self.model.disable_automated_spring_adhesions:
            return
        if not (c.attachment_rate[:n].any() or c.detachment_rate[:n].any()):
            return
        # detach first, so a freed slot can be refilled this step
        i, j = c.attached_pairs()
        if i.size:
            drop = self.rng.random(i.size) < c.detachment_rate[i] * dt
            for k in np.flatnonzero(drop):
                c.detach(int(i[k]), int(j[k]))
        cap = self.neighbor_idx.shape[1]
        counts = np.minimum(c.view("neighbors"), cap)
        if counts.max() == 0:
            return
        slots = np.arange(cap)[None, :] < counts[:, None]
        ii = np.repeat(np.arange(n), counts)
        jj = self.neighbor_idx[:n][slots]
        room = (c.n_attached[ii] < c.max_attachments[ii]) & \
               (c.n_attached[jj] < c.max_attachments[jj])
        take = room & (self.rng.random(ii.size) < c.attachment_rate[ii] * dt)
        for k in np.flatnonzero(take):
            a, b = int(ii[k]), int(jj[k])
            if c.n_attached[a] < c.max_attachments[a] and c.n_attached[b] < c.max_attachments[b]:
                c.attach(a, b)

    def _dispatch_pairs(self, hooks, i, j, dt):
        c = self.cells
        for _slot, ti, fn in hooks:
            sel = slice(None) if ti is None else np.flatnonzero(c.type_idx[i] == ti)
            if ti is not None and len(sel) == 0:
                continue
            fn(PairContext(self, i[sel], j[sel], dt))

    def _run_division_hooks(self, parent: np.ndarray, child: np.ndarray, dt: float):
        for slot, ti, fn in self._hooks:
            if slot != "division":
                continue
            sel = (slice(None) if ti is None
                   else np.flatnonzero(self.cells.type_idx[parent] == ti))
            if ti is not None and len(sel) == 0:
                continue
            fn(PairContext(self, parent[sel], child[sel], dt, names=("parent", "daughter")))

    def branch(self, seed: int | None = None):
        """An independent copy of this simulation, including its cells and fields.

        PhysiCell keeps its state in process globals, so a second simulation in
        one process is impossible there; here the state is Python objects, so a
        counterfactual is a deepcopy."""
        import copy as _copy

        other = _copy.deepcopy(self)
        if seed is not None:
            other.rng = np.random.default_rng(seed)
        return other

    def save(self, path):
        """Checkpoint the whole simulation (cells, fields, time, RNG) to a file."""
        import pickle

        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load(path):
        import pickle

        with open(path, "rb") as fh:
            return pickle.load(fh)

    # -- one step ---------------------------------------------------------
    def _sample_fields(self):
        c, me = self.cells, self.microenvironment
        n = c.n
        if n == 0:
            return
        K.voxel_of(c.position[:n], me.origin, (me.dx, me.dy, me.dz),
                   (me.nx, me.ny, me.nz), c.voxel[:n])
        grad = me.gradient if self.needs_gradients else None
        gbuf = np.zeros((n, me.ns, 3)) if self.needs_gradients else None
        K.sample_at_cells(me.density, grad, c.voxel[:n], c.sampled[:n], gbuf)
        if gbuf is not None:
            c.sampled_gradient_mag[:n] = np.linalg.norm(gbuf, axis=2)
            self._gradients = gbuf

    def _mechanics(self, dt: float):
        c, me = self.cells, self.microenvironment
        n = c.n
        if n == 0:
            return
        h = max(30.0, float(2.0 * (c.rel_max_adhesion[:n] * c.radius[:n]).max()))
        d = self.model.domain
        gx = max(1, int(np.ceil((d.x_max - d.x_min) / h)))
        gy = max(1, int(np.ceil((d.y_max - d.y_min) / h)))
        gz = 1 if d.use_2D else max(1, int(np.ceil((d.z_max - d.z_min) / h)))
        head = np.full(gx * gy * gz, -1, dtype=np.int64)
        nxt = np.full(n, -1, dtype=np.int64)
        origin = (d.x_min, d.y_min, d.z_min)
        pos = np.ascontiguousarray(c.position[:n])
        K.build_cell_list(pos, origin, h, (gx, gy, gz), head, nxt)
        vel = np.zeros((n, 3))
        press = np.zeros(n)
        nbrs = np.zeros(n, dtype=np.int64)
        cap = 24
        self.neighbor_idx = np.full((n, cap), -1, dtype=np.int64)
        K.forces(pos, np.ascontiguousarray(c.radius[:n]),
                 np.ascontiguousarray(c.rel_max_adhesion[:n]),
                 np.ascontiguousarray(c.adhesion[:n]), np.ascontiguousarray(c.repulsion[:n]),
                 np.ascontiguousarray(c.type_idx[:n]),
                 np.ascontiguousarray(c.adhesive_affinity[:n]),
                 np.ascontiguousarray(c.movable[:n]), origin, h, (gx, gy, gz), head, nxt,
                 d.use_2D, vel, press, nbrs, self.neighbor_idx, cap)
        c.pressure[:n] = press
        c.neighbors[:n] = nbrs
        self._contact_counts()
        self._dynamic_attachments(dt)
        self._run_contact_hooks(dt)
        inter.run(self, dt)
        self._flush_removals()
        self._spring_forces(vel)
        if self.membrane is not None:
            c.velocity[:n] = vel
            membrane_mod.apply(self)
            vel = c.velocity[:n]
        if self.fibres is not None and self.fibres.n:
            c.velocity[:n] = vel
            fibre_mod.step(self, dt)
            vel = c.velocity[:n]
        self._run_hooks("custom_rule", dt)
        self._update_motility(dt)
        vel += c.motility_vector[:n]
        c.velocity[:n] = vel
        K.update_positions(c.position[:n], c.velocity[:n], c.prev_velocity[:n],
                           np.ascontiguousarray(c.movable[:n]), dt, d.use_2D)
        if self.model.virtual_wall:
            self._keep_inside()

    def _update_motility(self, dt: float):
        c = self.cells
        n = c.n
        motile = np.flatnonzero((c.is_motile[:n] == 1) & (c.dead[:n] == 0))
        c.motility_vector[np.flatnonzero(c.is_motile[:n] == 0)] = 0.0
        if motile.size == 0:
            return
        p = dt / np.maximum(c.persistence_time[motile], 1e-12)
        redo = motile[(self.rng.random(motile.size) < p) | (c.persistence_time[motile] < dt)]
        if redo.size == 0:
            return
        if self.model.domain.use_2D:
            ang = self.rng.random(redo.size) * 2.0 * np.pi
            rand = np.stack([np.cos(ang), np.sin(ang), np.zeros(redo.size)], axis=1)
        else:
            rand = self.rng.normal(size=(redo.size, 3))
            rand /= np.linalg.norm(rand, axis=1, keepdims=True)

        bias_dir = np.zeros((redo.size, 3))
        if any(h[0] == "migration_bias" for h in self._hooks):
            self._run_hooks("migration_bias", dt)
            bias_dir = c.bias_direction[redo].copy()
            nrm = np.linalg.norm(bias_dir, axis=1, keepdims=True)
            bias_dir = np.divide(bias_dir, nrm, out=np.zeros_like(bias_dir), where=nrm > 1e-16)
        elif self.needs_gradients and hasattr(self, "_gradients"):
            g = self._gradients[redo]                      # (k, ns, 3)
            norm_each = self.normalize_each_gradient[c.type_idx[redo]]
            if norm_each.any():
                gn = np.linalg.norm(g, axis=2, keepdims=True)
                g = np.where(norm_each[:, None, None],
                             np.divide(g, gn, out=np.zeros_like(g), where=gn > 1e-16), g)
            sens = c.chemotaxis_sensitivity[redo].copy()   # (k, ns), advanced
            simple = c.chemotaxis_index[redo].astype(np.int64)
            has_simple = simple >= 0
            if has_simple.any():                           # one substrate, one sign
                rows = np.flatnonzero(has_simple)
                sens[rows] = 0.0
                sens[rows, simple[rows]] = c.chemotaxis_direction[redo][rows]
            bias_dir = np.einsum("ks,ksa->ka", sens, g)
            nrm = np.linalg.norm(bias_dir, axis=1, keepdims=True)
            bias_dir = np.divide(bias_dir, nrm, out=np.zeros_like(bias_dir), where=nrm > 1e-16)

        bias = c.migration_bias[redo][:, None]
        v = bias * bias_dir + (1.0 - bias) * rand
        nrm = np.linalg.norm(v, axis=1, keepdims=True)
        v = np.divide(v, nrm, out=np.zeros_like(v), where=nrm > 1e-16)
        c.motility_vector[redo] = v * c.motility_speed[redo][:, None]

    def _spring_forces(self, vel):
        """standard_elastic_contact_function over attached pairs."""
        c = self.cells
        i, j = c.attached_pairs()
        if i.size == 0:
            return
        disp = c.position[j] - c.position[i]
        aff = self.affinity
        k_i = c.attachment_elastic_constant[i] * aff[c.type_idx[i], c.type_idx[j]]
        k_j = c.attachment_elastic_constant[j] * aff[c.type_idx[j], c.type_idx[i]]
        k = np.sqrt(np.maximum(k_i * k_j, 0.0))
        np.add.at(vel, i, k[:, None] * disp)

    def _contact_counts(self):
        """Who is touching whom, per cell type -- the 'contact with X' signals."""
        c, n = self.cells, self.cells.n
        for arr in ("contact_count", "contact_live", "contact_dead", "contact_apoptotic",
                    "contact_necrotic", "contact_other_dead"):
            getattr(c, arr)[:n] = 0.0
        if self.neighbor_idx is None or n == 0:
            return
        cap = self.neighbor_idx.shape[1]
        counts = np.minimum(c.view("neighbors"), cap)
        if counts.max() == 0:
            return
        slots = np.arange(cap)[None, :] < counts[:, None]
        i = np.repeat(np.arange(n), counts)
        j = self.neighbor_idx[:n][slots]
        np.add.at(c.contact_count, (i, c.type_idx[j]), 1.0)
        dead_j = c.dead[j] == 1
        np.add.at(c.contact_live, i[~dead_j], 1.0)
        np.add.at(c.contact_dead, i[dead_j], 1.0)
        apop = dead_j & (c.death_model[j] == 0)
        necr = dead_j & (c.death_model[j] == 1)
        np.add.at(c.contact_apoptotic, i[apop], 1.0)
        np.add.at(c.contact_necrotic, i[necr], 1.0)
        np.add.at(c.contact_other_dead, i[dead_j & ~(apop | necr)], 1.0)

    def convert_cell(self, index: int, type_idx: int):
        """PhysiCell's convert_to_cell_definition: the cell keeps its position and
        size, and takes the new type's parameters."""
        c = self.cells
        name = self.model.cell_types[type_idx].name
        ct = self.model.cell_type(name)
        c.type_idx[index] = type_idx
        c.cycle_entry_rate[index] = ct.cycle.entry_rate
        c.apoptosis_rate[index] = ct.deaths["apoptosis"].death_rate if "apoptosis" in ct.deaths else 0.0
        c.necrosis_rate[index] = ct.deaths["necrosis"].death_rate if "necrosis" in ct.deaths else 0.0
        c.motility_speed[index] = ct.motility.speed
        c.migration_bias[index] = ct.motility.migration_bias
        c.persistence_time[index] = max(ct.motility.persistence_time, 1e-12)
        c.is_motile[index] = 1 if ct.motility.enabled else 0
        c.adhesion[index] = ct.mechanics.cell_cell_adhesion_strength
        c.adhesive_affinity[index] = self.affinity[type_idx]
        c.attachment_rate[index] = ct.mechanics.attachment_rate
        c.detachment_rate[index] = ct.mechanics.detachment_rate
        c.repulsion[index] = ct.mechanics.cell_cell_repulsion_strength
        c.rel_max_adhesion[index] = ct.mechanics.relative_maximum_adhesion_distance
        c.attachment_elastic_constant[index] = ct.mechanics.attachment_elastic_constant
        for s_i, sname in enumerate(self.model.substrate_names):
            e = ct.secretion.get(sname)
            if e:
                c.secretion_rate[index, s_i] = e.secretion_rate
                c.secretion_target[index, s_i] = e.secretion_target
                c.uptake_rate[index, s_i] = e.uptake_rate
                c.net_export_rate[index, s_i] = e.net_export_rate
            c.chemotaxis_sensitivity[index, s_i] = self.chemotaxis_sensitivity[type_idx, s_i]
        for key, table in self.inter_tables.items():
            getattr(c, key)[index] = table[type_idx]
        for key, vec in self.inter_scalars.items():
            getattr(c, key)[index] = vec[type_idx]
        self.transformations = getattr(self, "transformations", 0) + 1

    def release_internalized(self, idx):
        """BioFVM's release_internalized_substrates: a dying cell's internal
        pool goes back into the voxel it occupies."""
        c, me = self.cells, self.microenvironment
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            return
        released = c.internalized[idx] * c.fraction_released_at_death[idx] / me.voxel_volume
        if released.any():
            flat = me.density.reshape(-1, me.ns)
            np.add.at(flat, c.voxel[idx], released)
        c.internalized[idx] = 0.0

    def _flush_removals(self):
        if not self._pending_removal:
            return
        doomed = np.asarray(sorted(set(self._pending_removal)), dtype=np.int64)
        self.release_internalized(doomed)
        mask = np.zeros(self.cells.n, dtype=bool)
        mask[doomed] = True
        self._pending_removal.clear()
        self._deaths += int(mask.sum())
        self.cells.remove(mask)
        self.neighbor_idx = None

    def _keep_inside(self):
        d, c, n = self.model.domain, self.cells, self.cells.n
        np.clip(c.position[:n, 0], d.x_min + 1e-6, d.x_max - 1e-6, out=c.position[:n, 0])
        np.clip(c.position[:n, 1], d.y_min + 1e-6, d.y_max - 1e-6, out=c.position[:n, 1])
        if not d.use_2D:
            np.clip(c.position[:n, 2], d.z_min + 1e-6, d.z_max - 1e-6, out=c.position[:n, 2])
        else:
            c.position[:n, 2] = 0.0

    def _divide(self, idx: np.ndarray):
        if idx.size == 0:
            return
        c = self.cells
        if self.model.domain.use_2D:
            ang = self.rng.random(idx.size) * 2.0 * np.pi
            u = np.stack([np.cos(ang), np.sin(ang), np.zeros(idx.size)], axis=1)
        else:
            u = self.rng.normal(size=(idx.size, 3))
            u /= np.linalg.norm(u, axis=1, keepdims=True)
        offset = u * c.radius[idx][:, None]

        child = c.add(idx.size)
        # one canonical field list, so a new per-cell array cannot be forgotten here
        for name in (*Cells._VEC3, *Cells._SCALAR_F, *Cells._SCALAR_I, *Cells._SCALAR_B,
                     *Cells._ALL_FIELDS):
            if name == "ID":
                continue
            getattr(c, name)[child] = getattr(c, name)[idx]
        c.position[child] = c.position[idx] + offset
        c.position[idx] -= 0.5 * offset
        # standard_asymmetric_division_function: roulette over the daughter's type
        probs = c.asymmetric_division_prob[idx]
        if probs.any():
            r = self.rng.random(idx.size)
            cum = np.cumsum(probs, axis=1)
            pick = np.argmax(cum >= r[:, None], axis=1)
            hit = (cum[:, -1] >= r) & probs.any(axis=1)
            for k in np.flatnonzero(hit):
                if int(pick[k]) != int(c.type_idx[child[k]]):
                    self.convert_cell(int(child[k]), int(pick[k]))

        ph.halve_volume(c, idx)
        ph.halve_volume(c, child)
        c.internalized[idx] *= 0.5          # PhysiCell divides internalized substrates evenly
        c.internalized[child] = c.internalized[idx]
        if self.custom_conserved.any():     # conserved custom quantities split too (1.10.3)
            cons = np.flatnonzero(self.custom_conserved)
            c.custom[np.ix_(idx, cons)] *= 0.5
            c.custom[np.ix_(child, cons)] = c.custom[np.ix_(idx, cons)]
        c.elapsed_time_in_phase[child] = 0.0
        c.elapsed_time_in_phase[idx] = 0.0
        self._run_division_hooks(np.asarray(idx), np.asarray(child), 0.0)
        self._births += idx.size
        if self.model.virtual_wall:
            self._keep_inside()

    def step(self):
        """One diffusion step, with mechanics and phenotype when they are due."""
        m, c = self.model, self.cells
        dt = m.time.dt_diffusion

        self.microenvironment.step(dt)
        self._sample_fields()
        if c.n:
            K.secretion_uptake(self.microenvironment.density, c.voxel[:c.n],
                               np.ascontiguousarray(c.volume_total[:c.n]),
                               np.ascontiguousarray(c.secretion_rate[:c.n]),
                               np.ascontiguousarray(c.uptake_rate[:c.n]),
                               np.ascontiguousarray(c.secretion_target[:c.n]),
                               np.ascontiguousarray(c.net_export_rate[:c.n]),
                               self.microenvironment.voxel_volume, dt,
                               np.ascontiguousarray(c.internalized[:c.n])
                               if self.model.track_internalized_substrates else None)

        if self.t + 1e-12 >= self._next_phenotype:
            dtp = m.time.dt_phenotype
            self._next_phenotype += dtp
            if c.n:
                self.ruleset.apply(c, self.type_index, self.t)
                self._run_hooks("phenotype", dtp)
                ph.update_volume(c, dtp)
                newly_dead = ph.check_for_death(c, self.rng, dtp, self.death_params)
                self._deaths += newly_dead.size
                divides = ph.advance_cycle(c, self.rng, dtp, self.ring_length,
                                           self.cycle_fixed, self.cycle_rates,
                                           self.growth_phase, self.division_phase)
                self._divide(divides)
                remove = ph.advance_death(c, dtp, self.death_params)
                if remove.size:
                    self.release_internalized(remove)
                    mask = np.zeros(c.n, dtype=bool)
                    mask[remove] = True
                    c.remove(mask)

        if self.t + 1e-12 >= self._next_mechanics:
            self._next_mechanics += m.time.dt_mechanics
            if self.needs_gradients:
                self.microenvironment.compute_gradients()
                self._sample_fields()
            self._mechanics(m.time.dt_mechanics)

        self._run_hooks("step", dt)
        for timer in self._timers:
            interval, last, fn = timer
            if self.t + 1e-9 >= last + interval:
                timer[1] = self.t
                fn(self)
        self.t += dt

    def run(self, until: float, record_every: float | None = None, progress: bool = False,
            stop_when=None):
        """Advance to `until` (minutes), recording tier-1 scalars as it goes.

        `stop_when(sim) -> bool` ends the run early; the executable has no such
        thing, so a question like "when does this population collapse?" means
        running to max_time and reading the files afterwards."""
        record_every = record_every or self.model.save.full_interval
        next_record = self.t
        t0 = _time.perf_counter()
        while self.t < until - 1e-9:
            if self.t + 1e-9 >= next_record:
                self.record()
                next_record += record_every
                if progress:
                    print(f"  t={self.t:8.1f} min  cells={self.cells.n:6d}", flush=True)
            self.step()
            if stop_when is not None and stop_when(self):
                self.stopped_early = True
                break
        self.record()
        self.wall_time += _time.perf_counter() - t0
        return self.scalars

    def record(self):
        c, me = self.cells, self.microenvironment
        if self._last_record_t is not None and abs(self.t - self._last_record_t) < 1e-9:
            return          # chained run() calls must not duplicate a time point
        self._last_record_t = self.t
        self.scalars.times.append(self.t)
        for name, i in self.type_index.items():
            alive = int(np.count_nonzero((c.view("type_idx") == i) & (c.view("dead") == 0)))
            self.scalars.counts.setdefault(name, []).append(alive)
        self.scalars.dead.append(int(np.count_nonzero(c.view("dead") == 1)))
        self.scalars.births.append(self._births)
        self.scalars.deaths.append(self._deaths)
        for s, name in enumerate(self.model.substrate_names):
            self.scalars.mean_substrate.setdefault(name, []).append(float(me.density[..., s].mean()))

    # -- convenience ------------------------------------------------------
    def dataframe(self):
        return self.cells.to_dataframe(self.model.cell_type_names)

    def __repr__(self):
        return (f"<Simulation {self.model.name!r} t={self.t:.1f} min "
                f"cells={self.cells.n} backend={K.BACKEND}:{K.MAX_THREADS}>")


def _d(ct, model, field, default):
    m = ct.deaths.get(model)
    return getattr(m, field) if m else default


def _dur(ct, model, index, default):
    m = ct.deaths.get(model)
    if not m:
        return default
    return m.phase_durations.get(index, default)


# ---------------------------------------------------------------------------
# What the file-and-executable workflow cannot do.
# ---------------------------------------------------------------------------

HOOK_SLOTS = ("phenotype", "custom_rule", "migration_bias", "contact", "touch",
              "division", "step")


class PairContext:
    """A hook over pairs of cells: touching neighbours, or parent and daughter."""

    def __init__(self, sim, i, j, dt, names=("self", "other")):
        self._sim, self.i, self.j, self.dt = sim, i, j, dt
        self.t = sim.t
        self.n = len(i)
        self.names = names
        self.rng = sim.rng

    @property
    def position(self):
        return self._sim.cells.position[self.i]

    @property
    def other_position(self):
        return self._sim.cells.position[self.j]

    def signals(self, name, which="self"):
        idx = self.i if which == "self" else self.j
        return self._sim.dictionary.signal_values(name, self._sim.cells, idx, self._sim.t)

    def set_behavior(self, name, values, which="self"):
        idx = self.i if which == "self" else self.j
        self._sim.dictionary.write_behavior(name, self._sim.cells, idx,
                                            np.broadcast_to(np.asarray(values, float),
                                                            (len(idx),)))

    def custom(self, name, which="self"):
        c = self._sim.cells
        idx = self.i if which == "self" else self.j
        return c.custom[idx, c.custom_names.index(name)]


class HookContext:
    """The arguments a vectorized hook gets: whole-population arrays.

    A hook is called once per phenotype (or diffusion) step for the cells of one
    type, not once per cell, so the Python cost is one call per step rather than
    one per cell.  Reads and writes go through the same signal/behavior names the
    rules use.
    """

    def __init__(self, sim, idx, dt):
        self._sim, self.idx, self.dt = sim, idx, dt
        self.t = sim.t
        self.n = idx.size
        self.signals = _SignalView(sim, idx)
        self.behaviors = _BehaviorView(sim, idx)
        self.rng = sim.rng

    @property
    def position(self):
        return self._sim.cells.position[self.idx]

    @property
    def volume(self):
        return self._sim.cells.volume_total[self.idx]

    @property
    def pressure(self):
        return self._sim.cells.pressure[self.idx]

    @property
    def dead(self):
        return self._sim.cells.dead[self.idx] == 1

    def base(self, name: str):
        """The behavior's value in each cell's definition -- PhysiCell's
        get_single_base_behavior, which custom functions scale from."""
        model = self._sim.model
        per_type = np.array([self._sim.dictionary.base_behavior(ct.name, name)
                             for ct in model.cell_types])
        return per_type[self._sim.cells.type_idx[self.idx]]

    @property
    def bias_direction(self):
        """Write a (n, 3) direction here from a 'migration_bias' hook."""
        return self._sim.cells.bias_direction[self.idx]

    @bias_direction.setter
    def bias_direction(self, value):
        self._sim.cells.bias_direction[self.idx] = value

    def kill(self, mask, model: str = "apoptosis"):
        """Send the selected cells into a death model, as PhysiCell's start_death."""
        target = self.idx[np.asarray(mask, dtype=bool)]
        if target.size:
            ph.start_death(self._sim.cells, target, 0 if model == "apoptosis" else 1,
                           self._sim.death_params)


class _SignalView:
    def __init__(self, sim, idx):
        self._sim, self._idx = sim, idx

    def __getitem__(self, name):
        return self._sim.dictionary.signal_values(name, self._sim.cells, self._idx, self._sim.t)

    def __contains__(self, name):
        return self._sim.dictionary.supports_signal(name)


class _BehaviorView:
    def __init__(self, sim, idx):
        self._sim, self._idx = sim, idx

    def __getitem__(self, name):
        return self._sim.dictionary.read_behavior(name, self._sim.cells, self._idx)

    def __setitem__(self, name, values):
        self._sim.dictionary.write_behavior(name, self._sim.cells, self._idx,
                                            np.broadcast_to(np.asarray(values, dtype=float),
                                                            (self._idx.size,)))

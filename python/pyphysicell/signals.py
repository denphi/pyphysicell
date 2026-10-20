"""The signal and behavior dictionary.

As in PhysiCell, the vocabulary is derived from the model: the substrates and
cell types present decide which signal and behavior names exist.  Unlike
PhysiCell, it can be listed, searched and validated without running anything.
"""

from __future__ import annotations

import numpy as np

from .model import Model

# Behaviors this MVP can actually apply, as name -> (array, column or None).
_SIMPLE_BEHAVIORS = {
    "phagocytose apoptotic cell": "apoptotic_phagocytosis_rate",
    "phagocytose necrotic cell": "necrotic_phagocytosis_rate",
    "phagocytose other dead cell": "other_dead_phagocytosis_rate",
    "phagocytose dead cell": "other_dead_phagocytosis_rate",
    "attack damage rate": "attack_damage_rate",
    "attack duration": "attack_duration",
    "damage rate": "damage_rate",
    "damage repair rate": "damage_repair_rate",
    "cycle entry": "cycle_entry_rate",
    "apoptosis": "apoptosis_rate",
    "necrosis": "necrosis_rate",
    "migration speed": "motility_speed",
    "migration bias": "migration_bias",
    "migration persistence time": "persistence_time",
    "cell-cell adhesion": "adhesion",
    "cell-cell repulsion": "repulsion",
    "relative maximum adhesion distance": "rel_max_adhesion",
    "cell-cell adhesion elastic constant": "attachment_elastic_constant",
    "attachment elastic constant": "attachment_elastic_constant",
    "cell-BM adhesion": "cell_BM_adhesion",
    "cell-BM repulsion": "cell_BM_repulsion",
    "cell attachment rate": "attachment_rate",
    "cell detachment rate": "detachment_rate",
    "maximum number of cell attachments": "max_attachments",
    "is_movable": "movable",
    "movable": "movable",
}
_PER_SUBSTRATE_BEHAVIORS = {
    "{s} secretion": "secretion_rate",
    "{s} secretion target": "secretion_target",
    "{s} uptake": "uptake_rate",
    "{s} export": "net_export_rate",
    "chemotactic response to {s}": "chemotaxis_sensitivity",
}
_SIMPLE_SIGNALS = ("pressure", "volume", "time", "dead", "apoptotic", "necrotic",
                   "damage", "total attack time", "damage delivered",
                   "contact with live cell", "contact with dead cell",
                   "contact with basement membrane",
                   "contact with apoptotic cell", "contact with necrotic cell",
                   "contact with other dead cell", "attacking",
                   "number of attached cells")
_PER_SUBSTRATE_SIGNALS = ("{s}", "intracellular {s}", "{s} gradient")


class Dictionary:
    """Signal and behavior names for one model, plus the code that reads/writes them."""

    def __init__(self, model: Model):
        self.model = model
        self.substrates = model.substrate_names
        self.cell_types = model.cell_type_names

        self.custom_names: list[str] = []
        for ct in model.cell_types:
            for k in ct.custom_data:
                if k not in self.custom_names:
                    self.custom_names.append(k)

        self.signals: list[str] = []
        for pat in _PER_SUBSTRATE_SIGNALS:
            self.signals += [pat.format(s=s) for s in self.substrates]
        self.signals += list(_SIMPLE_SIGNALS)
        self.signals += [f"custom:{k}" for k in self.custom_names]
        self.signals += [f"contact with {c}" for c in self.cell_types]

        # per-target-type behaviors
        self._per_type_behaviors = {
            "phagocytose {c}": "live_phagocytosis_rate",
            "attack {c}": "attack_rate",
            "fuse to {c}": "fusion_rate",
            "transform to {c}": "transformation_rate",
            "immunogenicity to {c}": "immunogenicity",
            "asymmetric division to {c}": "asymmetric_division_prob",
            "adhesive affinity to {c}": "adhesive_affinity",
            # PhysiCell's canonical name; "transform to" is its synonym
            "transition to {c}": "transformation_rate",
        }
        # named but not implemented in this MVP; listed so models can be checked
        self.unsupported_signals = ["attacking", "attack ack", "is movable",
                                    "contact with basement membrane"]
        self.unsupported_signals = [s for s in self.unsupported_signals if s not in self.signals]

        self.behaviors: list[str] = list(_SIMPLE_BEHAVIORS)
        self.behaviors += [f"custom:{k}" for k in self.custom_names]
        max_phase = max((max((i for (i, _j) in ct.cycle.transition_rates), default=0)
                         for ct in model.cell_types), default=0)
        # PhysiCell exposes one per phase of the largest standard cycle model
        self.behaviors += [f"exit from cycle phase {i}" for i in range(max(max_phase + 1, 6))]
        for pat in ("phagocytose {c}", "attack {c}", "fuse to {c}", "transform to {c}",
                    "transition to {c}", "immunogenicity to {c}",
                    "asymmetric division to {c}", "adhesive affinity to {c}"):
            self.behaviors += [pat.format(c=c) for c in self.cell_types]
        for pat in _PER_SUBSTRATE_BEHAVIORS:
            self.behaviors += [pat.format(s=s) for s in self.substrates]
        self.unsupported_behaviors = ["cell attachment rate", "cell detachment rate",
                                      "is movable", "cell BM adhesion", "cell BM repulsion"]

    # -- introspection ----------------------------------------------------
    def search(self, text: str, what: str = "both") -> list[str]:
        t = text.lower()
        out = []
        if what in ("both", "signals"):
            out += [s for s in self.signals if t in s.lower()]
        if what in ("both", "behaviors"):
            out += [b for b in self.behaviors if t in b.lower()]
        return out

    def supports_signal(self, name: str) -> bool:
        return name in self.signals

    def supports_behavior(self, name: str) -> bool:
        return name in self.behaviors

    # -- reading signals over a population --------------------------------
    def signal_values(self, name: str, cells, idx: np.ndarray, t: float) -> np.ndarray:
        """Values of one signal for the cells at `idx`. Always a fresh array."""
        if name in self.substrates:
            return cells.sampled[idx, self.substrates.index(name)]
        if name.startswith("intracellular "):
            s = self.substrates.index(name[len("intracellular "):])
            vol = np.maximum(cells.volume_total[idx], 1e-16)
            return cells.internalized[idx, s] / vol
        if name.endswith(" gradient"):
            s = self.substrates.index(name[: -len(" gradient")])
            return cells.sampled_gradient_mag[idx, s]
        if name == "pressure":
            return cells.pressure[idx]
        if name == "volume":
            return cells.volume_total[idx]
        if name == "time":
            return np.full(idx.size, float(t))
        if name == "dead":
            return cells.dead[idx].astype(float)
        if name == "apoptotic":
            return ((cells.dead[idx] == 1) & (cells.death_model[idx] == 0)).astype(float)
        if name == "necrotic":
            return ((cells.dead[idx] == 1) & (cells.death_model[idx] == 1)).astype(float)
        if name.startswith("custom:"):
            key = name[len("custom:"):].strip()
            if key in self.custom_names:
                return cells.custom[idx, self.custom_names.index(key)]
        if name == "damage":
            return cells.damage[idx]
        if name == "total attack time":
            return cells.total_attack_time[idx]
        if name == "damage delivered":
            return cells.damage_delivered[idx]
        if name == "contact with live cell":
            return cells.contact_live[idx]
        if name == "contact with dead cell":
            return cells.contact_dead[idx]
        if name == "contact with apoptotic cell":
            return cells.contact_apoptotic[idx]
        if name == "contact with necrotic cell":
            return cells.contact_necrotic[idx]
        if name == "contact with other dead cell":
            return cells.contact_other_dead[idx]
        if name == "attacking":
            return cells.attacking[idx]
        if name == "contact with basement membrane":
            return cells.contact_bm[idx]
        if name == "number of attached cells":
            return cells.n_attached[idx].astype(float)
        if name.startswith("contact with "):
            other = name[len("contact with "):]
            if other in self.cell_types:
                return cells.contact_count[idx, self.cell_types.index(other)]
        raise KeyError(f"signal {name!r} is not supported by this MVP")

    # -- writing behaviors over a population ------------------------------
    def write_behavior(self, name: str, cells, idx: np.ndarray, values: np.ndarray):
        if name.startswith("exit from cycle phase "):
            phase = int(name.rsplit(" ", 1)[1])
            if phase == 0:
                cells.cycle_entry_rate[idx] = values
            cells.cycle_exit_rate[idx, phase] = values
            return
        if name.startswith("custom:"):
            key = name[len("custom:"):].strip()
            if key in self.custom_names:
                cells.custom[idx, self.custom_names.index(key)] = values
                return
        if name in _SIMPLE_BEHAVIORS:
            getattr(cells, _SIMPLE_BEHAVIORS[name])[idx] = values
            return
        for pat, arr in self._per_type_behaviors.items():
            prefix = pat.split("{c}")[0]
            if name.startswith(prefix):
                other = name[len(prefix):]
                if other in self.cell_types:
                    getattr(cells, arr)[idx, self.cell_types.index(other)] = values
                    return
        for pat, arr in _PER_SUBSTRATE_BEHAVIORS.items():
            prefix, suffix = pat.split("{s}")
            if name.startswith(prefix) and name.endswith(suffix):
                sub = name[len(prefix): len(name) - len(suffix) if suffix else None]
                if sub in self.substrates:
                    getattr(cells, arr)[idx, self.substrates.index(sub)] = values
                    return
        raise KeyError(f"behavior {name!r} is not supported by this MVP")

    def read_behavior(self, name: str, cells, idx: np.ndarray) -> np.ndarray:
        """The current value of any behavior, the counterpart of write_behavior.

        PhysiCell's get_single_behavior reads the live phenotype; so does this.
        """
        if name.startswith("exit from cycle phase "):
            phase = int(name.rsplit(" ", 1)[1])
            if phase == 0:
                return cells.cycle_entry_rate[idx]
            return cells.cycle_exit_rate[idx, phase]
        if name.startswith("custom:"):
            key = name[len("custom:"):].strip()
            if key in self.custom_names:
                return cells.custom[idx, self.custom_names.index(key)]
        if name in _SIMPLE_BEHAVIORS:
            return getattr(cells, _SIMPLE_BEHAVIORS[name])[idx].astype(float)
        for pat, arr in _PER_SUBSTRATE_BEHAVIORS.items():
            prefix, suffix = pat.split("{s}")
            if name.startswith(prefix) and name.endswith(suffix):
                sub = name[len(prefix): len(name) - len(suffix) if suffix else None]
                if sub in self.substrates:
                    return getattr(cells, arr)[idx, self.substrates.index(sub)]
        for pat, arr in self._per_type_behaviors.items():
            prefix = pat.split("{c}")[0]
            if name.startswith(prefix):
                other = name[len(prefix):]
                if other in self.cell_types:
                    return getattr(cells, arr)[idx, self.cell_types.index(other)]
        raise KeyError(f"behavior {name!r} cannot be read")

    def base_behavior(self, cell_type: str, name: str) -> float:
        """The behavior's value in the cell definition — the Hill curve's base."""
        ct = self.model.cell_type(cell_type)
        if name == "cycle entry":
            return ct.cycle.entry_rate
        if name == "apoptosis":
            d = ct.deaths.get("apoptosis")
            return d.death_rate if d else 0.0
        if name == "necrosis":
            d = ct.deaths.get("necrosis")
            return d.death_rate if d else 0.0
        if name == "migration speed":
            return ct.motility.speed
        if name == "migration bias":
            return ct.motility.migration_bias
        if name == "migration persistence time":
            return ct.motility.persistence_time
        if name == "cell-cell adhesion":
            return ct.mechanics.cell_cell_adhesion_strength
        if name == "cell-cell repulsion":
            return ct.mechanics.cell_cell_repulsion_strength
        if name == "relative maximum adhesion distance":
            return ct.mechanics.relative_maximum_adhesion_distance
        if name in ("cell-cell adhesion elastic constant", "attachment elastic constant"):
            return ct.mechanics.attachment_elastic_constant
        for pat, field in (("{s} secretion target", "secretion_target"),
                           ("{s} secretion", "secretion_rate"),
                           ("{s} uptake", "uptake_rate"),
                           ("{s} export", "net_export_rate")):
            prefix, suffix = pat.split("{s}")
            if name.startswith(prefix) and name.endswith(suffix) and suffix:
                sub = name[: -len(suffix)]
                if sub in ct.secretion:
                    return getattr(ct.secretion[sub], field)
        if name.startswith("chemotactic response to "):
            return ct.motility.sensitivities.get(name[len("chemotactic response to "):], 0.0)
        if name.startswith("custom:"):
            return ct.custom_data.get(name[len("custom:"):].strip(), 0.0)
        if name.startswith("exit from cycle phase "):
            phase = int(name.rsplit(" ", 1)[1])
            for (i, _j), rate in ct.cycle.transition_rates.items():
                if i == phase:
                    return rate
            return 0.0
        if name in ("cell attachment rate",):
            return ct.mechanics.attachment_rate
        if name in ("cell detachment rate",):
            return ct.mechanics.detachment_rate
        if name == "maximum number of cell attachments":
            return 12.0
        if name == "cell-BM adhesion":
            return getattr(ct.mechanics, "cell_BM_adhesion_strength", 4.0)
        if name == "cell-BM repulsion":
            return getattr(ct.mechanics, "cell_BM_repulsion_strength", 10.0)
        if name in ("is_movable", "movable"):
            return 1.0
        if name.startswith("adhesive affinity to "):
            return ct.mechanics.adhesion_affinities.get(
                name[len("adhesive affinity to "):], 1.0)
        it = ct.interactions
        for key, attr in (("phagocytose apoptotic cell", "apoptotic_phagocytosis_rate"),
                          ("phagocytose necrotic cell", "necrotic_phagocytosis_rate"),
                          ("phagocytose other dead cell", "other_dead_phagocytosis_rate"),
                          ("phagocytose dead cell", "other_dead_phagocytosis_rate"),
                          ("attack damage rate", "attack_damage_rate"),
                          ("attack duration", "attack_duration"),
                          ("damage rate", "damage_rate"),
                          ("damage repair rate", "damage_repair_rate")):
            if name == key:
                return getattr(it, attr)
        for pat, src in (("phagocytose {c}", it.live_phagocytosis_rates),
                         ("attack {c}", it.attack_rates),
                         ("fuse to {c}", it.fusion_rates),
                         ("transform to {c}", it.transformation_rates),
                         ("immunogenicity to {c}", it.immunogenicities),
                         ("asymmetric division to {c}", ct.cycle.asymmetric_division)):
            prefix = pat.split("{c}")[0]
            if name.startswith(prefix) and name[len(prefix):] in self.cell_types:
                return src.get(name[len(prefix):], 1.0 if "immunogenicity" in pat else 0.0)
        raise KeyError(f"no base value known for behavior {name!r}")

"""The model: a plain-data description of a PhysiCell simulation.

Flat, typed mirrors of PhysiCell's XML, not of its 20-class Phenotype graph.
Every value is a float in PhysiCell's units (minutes, microns, and each
substrate's own concentration unit).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass
class Domain:
    x_min: float = -500.0
    x_max: float = 500.0
    y_min: float = -500.0
    y_max: float = 500.0
    z_min: float = -10.0
    z_max: float = 10.0
    dx: float = 20.0
    dy: float = 20.0
    dz: float = 20.0
    use_2D: bool = True

    @property
    def shape(self) -> tuple[int, int, int]:
        nx = max(1, int(round((self.x_max - self.x_min) / self.dx)))
        ny = max(1, int(round((self.y_max - self.y_min) / self.dy)))
        nz = 1 if self.use_2D else max(1, int(round((self.z_max - self.z_min) / self.dz)))
        return nx, ny, nz

    @property
    def voxel_volume(self) -> float:
        return self.dx * self.dy * (self.dz if not self.use_2D else self.dz)


@dataclass
class Time:
    dt_diffusion: float = 0.01
    dt_mechanics: float = 0.1
    dt_phenotype: float = 6.0
    max_time: float = 7200.0   # Studio's canonical <overall> default
    time_units: str = "min"
    space_units: str = "micron"


@dataclass
class Substrate:
    name: str
    diffusion_coefficient: float = 100000.0
    decay_rate: float = 0.1
    initial_condition: float = 0.0
    dirichlet_enabled: bool = False
    dirichlet_value: float = 0.0
    # face -> (enabled, value); faces: xmin xmax ymin ymax zmin zmax
    dirichlet_faces: dict[str, tuple[bool, float]] = field(default_factory=dict)

    @property
    def diffusion_length(self) -> float:
        """sqrt(D/lambda): the distance over which the substrate is felt."""
        return float("inf") if self.decay_rate <= 0 else (self.diffusion_coefficient / self.decay_rate) ** 0.5


@dataclass
class Cycle:
    code: int = 5
    name: str = "live"
    # (start_index, end_index) -> rate [1/min]
    transition_rates: dict[tuple[int, int], float] = field(default_factory=dict)
    fixed_duration: dict[tuple[int, int], bool] = field(default_factory=dict)
    # <standard_asymmetric_division>: the daughter's type is drawn from these
    asymmetric_division: dict[str, float] = field(default_factory=dict)

    @property
    def entry_rate(self) -> float:
        """The rate out of phase 0 — what the 'cycle entry' behavior sets."""
        for (i, j), r in self.transition_rates.items():
            if i == 0:
                return r
        return 0.0

    def mean_duration(self) -> float:
        """Closed-form mean time around the cycle, no simulation needed."""
        total = 0.0
        for (_i, _j), r in sorted(self.transition_rates.items()):
            total += float("inf") if r <= 0 else 1.0 / r
        return total


@dataclass
class DeathModel:
    code: int
    name: str
    death_rate: float = 0.0
    phase_durations: dict[int, float] = field(default_factory=dict)
    unlysed_fluid_change_rate: float = 0.05
    lysed_fluid_change_rate: float = 0.0
    cytoplasmic_biomass_change_rate: float = 0.0166667
    nuclear_biomass_change_rate: float = 0.00583333
    calcification_rate: float = 0.0
    relative_rupture_volume: float = 2.0


@dataclass
class Volume:
    total: float = 2494.0
    fluid_fraction: float = 0.75
    nuclear: float = 540.0
    fluid_change_rate: float = 0.05
    cytoplasmic_biomass_change_rate: float = 0.0045
    nuclear_biomass_change_rate: float = 0.0055
    calcified_fraction: float = 0.0
    calcification_rate: float = 0.0
    relative_rupture_volume: float = 2.0


@dataclass
class Mechanics:
    cell_cell_adhesion_strength: float = 0.4
    cell_cell_repulsion_strength: float = 10.0
    relative_maximum_adhesion_distance: float = 1.25
    adhesion_affinities: dict[str, float] = field(default_factory=dict)
    attachment_elastic_constant: float = 0.01
    attachment_rate: float = 0.0
    detachment_rate: float = 0.0
    maximum_number_of_attachments: int = 12
    cell_BM_adhesion_strength: float = 4.0
    cell_BM_repulsion_strength: float = 100.0


@dataclass
class Motility:
    speed: float = 1.0
    persistence_time: float = 1.0
    migration_bias: float = 0.0      # PhysiCell's Motility::Motility()
    enabled: bool = False
    use_2D: bool = False             # restrict_to_2D
    chemotaxis_enabled: bool = False
    chemotaxis_substrate: str = ""
    chemotaxis_direction: float = 1.0
    advanced_chemotaxis: bool = False
    normalize_each_gradient: bool = False
    sensitivities: dict[str, float] = field(default_factory=dict)


@dataclass
class SecretionEntry:
    secretion_rate: float = 0.0
    secretion_target: float = 1.0
    uptake_rate: float = 0.0
    net_export_rate: float = 0.0


@dataclass
class Interactions:
    """<cell_interactions>, <cell_transformations>, <cell_integrity>."""

    apoptotic_phagocytosis_rate: float = 0.0
    necrotic_phagocytosis_rate: float = 0.0
    other_dead_phagocytosis_rate: float = 0.0
    attack_damage_rate: float = 1.0
    attack_duration: float = 30.0   # Cell_Interactions::attack_duration
    damage_rate: float = 0.0
    damage_repair_rate: float = 0.0
    live_phagocytosis_rates: dict[str, float] = field(default_factory=dict)
    attack_rates: dict[str, float] = field(default_factory=dict)
    fusion_rates: dict[str, float] = field(default_factory=dict)
    transformation_rates: dict[str, float] = field(default_factory=dict)
    immunogenicities: dict[str, float] = field(default_factory=dict)


@dataclass
class ParameterDistribution:
    """<initial_parameter_distributions>: a behavior drawn per cell at setup."""

    behavior: str
    kind: str = "uniform"          # uniform | normal | log_normal | log10_normal
    parameters: dict[str, float] = field(default_factory=dict)
    enabled: bool = True
    check_base: bool = False


@dataclass
class CellType:
    name: str
    ID: int = 0
    cycle: Cycle = field(default_factory=Cycle)
    deaths: dict[str, DeathModel] = field(default_factory=dict)
    volume: Volume = field(default_factory=Volume)
    mechanics: Mechanics = field(default_factory=Mechanics)
    motility: Motility = field(default_factory=Motility)
    secretion: dict[str, SecretionEntry] = field(default_factory=dict)
    custom_data: dict[str, float] = field(default_factory=dict)
    # conserved quantities are halved between daughters (PhysiCell 1.10.3)
    custom_conserved: dict[str, bool] = field(default_factory=dict)
    interactions: Interactions = field(default_factory=Interactions)
    distributions: list[ParameterDistribution] = field(default_factory=list)

    def doubling_time(self) -> float:
        return self.cycle.mean_duration()


@dataclass
class Rule:
    """One CBHG hypothesis: signal (increases|decreases) behavior, Hill-shaped."""

    cell_type: str
    signal: str
    direction: str  # "increases" | "decreases"
    behavior: str
    max_response: float
    half_max: float
    hill_power: float
    apply_to_dead: bool = False

    def english(self) -> str:
        return (f"In {self.cell_type} cells, {self.signal} {self.direction} {self.behavior} "
                f"towards {self.max_response:g} with half-max {self.half_max:g} "
                f"and Hill power {self.hill_power:g}.")

    def response(self, signal_values, base_value: float):
        """The behavior value this rule alone produces. Pure function, no simulation."""
        import numpy as np

        s = np.asarray(signal_values, dtype=float)
        h = np.power(np.maximum(s, 0.0) / self.half_max, self.hill_power)
        hill = h / (1.0 + h)
        if self.direction == "increases":
            return base_value + (self.max_response - base_value) * hill
        return base_value + (self.max_response - base_value) * hill


@dataclass
class SaveOptions:
    folder: str = "output"
    full_interval: float = 60.0
    full_enabled: bool = True
    svg_interval: float = 60.0
    svg_enabled: bool = True


@dataclass
class Model:
    domain: Domain = field(default_factory=Domain)
    time: Time = field(default_factory=Time)
    substrates: list[Substrate] = field(default_factory=list)
    cell_types: list[CellType] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    initial_cells: list[tuple[float, float, float, str]] = field(default_factory=list)
    user_parameters: dict[str, object] = field(default_factory=dict)
    save: SaveOptions = field(default_factory=SaveOptions)
    random_seed: int = 0
    virtual_wall: bool = True
    omp_num_threads: int = 1
    calculate_gradients: bool = True
    track_internalized_substrates: bool = False
    disable_automated_spring_adhesions: bool = False
    legacy_division_orientation: bool = False
    name: str = "model"

    # -- lookups ----------------------------------------------------------
    def substrate(self, name: str) -> Substrate:
        for s in self.substrates:
            if s.name == name:
                return s
        raise KeyError(f"no substrate {name!r}; have {[s.name for s in self.substrates]}")

    def cell_type(self, name: str) -> CellType:
        for c in self.cell_types:
            if c.name == name:
                return c
        raise KeyError(f"no cell type {name!r}; have {[c.name for c in self.cell_types]}")

    @property
    def substrate_names(self) -> list[str]:
        return [s.name for s in self.substrates]

    @property
    def cell_type_names(self) -> list[str]:
        return [c.name for c in self.cell_types]

    def copy(self) -> "Model":
        """An independent copy, so a sweep can edit one point without touching
        the original."""
        import copy as _copy

        return _copy.deepcopy(self)

    # -- identity ---------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        # dict keys that are tuples are not JSON-able; stringify them
        for ct in d["cell_types"]:
            ct["cycle"]["transition_rates"] = {f"{i}->{j}": v for (i, j), v in
                                               ct["cycle"]["transition_rates"].items()}
            ct["cycle"]["fixed_duration"] = {f"{i}->{j}": v for (i, j), v in
                                             ct["cycle"]["fixed_duration"].items()}
            for dm in ct["deaths"].values():
                dm["phase_durations"] = {str(k): v for k, v in dm["phase_durations"].items()}
        return d

    @property
    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def summary(self) -> str:
        nx, ny, nz = self.domain.shape
        lines = [
            f"model {self.name!r}  [{self.hash}]",
            f"  domain     {nx} x {ny} x {nz} voxels, dx={self.domain.dx:g} um, "
            f"{'2D' if self.domain.use_2D else '3D'}",
            f"  time       dt_diff={self.time.dt_diffusion:g} dt_mech={self.time.dt_mechanics:g} "
            f"dt_pheno={self.time.dt_phenotype:g}  max_time={self.time.max_time:g} min",
            f"  substrates {', '.join(self.substrate_names) or '(none)'}",
            f"  cell types {', '.join(self.cell_type_names) or '(none)'}",
            f"  rules      {len(self.rules)}",
            f"  initial    {len(self.initial_cells)} cells",
        ]
        return "\n".join(lines)

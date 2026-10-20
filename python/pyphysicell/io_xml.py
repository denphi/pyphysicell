"""Read PhysiCell project files into a Model.

Reads what a project actually is: `PhysiCell_settings.xml`, the CBHG rules CSV
it points at, and the initial-conditions CSV.  No C++ is involved, so this
works on any project without a compiler.
"""

from __future__ import annotations

import copy
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

from . import cycles
from .model import (Cycle, CellType, DeathModel, Domain, Interactions, Mechanics, Model,
                    Motility, ParameterDistribution, Rule, SaveOptions, SecretionEntry,
                    Substrate, Time, Volume)

_FACES = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]


def _f(node, path, default=0.0) -> float:
    el = node.find(path) if node is not None else None
    if el is None or el.text is None or not el.text.strip():
        return float(default)
    return float(el.text)


def _s(node, path, default="") -> str:
    el = node.find(path) if node is not None else None
    return default if el is None or el.text is None else el.text.strip()


def _b(node, path, default=False) -> bool:
    el = node.find(path) if node is not None else None
    if el is None or el.text is None:
        return default
    return el.text.strip().lower() in ("true", "1", "yes")


def _attr_bool(el, name, default=False) -> bool:
    if el is None:
        return default
    return str(el.get(name, default)).strip().lower() in ("true", "1", "yes")


def read_xml(path: str | Path) -> Model:
    path = Path(path)
    root = ET.parse(path).getroot()
    m = Model(name=path.parent.parent.name if path.parent.name == "config" else path.stem)

    d = root.find("domain")
    m.domain = Domain(
        x_min=_f(d, "x_min", -500), x_max=_f(d, "x_max", 500),
        y_min=_f(d, "y_min", -500), y_max=_f(d, "y_max", 500),
        z_min=_f(d, "z_min", -10), z_max=_f(d, "z_max", 10),
        dx=_f(d, "dx", 20), dy=_f(d, "dy", 20), dz=_f(d, "dz", 20),
        use_2D=_b(d, "use_2D", True),
    )

    o = root.find("overall")
    m.time = Time(
        dt_diffusion=_f(o, "dt_diffusion", 0.01),
        dt_mechanics=_f(o, "dt_mechanics", 0.1),
        dt_phenotype=_f(o, "dt_phenotype", 6.0),
        max_time=_f(o, "max_time", 7200.0),
        time_units=_s(o, "time_units", "min"),
        space_units=_s(o, "space_units", "micron"),
    )
    m.omp_num_threads = int(_f(root.find("parallel"), "omp_num_threads", 1) or 1)

    sv = root.find("save")
    if sv is not None:
        m.save = SaveOptions(
            folder=_s(sv, "folder", "output"),
            full_interval=_f(sv, "full_data/interval", 60),
            full_enabled=_b(sv, "full_data/enable", True),
            svg_interval=_f(sv, "SVG/interval", 60),
            svg_enabled=_b(sv, "SVG/enable", True),
        )

    opts = root.find("options")
    if opts is not None:
        m.virtual_wall = _b(opts, "virtual_wall_at_domain_edge", True)
        m.disable_automated_spring_adhesions = _b(opts, "disable_automated_spring_adhesions", False)
        m.legacy_division_orientation = _b(
            opts, "legacy_random_points_on_sphere_in_divide", False)
        if opts.find("random_seed") is not None:
            try:
                m.random_seed = int(_f(opts, "random_seed", 0))
            except ValueError:
                m.random_seed = 0

    # -- substrates --------------------------------------------------------
    me = root.find("microenvironment_setup")
    if me is not None:
        me_opts = me.find("options")
        if me_opts is not None:
            m.calculate_gradients = _b(me_opts, "calculate_gradients", True)
            m.track_internalized_substrates = _b(
                me_opts, "track_internalized_substrates_in_each_agent", False)
        for var in me.findall("variable"):
            pset = var.find("physical_parameter_set")
            sub = Substrate(
                name=var.get("name", "substrate"),
                diffusion_coefficient=_f(pset, "diffusion_coefficient", 0.0),
                decay_rate=_f(pset, "decay_rate", 0.0),
                initial_condition=_f(var, "initial_condition", 0.0),
            )
            dbc = var.find("Dirichlet_boundary_condition")
            sub.dirichlet_enabled = _attr_bool(dbc, "enabled", False)
            sub.dirichlet_value = _f(var, "Dirichlet_boundary_condition", 0.0)
            dopts = var.find("Dirichlet_options")
            for face in _FACES:
                enabled, value = sub.dirichlet_enabled, sub.dirichlet_value
                if dopts is not None:
                    for bv in dopts.findall("boundary_value"):
                        if bv.get("ID") == face:
                            enabled = _attr_bool(bv, "enabled", False)
                            value = float(bv.text) if bv.text and bv.text.strip() else 0.0
                sub.dirichlet_faces[face] = (bool(enabled), float(value))
            m.substrates.append(sub)

    # -- cell types --------------------------------------------------------
    cds = root.find("cell_definitions")
    if cds is not None:
        # PhysiCell: a cell definition starts as a copy of its parent -- the
        # parent_type attribute, else cell_defaults (the "default" definition,
        # itself seeded from the C++ constructors) -- and the XML overrides it.
        nodes = cds.findall("cell_definition")
        by_name: dict[str, CellType] = {}
        default_base: CellType | None = None
        for idx, cd in enumerate(nodes):
            is_default = cd.get("name") == "default" or cd.get("ID") == "0"
            parent = cd.get("parent_type")
            if parent and parent in by_name:
                base = by_name[parent]
            elif not is_default and default_base is not None:
                base = default_base
            else:
                base = None
            ct = _read_cell_definition(cd, idx, m.substrate_names, base)
            by_name[ct.name] = ct
            if is_default and default_base is None:
                default_base = ct
            m.cell_types.append(ct)

    # -- user parameters ---------------------------------------------------
    up = root.find("user_parameters")
    if up is not None:
        for el in up:
            txt = (el.text or "").strip()
            kind = el.get("type", "double")
            try:
                val = int(txt) if kind == "int" else (txt if kind == "string" else float(txt))
            except ValueError:
                val = txt
            m.user_parameters[el.tag] = val
            if el.tag == "random_seed" and isinstance(val, int):
                m.random_seed = val

    # -- rules and initial conditions (sibling files) ----------------------
    base = path.parent
    cr = root.find("cell_rules")
    if cr is not None:
        for rs in cr.findall("rulesets/ruleset"):
            if not _attr_bool(rs, "enabled", True):
                continue
            folder = _s(rs, "folder", ".")
            fname = _s(rs, "filename", "cell_rules.csv")
            for cand in (base / fname, Path(folder) / fname, base.parent / folder / fname):
                if cand.exists():
                    m.rules.extend(read_rules_csv(cand))
                    break

    ic = root.find("initial_conditions/cell_positions")
    if ic is not None and _attr_bool(ic, "enabled", False):
        fname = _s(ic, "filename", "cells.csv")
        folder = _s(ic, "folder", ".")
        for cand in (base / fname, Path(folder) / fname, base.parent / folder / fname):
            if cand.exists():
                m.initial_cells = read_cells_csv(cand, m.cell_type_names)
                break
    return m


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def _read_cell_definition(cd, idx: int, substrates: list[str],
                          base: CellType | None = None) -> CellType:
    name = cd.get("name", f"cell_type_{idx}")
    try:
        cid = int(cd.get("ID", idx))
    except ValueError:
        cid = idx
    ct = copy.deepcopy(base) if base is not None else CellType(name=name, ID=cid)
    ct.name, ct.ID = name, cid
    ph = cd.find("phenotype")
    if ph is None:
        return ct

    cy = ph.find("cycle")
    if cy is not None and (cy.find("phase_transition_rates") is not None
                           or cy.find("phase_durations") is not None
                           or base is None):
        code = int(cy.get("code", ct.cycle.code))
        ct.cycle = Cycle(code=code, name=cy.get("name", ct.cycle.name))
        # PhysiCell starts from the standard model, so a cycle with no rates in
        # the XML still runs at the model's own rates
        std = cycles.for_code(code)
        if std is not None:
            for (i, j, fixed, rate) in std.links:
                ct.cycle.transition_rates[(i, j)] = rate
                ct.cycle.fixed_duration[(i, j)] = fixed
        for r in cy.findall("phase_transition_rates/rate"):
            key = (int(r.get("start_index", 0)), int(r.get("end_index", 0)))
            ct.cycle.transition_rates[key] = float(r.text or 0.0)
            ct.cycle.fixed_duration[key] = str(r.get("fixed_duration", "false")).lower() == "true"
        durations = cy.findall("phase_durations/duration")
        n_phases = len(durations)
        ad = cy.find("standard_asymmetric_division")
        if ad is not None and _attr_bool(ad, "enabled", False):
            for el in ad.findall("asymmetric_division_probability"):
                ct.cycle.asymmetric_division[el.get("name", "")] = float(el.text or 0.0)
        for dur in durations:
            i = int(dur.get("index", 0))
            v = float(dur.text or 0.0)
            # the last phase links back to phase 0; without the wrap the ring
            # grows a phantom terminal phase and the cell never divides
            key = (i, (i + 1) % n_phases) if n_phases else (i, 0)
            ct.cycle.transition_rates[key] = (1.0 / v) if v > 0 else 9e99
            ct.cycle.fixed_duration[key] = str(dur.get("fixed_duration", "true")).lower() == "true"

    if ph.find("death/model") is not None:
        ct.deaths = {}
    for dm in ph.findall("death/model"):
        dname = dm.get("name", "death")
        par = dm.find("parameters")
        model = DeathModel(
            code=int(dm.get("code", 100)), name=dname,
            death_rate=_f(dm, "death_rate", 0.0),
            unlysed_fluid_change_rate=_f(par, "unlysed_fluid_change_rate", 0.05),
            lysed_fluid_change_rate=_f(par, "lysed_fluid_change_rate", 0.0),
            cytoplasmic_biomass_change_rate=_f(par, "cytoplasmic_biomass_change_rate", 0.0166667),
            nuclear_biomass_change_rate=_f(par, "nuclear_biomass_change_rate", 0.00583333),
            calcification_rate=_f(par, "calcification_rate", 0.0),
            relative_rupture_volume=_f(par, "relative_rupture_volume", 2.0),
        )
        # the standard model first, then whatever the XML says: rates or
        # durations, with durations winning when a project gives both
        std_death = cycles.STANDARD_DEATHS.get(dname.lower())
        if std_death is not None:
            model.phase_durations.update(std_death.durations)
        for r in dm.findall("phase_transition_rates/rate"):
            rate = float(r.text or 0.0)
            model.phase_durations[int(r.get("start_index", 0))] = (
                (1.0 / rate) if rate > 0 else 9e99)
        for dur in dm.findall("phase_durations/duration"):
            model.phase_durations[int(dur.get("index", 0))] = float(dur.text or 0.0)
        ct.deaths[dname] = model

    vo = ph.find("volume")
    if vo is not None:
        b = ct.volume
        ct.volume = Volume(
            total=_f(vo, "total", b.total), fluid_fraction=_f(vo, "fluid_fraction", b.fluid_fraction),
            nuclear=_f(vo, "nuclear", b.nuclear),
            fluid_change_rate=_f(vo, "fluid_change_rate", b.fluid_change_rate),
            cytoplasmic_biomass_change_rate=_f(vo, "cytoplasmic_biomass_change_rate",
                                               b.cytoplasmic_biomass_change_rate),
            nuclear_biomass_change_rate=_f(vo, "nuclear_biomass_change_rate",
                                           b.nuclear_biomass_change_rate),
            calcified_fraction=_f(vo, "calcified_fraction", b.calcified_fraction),
            calcification_rate=_f(vo, "calcification_rate", b.calcification_rate),
            relative_rupture_volume=_f(vo, "relative_rupture_volume", b.relative_rupture_volume),
        )

    mech = ph.find("mechanics")
    if mech is not None:
        b = ct.mechanics
        ct.mechanics = Mechanics(
            cell_cell_adhesion_strength=_f(mech, "cell_cell_adhesion_strength",
                                           b.cell_cell_adhesion_strength),
            cell_cell_repulsion_strength=_f(mech, "cell_cell_repulsion_strength",
                                            b.cell_cell_repulsion_strength),
            relative_maximum_adhesion_distance=_f(mech, "relative_maximum_adhesion_distance",
                                                  b.relative_maximum_adhesion_distance),
            attachment_elastic_constant=_f(mech, "attachment_elastic_constant",
                                           b.attachment_elastic_constant),
            attachment_rate=_f(mech, "attachment_rate", b.attachment_rate),
            detachment_rate=_f(mech, "detachment_rate", b.detachment_rate),
            maximum_number_of_attachments=int(_f(mech, "maximum_number_of_attachments",
                                                 b.maximum_number_of_attachments)),
            cell_BM_adhesion_strength=_f(mech, "cell_BM_adhesion_strength",
                                         b.cell_BM_adhesion_strength),
            cell_BM_repulsion_strength=_f(mech, "cell_BM_repulsion_strength",
                                          b.cell_BM_repulsion_strength),
            adhesion_affinities=dict(b.adhesion_affinities),
        )
        for aff in mech.findall("cell_adhesion_affinities/cell_adhesion_affinity"):
            ct.mechanics.adhesion_affinities[aff.get("name", "")] = float(aff.text or 1.0)
        # Mechanics::set_relative/absolute_equilibrium_distance: the adhesion
        # strength is recomputed so that cells rest at the requested spacing
        for tag, absolute in (("set_relative_equilibrium_distance", False),
                              ("set_absolute_equilibrium_distance", True)):
            el = mech.find(f"options/{tag}")
            if el is None or not _attr_bool(el, "enabled", False):
                continue
            value = float(el.text or 0.0)
            if absolute:
                radius = (3.0 * ct.volume.total / (4.0 * 3.141592653589793)) ** (1.0 / 3.0)
                value /= max(radius, 1e-12)
            value = min(value, 2.0)
            t1 = value / 2.0
            t2 = 1.0 - t1
            t1 = 1.0 - t1 / ct.mechanics.relative_maximum_adhesion_distance
            ct.mechanics.cell_cell_adhesion_strength = (
                ct.mechanics.cell_cell_repulsion_strength * (t2 / t1) ** 2)

    mo = ph.find("motility")
    if mo is not None:
        opt = mo.find("options")
        chem = mo.find("options/chemotaxis")
        adv = mo.find("options/advanced_chemotaxis")
        b = ct.motility
        ct.motility = Motility(
            speed=_f(mo, "speed", b.speed),
            persistence_time=_f(mo, "persistence_time", b.persistence_time),
            migration_bias=_f(mo, "migration_bias", b.migration_bias),
            enabled=_b(opt, "enabled", b.enabled),
            use_2D=_b(opt, "use_2D", b.use_2D),
            chemotaxis_enabled=_b(chem, "enabled", b.chemotaxis_enabled),
            chemotaxis_substrate=_s(chem, "substrate", b.chemotaxis_substrate),
            chemotaxis_direction=_f(chem, "direction", b.chemotaxis_direction),
            advanced_chemotaxis=_b(adv, "enabled", b.advanced_chemotaxis),
            normalize_each_gradient=_b(adv, "normalize_each_gradient", b.normalize_each_gradient),
            sensitivities=dict(b.sensitivities),
        )
        if adv is not None:
            for cs in adv.findall("chemotactic_sensitivities/chemotactic_sensitivity"):
                ct.motility.sensitivities[cs.get("substrate", "")] = float(cs.text or 0.0)

    for sub in ph.findall("secretion/substrate"):
        key = sub.get("name", "")
        b = ct.secretion.get(key, SecretionEntry())
        ct.secretion[key] = SecretionEntry(
            secretion_rate=_f(sub, "secretion_rate", b.secretion_rate),
            secretion_target=_f(sub, "secretion_target", b.secretion_target),
            uptake_rate=_f(sub, "uptake_rate", b.uptake_rate),
            net_export_rate=_f(sub, "net_export_rate", b.net_export_rate),
        )
    for s in substrates:
        ct.secretion.setdefault(s, SecretionEntry())

    ci, ctr, cint = ph.find("cell_interactions"), ph.find("cell_transformations"), \
        ph.find("cell_integrity")
    if ci is not None or ctr is not None or cint is not None:
        b = ct.interactions
        # the legacy single rate seeds all three; specific entries then override
        legacy = _f(ci, "dead_phagocytosis_rate", None) if (
            ci is not None and ci.find("dead_phagocytosis_rate") is not None) else None
        it = Interactions(
            apoptotic_phagocytosis_rate=_f(ci, "apoptotic_phagocytosis_rate",
                                           legacy if legacy is not None
                                           else b.apoptotic_phagocytosis_rate),
            necrotic_phagocytosis_rate=_f(ci, "necrotic_phagocytosis_rate",
                                          legacy if legacy is not None
                                          else b.necrotic_phagocytosis_rate),
            other_dead_phagocytosis_rate=_f(ci, "other_dead_phagocytosis_rate",
                                            legacy if legacy is not None
                                            else b.other_dead_phagocytosis_rate),
            attack_damage_rate=_f(ci, "attack_damage_rate", b.attack_damage_rate),
            attack_duration=_f(ci, "attack_duration", b.attack_duration),
            damage_rate=_f(cint, "damage_rate", b.damage_rate),
            damage_repair_rate=_f(cint, "damage_repair_rate", b.damage_repair_rate),
            live_phagocytosis_rates=dict(b.live_phagocytosis_rates),
            attack_rates=dict(b.attack_rates),
            fusion_rates=dict(b.fusion_rates),
            transformation_rates=dict(b.transformation_rates),
            immunogenicities=dict(b.immunogenicities),
        )
        for path, tag, into in (("live_phagocytosis_rates/phagocytosis_rate", "name",
                                 it.live_phagocytosis_rates),
                                ("attack_rates/attack_rate", "name", it.attack_rates),
                                ("fusion_rates/fusion_rate", "name", it.fusion_rates),
                                ("immunogenicities/immunogenicity", "name", it.immunogenicities)):
            for el in (ci.findall(path) if ci is not None else []):
                into[el.get(tag, "")] = float(el.text or 0.0)
        for el in (ctr.findall("transformation_rates/transformation_rate")
                   if ctr is not None else []):
            it.transformation_rates[el.get("name", "")] = float(el.text or 0.0)
        ct.interactions = it

    dists = cd.find("initial_parameter_distributions")
    if dists is not None and _attr_bool(dists, "enabled", False):
        ct.distributions = []
        for dist in dists.findall("distribution"):
            if not _attr_bool(dist, "enabled", True):
                continue
            behavior = _s(dist, "behavior", "")
            params = {child.tag: float(child.text or 0.0) for child in dist
                      if child.tag != "behavior" and (child.text or "").strip()
                      and _is_number(child.text)}
            ct.distributions.append(ParameterDistribution(
                behavior=behavior, kind=dist.get("type", "uniform").lower(),
                parameters=params,
                check_base=str(dist.get("check_base", "false")).lower() == "true"))

    cdta = cd.find("custom_data")
    if cdta is not None:
        for el in cdta:
            try:
                ct.custom_data[el.tag] = float(el.text or 0.0)
            except ValueError:
                continue
            ct.custom_conserved[el.tag] = str(el.get("conserved", "false")).lower() in ("true", "1")
    return ct


def read_rules_csv(path: str | Path) -> list[Rule]:
    """Read a CBHG rules CSV (v3: 8 columns, or v1: 9 with an explicit base)."""
    rules: list[Rule] = []
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if not row or not row[0].strip() or row[0].lstrip().startswith(("//", "#")):
                continue
            row = [c.strip() for c in row]
            if len(row) == 8:
                ct, sig, direction, beh, max_resp, half, hill, dead = row
            elif len(row) == 9:  # v1 carries the base value in column 4
                ct, sig, direction, beh, _base, max_resp, half, hill, dead = row
            else:
                continue
            rules.append(Rule(cell_type=ct, signal=sig, direction=direction.lower(), behavior=beh,
                              max_response=float(max_resp), half_max=float(half),
                              hill_power=float(hill), apply_to_dead=bool(float(dead))))
    return rules


def read_cells_csv(path: str | Path, type_names: list[str]):
    """Read an initial-conditions CSV (v2 with a header, or legacy x,y,z,type)."""
    out = []
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return out
    header = [c.strip().lower() for c in rows[0]]
    has_header = "x" in header and "y" in header
    start = 1 if has_header else 0
    tcol = header.index("type") if has_header and "type" in header else 3
    for row in rows[start:]:
        if len(row) < 4 or not row[0].strip() or row[0].lstrip().startswith(("//", "#")):
            continue
        try:
            x, y, z = float(row[0]), float(row[1]), float(row[2])
        except ValueError:
            continue
        raw = row[tcol].strip() if len(row) > tcol else "0"
        # the type column is an index or a name, and the index may be written as a
        # float ("1.000000000000000000e+00" in mechano's cells.csv)
        try:
            f = float(raw)
        except ValueError:
            name = raw
        else:
            i = int(round(f))
            name = (type_names[i] if 0 <= i < len(type_names)
                    else (type_names[0] if type_names else "cell"))
        out.append((x, y, z, name))
    return out


def write_cells_csv(path: str | Path, cells) -> Path:
    """Write initial conditions as PhysiCell's v2 cells.csv (x,y,z,type)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "z", "type"])
        for (x, y, z, name) in cells:
            w.writerow([repr(float(x)), repr(float(y)), repr(float(z)), name])
    return path


def scenarios(project: str | Path) -> list[Path]:
    """Every settings XML in a project.

    Published projects often ship one per scenario (`PhysiCell_settings_go.xml`,
    `..._grow.xml`, ...), so a project is not always one model.
    """
    p = Path(project)
    if p.is_file():
        return [p]
    found: list[Path] = []
    for sub in ("config", "data", "."):
        d = p / sub
        if d.is_dir():
            found += sorted(d.glob("PhysiCell_settings*.xml"))
    # exact name first, then the rest, without duplicates
    exact = [f for f in found if f.name == "PhysiCell_settings.xml"]
    rest = [f for f in found if f.name != "PhysiCell_settings.xml"]
    out, seen = [], set()
    for f in exact + rest:
        if f.resolve() not in seen:
            seen.add(f.resolve())
            out.append(f)
    return out


def load(project: str | Path, scenario: str | None = None) -> Model:
    """Load a PhysiCell project directory (or a settings XML) as a Model.

    `scenario` picks one when a project ships several, by substring of the file
    name.  With several and no choice made, the error lists them rather than
    guessing.
    """
    p = Path(project)
    if p.is_file():
        return read_xml(p)
    found = scenarios(p)
    if not found:
        raise FileNotFoundError(f"no PhysiCell_settings*.xml under {p}")
    if scenario is not None:
        picked = [f for f in found if scenario.lower() in f.name.lower()]
        if not picked:
            raise KeyError(f"no scenario matching {scenario!r}; have "
                           f"{[f.name for f in found]}")
        return read_xml(picked[0])
    if len(found) > 1 and found[0].name != "PhysiCell_settings.xml":
        raise ValueError(
            f"{p.name} ships {len(found)} scenarios; pass scenario=... to choose: "
            f"{[f.name for f in found]}")
    return read_xml(found[0])

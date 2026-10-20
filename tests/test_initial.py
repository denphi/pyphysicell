"""The ported initial placements, and the defaults they rely on."""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PC = ROOT.parent / "PhysiCell"
sys.path.insert(0, str(ROOT / "python"))

import pyphysicell as pc  # noqa: E402


def counts(project: str) -> Counter:
    model = pc.load(PC / "sample_projects" / project)
    cells, _desc = pc.initial.auto(model)
    return Counter(name for *_xyz, name in cells)


def test_named_counts_are_matched_to_cell_types():
    """number_of_prey -> "prey", number_of_CD8T_cells -> "CD8+ T cell"."""
    assert counts("pred_prey_farmer") == {"prey": 50, "predator": 20, "farmer": 15}
    c = counts("interactions")
    assert c["CD8+ T cell"] == 50 and c["blood vessel"] == 40 and c["differentiated"] == 400


def test_unmatched_counts_fall_back_to_the_spare_cell_type():
    """virus_macrophage counts infected and uninfected cells; both are epithelial."""
    c = counts("virus_macrophage")
    assert c["macrophage"] == 50
    assert c["epithelial cell"] == 1010      # 10 infected + 1000 uninfected


def test_a_radius_parameter_adds_the_tumor():
    """cancer_immune places immune cells *and* a spheroid of cancer cells."""
    c = counts("cancer_immune")
    assert c["immune cell"] == 7500
    assert c["cancer cell"] > 1000, "the tumor_radius spheroid was not placed"


def test_cells_csv_wins_when_the_project_has_one():
    model = pc.load(PC / "sample_projects" / "rules_sample")
    cells, desc = pc.initial.auto(model)
    assert len(cells) == len(model.initial_cells) == 1300
    assert "cells.csv" in desc


def test_hex_disc_is_packed_and_inside_the_radius():
    import numpy as np

    model = pc.load(PC / "sample_projects" / "heterogeneity")
    cells = pc.initial.hex_disc(model, 100.0, model.cell_type_names[0])
    pos = np.array([[x, y] for x, y, _z, _n in cells])
    assert len(cells) > 50
    assert np.all(np.linalg.norm(pos, axis=1) <= 100.0 + 1e-9)
    # nearest-neighbour spacing should be the hex spacing, not a grid
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    assert 0.8 * 16.0 < np.median(d.min(axis=1)) < 1.2 * 20.0


def test_defaults_match_physicell_constructors():
    """Motility::Motility() and Volume::Volume(), not invented values."""
    m = pc.Motility()
    assert m.migration_bias == 0.0 and m.persistence_time == 1.0 and m.speed == 1.0
    assert m.enabled is False and m.use_2D is False
    v = pc.Volume()
    assert v.total == 2494 and v.nuclear == 540 and v.fluid_fraction == 0.75
    assert abs(v.cytoplasmic_biomass_change_rate - 0.27 / 60.0) < 1e-9
    assert abs(v.nuclear_biomass_change_rate - 0.33 / 60.0) < 1e-9
    assert abs(v.fluid_change_rate - 3.0 / 60.0) < 1e-9

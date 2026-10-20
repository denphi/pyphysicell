"""The generated Python must rebuild exactly the model it came from."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PC = ROOT.parent / "PhysiCell"
sys.path.insert(0, str(ROOT / "python"))

import pyphysicell as pc  # noqa: E402

PROJECTS = sorted(d.name for d in (PC / "sample_projects").iterdir()
                  if (d / "config" / "PhysiCell_settings.xml").exists())


@pytest.mark.parametrize("project", PROJECTS)
def test_generated_code_rebuilds_the_model(project):
    loaded = pc.load(PC / "sample_projects" / project)
    src = pc.codegen.to_python(loaded, src=project, initial="literal"
                               if loaded.initial_cells else "none")
    ns = {"pc": pc}
    exec(compile(src, f"<{project}>", "exec"), ns)       # noqa: S102 - that is the test
    rebuilt = ns["model"]
    assert rebuilt.hash == loaded.hash, (
        f"{project}: round trip changed the model\n"
        f"  loaded  {loaded.hash}\n  rebuilt {rebuilt.hash}")


def test_cell_definitions_inherit_from_default():
    """PhysiCell copies the parent (or cell_defaults) before applying the XML."""
    import xml.etree.ElementTree as ET
    import tempfile

    xml = """<PhysiCell_settings>
      <domain><x_min>-50</x_min><x_max>50</x_max><y_min>-50</y_min><y_max>50</y_max>
        <z_min>-10</z_min><z_max>10</z_max><dx>20</dx><dy>20</dy><dz>20</dz>
        <use_2D>true</use_2D></domain>
      <cell_definitions>
        <cell_definition name="default" ID="0"><phenotype>
          <volume><total>1234</total></volume>
          <motility><speed>7.5</speed><options><enabled>true</enabled></options></motility>
        </phenotype></cell_definition>
        <cell_definition name="child" ID="1"><phenotype>
          <motility><speed>2.0</speed></motility>
        </phenotype></cell_definition>
        <cell_definition name="grandchild" ID="2" parent_type="child"><phenotype/>
        </cell_definition>
      </cell_definitions></PhysiCell_settings>"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "PhysiCell_settings.xml"
        p.write_text(xml)
        m = pc.read_xml(p)
    child = m.cell_type("child")
    assert child.volume.total == 1234, "child did not inherit the default's volume"
    assert child.motility.speed == 2.0, "child did not override the speed"
    assert child.motility.enabled is True, "child lost the default's motility options"
    grand = m.cell_type("grandchild")
    assert grand.motility.speed == 2.0, "grandchild did not inherit from parent_type"
    assert grand.volume.total == 1234


def test_cells_csv_round_trips_exactly(tmp_path):
    """The data file beside a notebook must reproduce the project's cells."""
    model = pc.load(PC / "sample_projects" / "rules_sample")
    path = pc.write_cells_csv(tmp_path / "cells.csv", model.initial_cells)
    back = pc.read_cells_csv(path, model.cell_type_names)
    assert back == model.initial_cells


def test_every_notebook_has_the_files_it_needs():
    """A notebook may only read data that is shipped beside it."""
    import json
    import re

    nbdir = ROOT / "notebooks"
    if not any(nbdir.glob("*.ipynb")):
        pytest.skip("notebooks have not been generated")
    missing = []
    for nb_path in sorted(nbdir.glob("*.ipynb")):
        nb = json.loads(nb_path.read_text())
        src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        for ref in re.findall(r"DATA / '([^']+)'", src):
            if not (nbdir / "data" / ref).exists():
                missing.append(f"{nb_path.name} -> data/{ref}")
    assert not missing, "notebooks reference data that is not shipped: " + ", ".join(missing)


def test_signal_behavior_parity_with_physicell():
    """Every signal and behavior PhysiCell exposes must exist here too.

    The reference list comes from PhysiCell itself (tests/ref_dict, built from
    the vendored sources by tools/parity.md); skipped when it is not built.
    """
    import subprocess

    ref = ROOT / "tests" / "ref_dict"
    if not ref.exists():
        pytest.skip("tests/ref_dict is not built")
    config = PC / "sample_projects" / "rules_sample" / "config" / "PhysiCell_settings.xml"
    out = subprocess.run([str(ref), str(config)], capture_output=True, text=True).stdout

    import re
    sig, beh, where = [], [], None
    for line in out.splitlines():
        if line.startswith("#SIGNALS"):
            where = sig
        elif line.startswith("#BEHAVIORS"):
            where = beh
        else:
            m = re.match(r"^\s*(\d+)\s*:\s*(.+?)\s*$", line)
            if m and where is not None:
                where.append(m.group(2))
    assert sig and beh, "the reference produced no dictionary"

    dic = pc.Dictionary(pc.load(config))
    # the basement membrane needs geometry this library does not model
    known_gaps = {"contact with basement membrane"}
    missing_sig = [s for s in sig if s not in set(dic.signals) and s not in known_gaps]
    missing_beh = [b for b in beh if b not in set(dic.behaviors)]
    assert not missing_sig, f"signals PhysiCell has and we do not: {missing_sig}"
    assert not missing_beh, f"behaviors PhysiCell has and we do not: {missing_beh}"


@pytest.mark.parametrize("project", ["rules_sample", "interactions", "biorobots", "worm"])
def test_signal_behavior_values_match_physicell(project):
    """Names are not enough: for the same state, every signal and behavior must
    report the same number PhysiCell reports.

    The reference is PhysiCell itself (tests/ref_values).  This comparison found
    five real defects: a zero attachment elastic constant on new cells, simple
    chemotaxis written into the advanced sensitivity array, the wrong default
    attack duration, the ignored legacy <dead_phagocytosis_rate>, and gradients
    computed under a narrower condition than PhysiCell's.
    """
    import sys as _sys

    if not (ROOT / "tests" / "ref_values").exists():
        pytest.skip("tests/ref_values is not built")
    _sys.path.insert(0, str(ROOT / "tools"))
    import parity_values

    config = PC / "sample_projects" / project / "config" / "PhysiCell_settings.xml"
    ref = parity_values.reference(config)
    sim = parity_values.ours(config)
    dic = sim.dictionary
    import numpy as np

    bad = []
    for i, name in enumerate(sim.model.cell_type_names):
        if name not in ref:
            continue
        idx = np.array([i])
        for key, expected in ref[name]["B"].items():
            got = float(np.asarray(dic.read_behavior(key, sim.cells, idx)).ravel()[0])
            if abs(got - expected) > 1e-9 * max(1.0, abs(expected)):
                bad.append(f"{name} behavior {key}: PhysiCell {expected} vs {got}")
        for key, expected in ref[name]["S"].items():
            got = float(np.asarray(dic.signal_values(key, sim.cells, idx, sim.t)).ravel()[0])
            if abs(got - expected) > 1e-9 * max(1.0, abs(expected)):
                bad.append(f"{name} signal {key}: PhysiCell {expected} vs {got}")
    assert not bad, "\n".join(bad[:20])

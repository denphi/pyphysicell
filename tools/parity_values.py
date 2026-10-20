"""Value-level parity: what PhysiCell reports for a fresh cell, against ours.

Names matching is not enough — a behavior can exist and mean something else.
This builds the same state on both sides (a deterministic field, one fresh cell
of each cell definition at the origin) and diffs every signal and behavior.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import pyphysicell as pc  # noqa: E402

PC = ROOT.parent / "PhysiCell"
REF = ROOT / "tests" / "ref_values"
TOL = 1e-9


def reference(config: Path) -> dict[str, dict[str, dict[str, float]]]:
    out = subprocess.run([str(REF), str(config)], capture_output=True, text=True).stdout
    cells: dict[str, dict[str, dict[str, float]]] = {}
    cur = None
    for line in out.splitlines():
        if line.startswith("#CELL\t"):
            cur = cells.setdefault(line.split("\t")[1], {"B": {}, "S": {}})
        elif cur is not None and line[:2] in ("B\t", "S\t"):
            kind, name, value = line.split("\t")
            cur[kind][name] = float(value)
    return cells


def ours(config: Path):
    """The same state here: the deterministic field, one cell of each type."""
    model = pc.load(config)
    model.initial_cells = [(0.0, 0.0, 0.0, ct.name) for ct in model.cell_types]
    sim = pc.Simulation(model, seed=0)
    me = sim.microenvironment
    n = np.arange(me.nx * me.ny * me.nz).reshape(me.nz, me.ny, me.nx)
    for s in range(me.ns):
        me.density[..., s] = 1.0 + s + n / 1000.0
    me.compute_gradients()
    sim._sample_fields()
    return sim


def main(project: str = "rules_sample"):
    config = PC / "sample_projects" / project / "config" / "PhysiCell_settings.xml"
    if not REF.exists():
        print("build tests/ref_values first")
        return
    ref = reference(config)
    sim = ours(config)
    dic = sim.dictionary
    names = sim.model.cell_type_names

    checked = diffs = skipped = 0
    report: list[str] = []
    for i, name in enumerate(names):
        if name not in ref:
            continue
        idx = np.array([i])
        for kind, reader in (("B", dic.read_behavior), ("S", dic.signal_values)):
            for key, expected in ref[name][kind].items():
                try:
                    got = reader(key, sim.cells, idx, sim.t) if kind == "S" \
                        else reader(key, sim.cells, idx)
                except KeyError:
                    skipped += 1
                    continue
                got = float(np.asarray(got).ravel()[0])
                checked += 1
                if abs(got - expected) > TOL * max(1.0, abs(expected)):
                    diffs += 1
                    report.append(f"  {name:<26} {kind} {key:<34} "
                                  f"PhysiCell {expected:<14.6g} here {got:.6g}")
    print(f"project {project}: {checked} values compared, {diffs} differ, {skipped} unreadable")
    for line in report[:40]:
        print(line)
    if len(report) > 40:
        print(f"  ... and {len(report) - 40} more")


if __name__ == "__main__":
    main(*sys.argv[1:])

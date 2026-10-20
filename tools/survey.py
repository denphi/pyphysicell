"""Survey every PhysiCell sample project: what loads, what runs, what is missing."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import pyphysicell as pc  # noqa: E402
from pyphysicell.rules import Ruleset  # noqa: E402
from pyphysicell.signals import Dictionary  # noqa: E402

PC = ROOT.parent / "PhysiCell"


def projects() -> list[Path]:
    out = []
    for pat in ("sample_projects/*", "sample_projects_intracellular/*/*"):
        for d in sorted(PC.glob(pat)):
            if (d / "config" / "PhysiCell_settings.xml").exists():
                out.append(d)
    return out


def survey_one(d: Path) -> dict:
    row = {"project": d.name, "path": d, "group": d.parent.name}
    try:
        m = pc.load(d)
    except Exception as e:                      # noqa: BLE001
        row["error"] = f"{type(e).__name__}: {e}"
        return row
    dic = Dictionary(m)
    rs = Ruleset(m, dic, strict=False)
    nx, ny, nz = m.domain.shape
    row.update(
        model=m,
        dims="2D" if m.domain.use_2D else "3D",
        voxels=nx * ny * nz,
        substrates=len(m.substrates),
        cell_types=len(m.cell_types),
        initial_cells=len(m.initial_cells),
        rules=len(m.rules),
        rules_run=len(m.rules) - len(rs.skipped),
        rules_skipped=len(rs.skipped),
        skipped_reasons=sorted({why for _r, why in rs.skipped}),
        cycles=sorted({c.cycle.name for c in m.cell_types}),
        motile=sum(1 for c in m.cell_types if c.motility.enabled),
        chemotaxis=sum(1 for c in m.cell_types
                       if c.motility.chemotaxis_enabled or c.motility.advanced_chemotaxis),
        max_time=m.time.max_time,
    )
    if row["rules"] and row["rules_skipped"] == 0:
        tier = "A"
    elif row["rules_skipped"]:
        tier = "B"
    elif row["initial_cells"] == 0:
        tier = "B"          # cells are placed by C++ setup_tissue
    else:
        tier = "A"
    row["tier"] = tier
    return row


def main():
    rows = [survey_one(d) for d in projects()]
    hdr = (f"{'project':<24}{'grp':<5}{'dim':<4}{'voxels':>8}{'sub':>4}{'types':>6}"
           f"{'cells':>7}{'rules':>6}{'run':>5}{'skip':>5}  tier  cycles")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['project']:<24}{'':<5}FAILED: {r['error']}")
            continue
        grp = "int" if r["group"] != "sample_projects" else "smp"
        print(f"{r['project']:<24}{grp:<5}{r['dims']:<4}{r['voxels']:>8}{r['substrates']:>4}"
              f"{r['cell_types']:>6}{r['initial_cells']:>7}{r['rules']:>6}{r['rules_run']:>5}"
              f"{r['rules_skipped']:>5}  {r['tier']:<5} {', '.join(r['cycles'])[:34]}")
    ok = [r for r in rows if "error" not in r]
    print(f"\n{len(ok)}/{len(rows)} projects load. "
          f"tier A: {sum(1 for r in ok if r['tier'] == 'A')}, "
          f"tier B: {sum(1 for r in ok if r['tier'] == 'B')}")
    missing: dict[str, int] = {}
    for r in ok:
        for why in r.get("skipped_reasons", []):
            missing[why] = missing.get(why, 0) + 1
    if missing:
        print("\nmost common gaps:")
        for why, n in sorted(missing.items(), key=lambda kv: -kv[1])[:12]:
            print(f"  {n:>2} projects  {why}")
    return rows


if __name__ == "__main__":
    main()

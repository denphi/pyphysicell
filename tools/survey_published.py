"""Coverage report for published PhysiCell models, not just the sample projects."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import pyphysicell as pc  # noqa: E402
from gaps import custom_slots  # noqa: E402
from pyphysicell.rules import Ruleset  # noqa: E402
from pyphysicell.signals import Dictionary  # noqa: E402

PUBLISHED = ROOT.parent / "published"


def projects():
    out = []
    for cfg in sorted(PUBLISHED.rglob("PhysiCell_settings*.xml")):
        path = str(cfg)
        if any(skip in path for skip in ("/sample_projects", "/unit_tests", "/tests/",
                                         "/output/", "/backup")):
            continue
        out.append(cfg)
    return out


def main():
    rows = []
    for d in projects():
        label = str(d.relative_to(PUBLISHED)).replace("/config/PhysiCell_settings", "")
        label = label.replace(".xml", "").replace("grammar_samples/user_projects/", "")
        try:
            m = pc.load(d)
        except Exception as e:                       # noqa: BLE001
            rows.append((label, None, f"{type(e).__name__}: {e}"))
            continue
        dic = Dictionary(m)
        rs = Ruleset(m, dic, strict=False)
        root = d.parent.parent
        slots = custom_slots(root) if (root / "custom_modules").exists() else {}
        rows.append((label, (m, rs, slots), None))

    hdr = (f"{'model':<34}{'dim':<4}{'types':>6}{'subs':>5}{'rules':>6}{'run':>5}"
           f"{'skip':>5}{'cells':>7}  custom C++")
    print(hdr)
    print("-" * len(hdr))
    gaps, ok = {}, 0
    for label, got, err in rows:
        if err:
            print(f"{label:<34}FAILED: {err[:60]}")
            continue
        m, rs, slots = got
        cells, _ = pc.initial.auto(m)
        runnable = len(m.rules) - len(rs.skipped)
        clean = not rs.skipped and not slots
        ok += clean
        print(f"{label:<34}{'2D' if m.domain.use_2D else '3D':<4}{len(m.cell_types):>6}"
              f"{len(m.substrates):>5}{len(m.rules):>6}{runnable:>5}{len(rs.skipped):>5}"
              f"{len(cells):>7}  {', '.join(slots) or '-'}")
        for _r, why in rs.skipped:
            gaps[why] = gaps.get(why, 0) + 1
    print(f"\n{ok}/{len([r for r in rows if not r[2]])} published models need neither a "
          f"skipped rule nor custom C++.")
    if gaps:
        print("\nmissing for the rest:")
        for why, n in sorted(gaps.items(), key=lambda kv: -kv[1]):
            print(f"  {n:>3}  {why}")


if __name__ == "__main__":
    main()

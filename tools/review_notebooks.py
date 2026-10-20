"""Review the executed notebooks: do they run, and do they show anything?

A notebook that executes cleanly can still be worthless -- zero cells, a flat
population, a run so short nothing happens, a figure with no data.  This checks
for those as well as for errors.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"


def outputs_text(cell) -> str:
    out = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream":
            out.append("".join(o.get("text", [])))
        elif o.get("output_type") == "error":
            out.append("ERROR: " + o.get("ename", "") + ": " + o.get("evalue", ""))
        elif o.get("output_type") in ("execute_result", "display_data"):
            data = o.get("data", {})
            if "text/plain" in data:
                out.append("".join(data["text/plain"]))
            if "image/png" in data:
                out.append(f"<image {len(data['image/png'])} b64 chars>")
    return "\n".join(out)


def review(path: Path) -> dict:
    nb = json.loads(path.read_text())
    cells = nb["cells"]
    code = [c for c in cells if c["cell_type"] == "code"]
    text = "\n".join(outputs_text(c) for c in code)
    src = "\n".join("".join(c["source"]) for c in code)

    issues = []
    errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    if errors:
        issues.append(f"{len(errors)} cell error(s): " + errors[0].get("ename", ""))
    unrun = [c for c in code if c.get("execution_count") is None and "".join(c["source"]).strip()]
    if unrun:
        issues.append(f"{len(unrun)} cell(s) never executed")

    images = sum(1 for c in code for o in c.get("outputs", [])
                 if "image/png" in o.get("data", {}))
    if images == 0:
        issues.append("no figures")

    m = re.search(r"cells=(\d+)", text)
    n_cells = int(m.group(1)) if m else None
    if n_cells == 0:
        issues.append("ran with zero cells")

    m = re.search(r"UNTIL = ([\d.]+)", src)
    until = float(m.group(1)) if m else None
    dt_pheno = 6.0
    mp = re.search(r"dt_phenotype=([\d.]+)", src)
    if mp:
        dt_pheno = float(mp.group(1))
    if until is not None and until < 3 * dt_pheno:
        issues.append(f"demo runs only {until:g} min (< 3 phenotype steps)")

    if "hook fired" in text and re.search(r"hook fired 0 times", text):
        issues.append("the hook never fired")
    # A sweep is vacuous only if *nothing* it reports moves.  Checking the live
    # count alone condemned 18 notebooks whose sweeps plainly worked: varying
    # the oxygen half-max in rules_sample leaves the count at 1300 while deaths
    # go 0 -> 5 -> 17, and worm's motility sweep halves the spread.
    sweep_rows = re.findall(r"rep \d+\s+->\s+(\d+) cells", text)
    table = re.findall(r"^\s*\d+\s+([\d.]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)",
                       text, re.M)
    if table:
        by_value = {}
        for value, _rep, cells, births, deaths, spread, pressure in table:
            by_value.setdefault(value, []).append((cells, births, deaths,
                                                   round(float(spread), 1),
                                                   round(float(pressure), 3)))
        means = [tuple(sum(float(r[i]) for r in rows) / len(rows) for i in range(5))
                 for rows in by_value.values()]
        if len(by_value) > 1 and len(set(means)) == 1:
            issues.append("sweep produced identical results at every point")
    elif sweep_rows and len(set(sweep_rows)) == 1:
        issues.append(f"sweep reported the same cell count ({sweep_rows[0]}) at every point "
                      f"and printed no table to check")
    if re.search(r"0 deaths reached at t", text):
        issues.append("the stopping condition never triggered")

    return {"notebook": path.name, "cells_code": len(code), "figures": images,
            "n_cells": n_cells, "until": until, "issues": issues,
            "bytes": path.stat().st_size}


def main():
    rows = [review(p) for p in sorted(NB.glob("*.ipynb"))]
    clean = [r for r in rows if not r["issues"]]
    print(f"{len(rows)} notebooks, {len(clean)} with no issues\n")
    print(f"{'notebook':<34}{'cells':>7}{'figs':>5}{'until':>8}  issues")
    print("-" * 96)
    for r in rows:
        print(f"{r['notebook']:<34}{str(r['n_cells'] or '-'):>7}{r['figures']:>5}"
              f"{str(r['until'] or '-'):>8}  {'; '.join(r['issues'])}")
    tally: dict[str, int] = {}
    for r in rows:
        for i in r["issues"]:
            key = re.sub(r"\d+", "N", i)
            tally[key] = tally.get(key, 0) + 1
    if tally:
        print("\nissue tally:")
        for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
            print(f"  {v:>3}  {k}")


if __name__ == "__main__":
    main()

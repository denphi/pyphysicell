"""Write notebooks/README.md: one row per project, straight from the survey."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from survey import projects, survey_one  # noqa: E402

import pyphysicell as pc  # noqa: E402


def run_results() -> dict[str, str]:
    """PASS/FAIL per notebook, from tools/run_notebooks.sh output if it was saved."""
    log = ROOT / "notebooks" / ".run_results"
    if not log.exists():
        return {}
    out = {}
    for line in log.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] in ("PASS", "FAIL"):
            out[parts[1]] = f"{parts[0]} ({parts[2]})"
    return out


def main():
    rows = [survey_one(d) for d in projects()]
    results = run_results()
    out = ["# Sample projects as notebooks", "",
           "Every PhysiCell sample project, each as a **self-contained notebook**: the model is",
           "Python generated from the project's XML and rules CSV, so nothing here reads a",
           "PhysiCell file at run time. `pyphysicell.codegen` writes them and a test asserts the",
           "generated code rebuilds a model with the same content hash as the XML it came from.",
           "",
           "`00_tour.ipynb` is the exception and the place to start: it *loads* a project to show",
           "the reader, the checks, the rules, stepping, intervening mid-run and re-colouring.",
           "",
           "Initial conditions ship as CSV beside the notebooks, in `data/`, so a notebook needs",
           "only itself, its data file and the `pyphysicell` package -- never the PhysiCell tree.",
           "Projects that place their cells in C++ have that placement ported instead, either in",
           "`pyphysicell.ports.<project>` or by `pyphysicell.initial.auto`.",
           "",
           "| notebook | dim | voxels | substrates | cell types | cells | rules (run/skipped) | executed |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        nb = ROOT / "notebooks" / f"{r['project']}.ipynb"
        if "error" in r:
            out.append(f"| {r['project']} | — | — | — | — | — | — | **no**: {r['error'][:60]} |")
            continue
        cells, _desc = pc.initial.auto(r["model"])
        link = f"[{r['project']}]({r['project']}.ipynb)" if nb.exists() else r["project"]
        rules = (f"{r['rules_run']}/{r['rules_skipped']}" if r["rules"] else "—")
        status = results.get(r["project"], "not run")
        mark = "yes" if status.startswith("PASS") else ("**fails**" if status.startswith("FAIL")
                                                        else status)
        if status.startswith("PASS"):
            mark = f"yes, {status.split('(')[1].rstrip(')')}"
        out.append(f"| {link} | {r['dims']} | {r['voxels']:,} | {r['substrates']} | "
                   f"{r['cell_types']} | {len(cells):,} | {rules} | {mark} |")
    ok = [r for r in rows if "error" not in r]
    gaps: dict[str, int] = {}
    for r in ok:
        for why in r.get("skipped_reasons", []):
            gaps[why] = gaps.get(why, 0) + 1
    out += ["", f"{len(ok)} of {len(rows)} projects load.", "",
            "## Where the cells come from", "",
            "Most projects place their cells in C++ (`setup_tissue`), not in `cells.csv`, so the",
            "notebook ports that placement with `pyphysicell.initial.auto`, driven by the project's",
            "own `number_of_*` and `*_radius` user parameters. Each notebook prints exactly what",
            "it did and says it is a port, not the original C++.", ""]
    if gaps:
        out += ["## Rules that do not run yet", "",
                "| missing | projects |", "|---|---|"]
        for why, n in sorted(gaps.items(), key=lambda kv: (-kv[1], kv[0])):
            out.append(f"| {why} | {n} |")
        out.append("")
    (ROOT / "notebooks" / "README.md").write_text("\n".join(out))
    print("wrote notebooks/README.md")


if __name__ == "__main__":
    main()

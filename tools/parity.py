"""Feature parity: PhysiCell's own dictionaries against this library's.

Builds PhysiCell's signal/behavior dictionary for a project (tests/ref_dict,
compiled from the vendored sources) and diffs it against pyphysicell.Dictionary
for the same model.  Anything PhysiCell exposes and we do not is a gap.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import pyphysicell as pc  # noqa: E402

PC = ROOT.parent / "PhysiCell"
REF = ROOT / "tests" / "ref_dict"


def reference(config: Path) -> tuple[list[str], list[str]]:
    out = subprocess.run([str(REF), str(config)], capture_output=True, text=True).stdout
    sig, beh, where = [], [], None
    for line in out.splitlines():
        if line.startswith("#SIGNALS"):
            where = sig
            continue
        if line.startswith("#BEHAVIORS"):
            where = beh
            continue
        m = re.match(r"^\s*(\d+)\s*:\s*(.+?)\s*$", line)
        if m and where is not None:
            where.append(m.group(2))
    return sig, beh


def main(project: str = "rules_sample"):
    config = PC / "sample_projects" / project / "config" / "PhysiCell_settings.xml"
    assert REF.exists(), "build tests/ref_dict first (see tools/parity.md)"
    ref_sig, ref_beh = reference(config)

    model = pc.load(config)
    dic = pc.Dictionary(model)
    ours_sig, ours_beh = set(dic.signals), set(dic.behaviors)
    # our names for the same thing
    alias = {"custom:": "custom:"}

    def norm(n: str) -> str:
        n = n.strip()
        for a, b in alias.items():
            if n.startswith(a):
                return b + n[len(a):].strip()
        return n

    missing_sig = [s for s in ref_sig if norm(s) not in ours_sig]
    missing_beh = [b for b in ref_beh if norm(b) not in ours_beh]
    extra_sig = [s for s in sorted(ours_sig) if s not in {norm(x) for x in ref_sig}]
    extra_beh = [b for b in sorted(ours_beh) if b not in {norm(x) for x in ref_beh}]

    print(f"project: {project}")
    print(f"  PhysiCell: {len(ref_sig)} signals, {len(ref_beh)} behaviors")
    print(f"  here     : {len(ours_sig)} signals, {len(ours_beh)} behaviors")
    print(f"\nsignals PhysiCell has and we do not ({len(missing_sig)}):")
    for s in missing_sig:
        print(f"    {s}")
    print(f"\nbehaviors PhysiCell has and we do not ({len(missing_beh)}):")
    for b in missing_beh:
        print(f"    {b}")
    if extra_sig or extra_beh:
        print(f"\nnames only we have ({len(extra_sig)} signals, {len(extra_beh)} behaviors):")
        for n in (extra_sig + extra_beh)[:12]:
            print(f"    {n}")
    return missing_sig, missing_beh


if __name__ == "__main__":
    main(*sys.argv[1:])

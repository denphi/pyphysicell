"""Which elements of PhysiCell's config the reader never looks at."""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PC = ROOT.parent / "PhysiCell"
READER = (ROOT / "python" / "pyphysicell" / "io_xml.py").read_text()

SKIP_PREFIX = ("PhysiCell_settings/cell_definitions/cell_definition/phenotype/intracellular",
               "PhysiCell_settings/user_parameters")


def paths(node, prefix=""):
    here = f"{prefix}/{node.tag}" if prefix else node.tag
    yield here
    for child in node:
        yield from paths(child, here)


def main():
    seen = Counter()
    for cfg in sorted(PC.glob("sample_projects/*/config/PhysiCell_settings.xml")):
        for p in paths(ET.parse(cfg).getroot()):
            seen[p] += 1
    read, ignored = [], []
    for path, n in seen.items():
        tag = path.rsplit("/", 1)[-1]
        if path.startswith(SKIP_PREFIX):
            continue
        # the reader names the element, or its parent handles it by attribute
        if tag in READER:          # the reader mentions the element somewhere
            read.append(path)
        else:
            ignored.append((n, path))
    print(f"{len(read)} element paths read, {len(ignored)} not read\n")
    print("not read (projects using it, path):")
    for n, path in sorted(ignored, key=lambda x: -x[0]):
        print(f"  {n:>3}  {path.replace('PhysiCell_settings/', '')}")


if __name__ == "__main__":
    main()

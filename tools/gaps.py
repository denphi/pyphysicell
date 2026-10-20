"""What stands between each sample project and running its actual model."""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import pyphysicell as pc  # noqa: E402
from pyphysicell.rules import Ruleset  # noqa: E402
from pyphysicell.signals import Dictionary  # noqa: E402
from survey import projects  # noqa: E402

SLOTS = ["update_phenotype", "custom_cell_rule", "update_velocity", "contact_function",
         "cell_division_function", "update_migration_bias", "volume_update_function",
         "add_cell_basement_membrane_interactions", "calculate_distance_to_membrane",
         "set_orientation", "instantiate_cell", "pre_update_intracellular",
         "post_update_intracellular"]
STANDARD = {"standard_volume_update_function", "standard_update_cell_velocity", "NULL",
            "standard_elastic_contact_function", "up_orientation",
            "standard_cell_cell_interactions", "standard_add_basement_membrane_interactions"}


def _body_of(text: str, fn: str) -> str | None:
    """The body of a function defined in this translation unit, if it is here."""
    m = re.search(rf"\n[A-Za-z_][\w:<>,&\* ]*\b{re.escape(fn)}\s*\([^)]*\)\s*\{{", text)
    if not m:
        return None
    i, depth = m.end() - 1, 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def _is_stub(body: str | None) -> bool:
    if body is None:
        return False
    stripped = re.sub(r"//[^\n]*|/\*.*?\*/", "", body, flags=re.S)
    stripped = re.sub(r"\s+|return\s*;|;", "", stripped)
    return stripped == ""


def custom_slots(project: Path) -> Counter:
    """Slots assigned to project code that actually does something.

    Most projects assign the stubs the template ships with (`{ return; }`), which
    need no support at all; counting those would overstate the gap.
    """
    out = Counter()
    texts = [f.read_text(errors="ignore") for f in (project / "custom_modules").glob("*.cpp")]
    joined = "\n".join(texts)
    for text in texts:
        for slot in SLOTS:
            for m in re.finditer(rf"functions\.{slot}\s*=\s*([A-Za-z_][\w:]*)", text):
                fn = m.group(1)
                if fn in STANDARD:
                    continue
                if _is_stub(_body_of(joined, fn)):
                    continue
                out[slot] += 1
    return out


def main():
    rows, slot_tally, rule_tally = [], Counter(), Counter()
    for d in projects():
        try:
            m = pc.load(d)
        except Exception:                               # noqa: BLE001
            continue
        dic = Dictionary(m)
        rs = Ruleset(m, dic, strict=False)
        slots = custom_slots(d)
        slot_tally.update(slots)
        for _r, why in rs.skipped:
            rule_tally[why.split("'")[1] if "'" in why else why] += 1
        rows.append((d.name, len(rs.skipped), slots))

    print(f"{'project':<24}{'rules skipped':>14}  custom C++ slots in use")
    print("-" * 78)
    for name, nskipped, slots in rows:
        tag = ", ".join(f"{k}" for k in slots) or "-"
        print(f"{name:<24}{nskipped:>14}  {tag}")

    print("\ncustom C++ slots, across all projects:")
    for slot, n in slot_tally.most_common():
        print(f"  {n:>2}  {slot}")
    print("\nsignals/behaviors the rules need and we lack:")
    for what, n in rule_tally.most_common():
        print(f"  {n:>2}  {what}")
    clean = sum(1 for _n, s, sl in rows if s == 0 and not sl)
    print(f"\n{clean}/{len(rows)} projects need neither a skipped rule nor real custom C++.")


if __name__ == "__main__":
    main()

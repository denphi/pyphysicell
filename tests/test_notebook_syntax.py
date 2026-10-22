"""The generated notebooks must parse on the oldest Python we support.

A notebook that only runs on 3.12 is no use on a gateway whose default kernel
is 3.8, and the failure is a SyntaxError at cell execution -- after the user
has already installed everything and waited for a run.

`ast.parse(..., feature_version=(3, 8))` does not catch this: PEP 701 relaxed
the *tokenizer*, so a nested same-quote f-string is accepted by a 3.12 parser
whatever feature_version says. Hence the scan.
"""
import json
import re
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))

# an f-string opened with one quote char that contains that same char before
# it closes -- legal from 3.12, a SyntaxError on 3.8 through 3.11
NESTED_QUOTE = re.compile(r"""(?:^|[^\w'"])[rbu]*f(['"])(?:[^\n\\])*?\{[^}\n]*\1""")


def _code(nb: Path) -> list[tuple[int, str]]:
    cells = json.loads(nb.read_text())["cells"]
    return [(i, "".join(c["source"])) for i, c in enumerate(cells) if c["cell_type"] == "code"]


@pytest.mark.skipif(not NOTEBOOKS, reason="no notebooks generated")
@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.stem)
def test_no_pep701_fstrings(nb):
    offenders = [(i, line.strip())
                 for i, src in _code(nb)
                 for line in src.splitlines()
                 if NESTED_QUOTE.search(line)]
    assert not offenders, (
        f"{nb.name} uses f-strings that need Python 3.12:\n  "
        + "\n  ".join(f"cell {i}: {line}" for i, line in offenders[:3]))


def test_the_scanner_catches_the_real_case():
    bad = "print(f'mean oxygen {before:.3g} -> {sim.microenvironment['oxygen'].mean():.3g}')"
    good = "print(f'mean oxygen {before:.3g} -> {after:.3g}')"
    also_good = 'print(f"mean {sim.microenvironment[\'oxygen\'].mean():.3g}")'
    assert NESTED_QUOTE.search(bad)
    assert not NESTED_QUOTE.search(good)
    assert not NESTED_QUOTE.search(also_good)

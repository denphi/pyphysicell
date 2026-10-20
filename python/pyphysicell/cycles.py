"""PhysiCell's standard cycle and death models, as data.

Dumped from PhysiCell itself (`tests/physicell_reference_models.cpp`) rather
than assumed.  Two things depend on this and cannot be guessed from the XML:

* which phase divides.  It is *not* always the one that returns to phase 0 —
  Ki67 (advanced) divides on exit from phase 1, leaving a postmitotic phase
  behind it.
* the default transition rates.  A project may give a `<cycle>` with no rates
  at all, and PhysiCell then uses the model's own.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StandardCycle:
    code: int
    name: str
    phases: list[str]
    # (start, end, fixed_duration, default rate 1/min)
    links: list[tuple[int, int, bool, float]]
    division_phase: int           # division happens on exit from this phase
    entry_phase: int              # the phase whose entry doubles the volume targets
    removal_phase: int | None = None


STANDARD_CYCLES: dict[int, StandardCycle] = {
    0: StandardCycle(0, "Ki67 (advanced)", ["Ki67-", "Ki67+ (premitotic)", "Ki67+ (postmitotic)"],
                     [(0, 1, False, 0.00460405), (1, 2, True, 0.00128205),
                      (2, 0, True, 0.00666667)],
                     division_phase=1, entry_phase=1),
    1: StandardCycle(1, "Ki67 (basic)", ["Ki67-", "Ki67+"],
                     [(0, 1, False, 0.00363108), (1, 0, True, 0.00107527)],
                     division_phase=1, entry_phase=1),
    2: StandardCycle(2, "Flow cytometry model (basic)", ["G0/G1", "S", "G2/M"],
                     [(0, 1, False, 0.00324), (1, 2, False, 0.00208), (2, 0, False, 0.00333)],
                     division_phase=2, entry_phase=1),
    5: StandardCycle(5, "Live", ["Live"], [(0, 0, False, 0.00072)],
                     division_phase=0, entry_phase=0),
    6: StandardCycle(6, "Flow cytometry model (separated)", ["G0/G1", "S", "G2", "M"],
                     [(0, 1, False, 0.00335), (1, 2, False, 0.00208), (2, 3, False, 0.00417),
                      (3, 0, False, 0.0167)],
                     division_phase=3, entry_phase=1),
    7: StandardCycle(7, "Cycling-Quiescent model", ["Quiescent", "Cycling"],
                     [(0, 1, False, 0.00363108), (1, 0, True, 0.00107527)],
                     division_phase=1, entry_phase=1),
}


@dataclass
class StandardDeath:
    code: int
    name: str
    phases: list[str]
    durations: dict[int, float] = field(default_factory=dict)
    removal_phase: int = 0


STANDARD_DEATHS: dict[str, StandardDeath] = {
    "apoptosis": StandardDeath(100, "Apoptosis", ["Apoptotic", "Debris"],
                               {0: 516.0}, removal_phase=0),
    # necrosis: swelling is held by an arrest function until the rupture volume
    "necrosis": StandardDeath(101, "Necrosis", ["Necrotic (swelling)", "Necrotic (lysed)",
                                                "Debris"],
                              {0: 0.0, 1: 86400.0}, removal_phase=1),
}


def for_code(code: int) -> StandardCycle | None:
    return STANDARD_CYCLES.get(int(code))


def division_phase(code: int, n_phases: int) -> int:
    """Which phase divides, falling back to 'the one that closes the ring'."""
    std = for_code(code)
    if std is not None:
        return std.division_phase
    return max(n_phases - 1, 0)


def entry_phase(code: int, n_phases: int) -> int:
    std = for_code(code)
    if std is not None:
        return std.entry_phase
    return 1 if n_phases > 1 else 0

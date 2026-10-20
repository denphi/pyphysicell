"""Cheap model checks: dimensional analysis and a coverage report.

Neither runs a simulation.  `check` answers "is this model resolved and
stable?", `inspect` answers "how much of this project can the MVP run?".
"""

from __future__ import annotations

import numpy as np

from .model import Model
from .rules import Ruleset
from .signals import Dictionary


def check(model: Model) -> str:
    """Dimensional analysis of a model: resolution, time steps, mesh."""
    out = [f"check: {model.name!r} [{model.hash}]"]
    d, t = model.domain, model.time
    nx, ny, nz = d.shape
    h = min(d.dx, d.dy, d.dz if not d.use_2D else d.dx)

    for s in model.substrates:
        L = s.diffusion_length
        if np.isinf(L):
            spread = np.sqrt(2.0 * s.diffusion_coefficient * 1440.0)
            verdict = "no decay: no steady length scale"
            detail = f"spreads ~{spread:.0f} um per day"
            flag = "note" if spread > h else "WARN"
            out.append(f"  [{flag}] {s.name:<26} D={s.diffusion_coefficient:g}, lambda=0; "
                       f"{verdict}, {detail}")
        else:
            ratio = L / h
            flag = "ok" if ratio >= 10 else ("marginal" if ratio >= 3 else "WARN")
            out.append(f"  [{flag:<8}] {s.name:<26} L=sqrt(D/lambda)={L:8.1f} um "
                       f"= {ratio:5.1f} dx")
        # explicit stability is not required (the solver is implicit), but a
        # step much longer than the decay time loses accuracy
        if s.decay_rate > 0 and t.dt_diffusion > 0.5 / s.decay_rate:
            out.append(f"  [WARN    ] {s.name}: dt_diffusion {t.dt_diffusion:g} min is long "
                       f"against the decay time {1.0 / s.decay_rate:g} min")

    for c in model.cell_types:
        if c.motility.enabled and c.motility.speed > 0:
            step = c.motility.speed * t.dt_mechanics
            if step > 0.25 * h:
                out.append(f"  [WARN    ] {c.name}: moves {step:.2f} um per mechanics step, "
                           f"over a quarter of dx={h:g}")
        mean = c.cycle.mean_duration()
        if np.isfinite(mean) and mean > 0:
            out.append(f"  [info    ] {c.name:<26} cycle mean duration {mean / 60.0:.1f} h "
                       f"(from the phase graph, no simulation)")
        else:
            out.append(f"  [info    ] {c.name:<26} does not cycle on its own "
                       f"(rate 0; a rule may drive it)")

    voxels = nx * ny * nz
    field_mb = voxels * len(model.substrates) * 8 / 1e6
    out.append(f"  [info    ] mesh {nx}x{ny}x{nz} = {voxels} voxels, "
               f"{field_mb:.1f} MB of field per saved snapshot")
    steps = int(t.max_time / t.dt_diffusion)
    out.append(f"  [info    ] {steps} diffusion steps to reach max_time={t.max_time:g} min")
    return "\n".join(out)


def inspect(model: Model) -> str:
    """What this MVP can and cannot run in a given project."""
    dic = Dictionary(model)
    rs = Ruleset(model, dic, strict=False)
    lines = [f"inspect: {model.name!r} [{model.hash}]", model.summary(), ""]

    cycles = {c.cycle.name for c in model.cell_types}
    lines.append(f"  cycle models      {', '.join(sorted(cycles))}")
    lines.append(f"  signals available {len(dic.signals)}")
    lines.append(f"  behaviors available {len(dic.behaviors)}")
    applied = sum(len(v) for v in rs.compiled.values())
    lines.append(f"  rules             {len(model.rules)} total, "
                 f"{len(model.rules) - len(rs.skipped)} runnable, {len(rs.skipped)} skipped")
    if rs.skipped:
        lines.append("  not implemented in this MVP:")
        seen = set()
        for r, why in rs.skipped:
            key = why.split(" not implemented")[0]
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"    - {why}")
    lines.append(f"  behaviors driven by rules: {applied}")
    tier = "A (runs as-is)" if not rs.skipped else "B (runs with the listed rules skipped)"
    lines.append(f"  tier: {tier}")
    return "\n".join(lines)

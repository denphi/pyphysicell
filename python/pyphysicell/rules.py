"""The CBHG rules engine, vectorized over the population.

PhysiCell evaluates one cell at a time through `apply_ruleset`.  The maths is
identical here, but it runs once per cell type per phenotype step over NumPy
arrays, which is what makes a Python-side rules engine affordable.

For each (cell type, behavior):
    U   = base + (max - base) * Hill(up signals)
    out = U    + (min - U)    * Hill(down signals)
with PhysiCell's v3 CSV convention: an "increases" row sets max = its response
and min = base; a "decreases" row sets min = its response and max = base.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import Model, Rule
from .signals import Dictionary


def multivariate_hill(signals: list[np.ndarray], half_maxes: list[float],
                      powers: list[float]) -> np.ndarray:
    """PhysiCell's multivariate_Hill_response_function: sum of (s/s_half)^h."""
    if not signals:
        return np.zeros(0)
    total = np.zeros_like(signals[0], dtype=float)
    for s, hm, h in zip(signals, half_maxes, powers):
        total += np.power(np.maximum(s, 0.0) / hm, h)
    return total / (1.0 + total)


@dataclass
class CompiledBehavior:
    cell_type: str
    behavior: str
    base: float
    min_value: float
    max_value: float
    up: list[Rule] = field(default_factory=list)
    down: list[Rule] = field(default_factory=list)

    def applies(self, cells, idx: np.ndarray) -> np.ndarray:
        """PhysiCell keeps the previous value when no signal applies to a dead cell."""
        if any(r.apply_to_dead for r in (*self.up, *self.down)):
            return np.ones(idx.size, dtype=bool)
        return cells.dead[idx] == 0

    def evaluate(self, dic: Dictionary, cells, idx: np.ndarray, t: float) -> np.ndarray:
        dead = cells.dead[idx] == 1

        def gather(rules):
            out = []
            for r in rules:
                v = dic.signal_values(r.signal, cells, idx, t).astype(float, copy=True)
                if not r.apply_to_dead:
                    v[dead] = 0.0  # PhysiCell zeroes signals that do not apply to dead cells
                out.append(v)
            return out

        hu = multivariate_hill(gather(self.up), [r.half_max for r in self.up],
                               [r.hill_power for r in self.up]) if self.up else 0.0
        u = self.base + (self.max_value - self.base) * hu
        if self.down:
            hd = multivariate_hill(gather(self.down), [r.half_max for r in self.down],
                                   [r.hill_power for r in self.down])
            return u + (self.min_value - u) * hd
        return u if isinstance(u, np.ndarray) else np.full(idx.size, u)


class Ruleset:
    """All rules of a model, compiled per cell type and behavior."""

    def __init__(self, model: Model, dic: Dictionary, strict: bool = False):
        self.model, self.dic = model, dic
        self.compiled: dict[str, list[CompiledBehavior]] = {}
        self.skipped: list[tuple[Rule, str]] = []

        for rule in model.rules:
            reason = None
            if rule.cell_type not in model.cell_type_names:
                reason = f"unknown cell type {rule.cell_type!r}"
            elif not dic.supports_signal(rule.signal):
                reason = f"signal {rule.signal!r} not implemented"
            elif not dic.supports_behavior(rule.behavior):
                reason = f"behavior {rule.behavior!r} not implemented"
            if reason:
                self.skipped.append((rule, reason))
                continue

            bucket = self.compiled.setdefault(rule.cell_type, [])
            cb = next((c for c in bucket if c.behavior == rule.behavior), None)
            if cb is None:
                base = dic.base_behavior(rule.cell_type, rule.behavior)
                cb = CompiledBehavior(rule.cell_type, rule.behavior, base, base, base)
                bucket.append(cb)
            if rule.direction == "increases":
                cb.up.append(rule)
                cb.max_value = rule.max_response
            else:
                cb.down.append(rule)
                cb.min_value = rule.max_response

        if strict and self.skipped:
            lines = [f"  {r.cell_type}: {r.signal} {r.direction} {r.behavior}  ({why})"
                     for r, why in self.skipped]
            raise NotImplementedError("rules this MVP cannot apply:\n" + "\n".join(lines))

    def apply(self, cells, type_index: dict[str, int], t: float):
        for cell_type, behaviors in self.compiled.items():
            idx = np.flatnonzero(cells.view("type_idx") == type_index[cell_type])
            if idx.size == 0:
                continue
            for cb in behaviors:
                live = cb.applies(cells, idx)
                if not live.any():
                    continue
                target = idx[live]
                values = cb.evaluate(self.dic, cells, target, t)
                self.dic.write_behavior(cb.behavior, cells, target, values)

    def report(self) -> str:
        n_applied = sum(len(v) for v in self.compiled.values())
        lines = [f"ruleset: {len(self.model.rules)} rules, "
                 f"{n_applied} behaviors applied, {len(self.skipped)} rules skipped"]
        for r, why in self.skipped:
            lines.append(f"  skipped: {r.cell_type} | {r.signal} {r.direction} {r.behavior}  -- {why}")
        return "\n".join(lines)

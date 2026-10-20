"""Experiment: which rules in a published CBHG model actually carry the result?

Two hypotheses, one validation run.

H1 (sparsity).  A published ruleset is a list of hypotheses, but not all of
them do work.  We predict that for a given model, fewer than a third of its
rules -- removed one at a time -- shift any reported outcome by more than a few
units of the model's own seed-to-seed noise.

H2 (one-run screening).  We predict the inert rules can be found without the
knock-out runs at all.  During one unperturbed run we measure each rule's
*in-run displacement*

    D_r = (1/sum_t N_t) * sum_t sum_{cells of that type} |b_with_r - b_without_r| / scale

i.e. how far, averaged over the whole population and the whole run, the rule
actually moves the behavior it writes.  We predict D_r ranks rules the same way
the knock-outs do, and in particular that D_r ~ 0 implies no knock-out effect,
so one instrumented run replaces N+1 runs.

Both are falsifiable: H1 fails if most rules matter, H2 fails if some rule with
D_r ~ 0 nevertheless changes the outcome (a false negative), which is what
would happen if the rules were strongly coupled through the microenvironment.

Design notes
  * knock-out = delete the rule from the ruleset, which is what "this
    hypothesis is not in the model" means.  Flattening max_response is not the
    same thing: several rules share one behavior's max, so flattening one
    rewrites the others.
  * the noise floor is measured, not assumed: R replicate seeds of the
    unperturbed model give sigma per metric, and two *sham* knock-outs (fresh
    seeds, nothing removed) measure the false-positive rate of the whole
    pipeline.
  * outcome = what these papers plot: live count per cell type, dead count,
    cumulative births and deaths, mean of each substrate, at the final time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import pyphysicell as pc                                    # noqa: E402
from pyphysicell.rules import CompiledBehavior              # noqa: E402

CORPUS = {
    "hypoxia":            "../published/grammar_samples/user_projects/hypoxia/config/PhysiCell_settings.xml",
    "tumor_immune_base":  "../published/grammar_samples/user_projects/tumor_immune_base/config/PhysiCell_settings.xml",
    "rules_sample":       "../PhysiCell/sample_projects/rules_sample",
    "immune_function":    "../PhysiCell/sample_projects/immune_function",
    "asymmetric_division": "../PhysiCell/sample_projects/asymmetric_division",
}


# ---------------------------------------------------------------- outcome ---
def outcome(sim) -> dict[str, float]:
    s = sim.scalars
    out = {f"live:{k}": float(v[-1]) for k, v in s.counts.items()}
    out["dead"] = float(s.dead[-1])
    out["births"] = float(s.births[-1])
    out["deaths"] = float(s.deaths[-1])
    for k, v in s.mean_substrate.items():
        out[f"sub:{k}"] = float(v[-1])
    return out


def build(path, scenario=None, drop: int | None = None):
    model = pc.load(path, scenario=scenario) if scenario else pc.load(path)
    if drop is not None:
        del model.rules[drop]
    return model


def simulate(path, scenario, drop, seed, until, probe=False):
    model = build(path, scenario, drop)
    sim = pc.Simulation(model, seed=seed)
    if sim.cells.n == 0:
        name = Path(path).name
        if name in pc.ports.available():
            pc.ports.install(sim)
        else:
            pc.initial.apply(sim)
    disp = attach_probe(sim, model) if probe else None
    sim.record()
    sim.run(until=until, record_every=max(until / 12.0, 30.0))
    res = outcome(sim)
    if probe:
        return res, disp.finish(model)
    return res


def pick_until(path, scenario, budget: float, steps: int = 20) -> float:
    """A run length this machine can do in roughly `budget` seconds.

    The models differ by two orders of magnitude in cost, and an ablation study
    is N+1 runs, so the duration has to come from a measurement rather than a
    constant.
    """
    model = build(path, scenario)
    sim = pc.Simulation(model, seed=0)
    if sim.cells.n == 0:
        name = Path(path).name
        pc.ports.install(sim) if name in pc.ports.available() else pc.initial.apply(sim)
    t0 = time.perf_counter()
    for _ in range(steps):
        sim.step()
    per_min = ((time.perf_counter() - t0) / steps) / model.time.dt_diffusion
    raw = budget / max(per_min, 1e-9)
    nice = [60, 120, 240, 360, 720, 1440, 2880, 4320, 7200]
    cap = min(model.time.max_time, 7200)
    return float(min(min((n for n in nice if n >= raw), default=cap), cap))


# ------------------------------------------------------- in-run displacement ---
class Displacement:
    """Per-rule |behavior with - behavior without|, accumulated over a baseline run."""

    def __init__(self, sim):
        self.sim = sim
        self.total = {}        # id(rule) -> summed |delta|/scale over cells & samples
        self.weight = 0.0      # summed population size over samples

    @staticmethod
    def _minus(cb: CompiledBehavior, rule) -> CompiledBehavior:
        up = [r for r in cb.up if r is not rule]
        down = [r for r in cb.down if r is not rule]
        return CompiledBehavior(cb.cell_type, cb.behavior, cb.base,
                                down[-1].max_response if down else cb.base,
                                up[-1].max_response if up else cb.base, up, down)

    def sample(self):
        sim = self.sim
        c, rs = sim.cells, sim.ruleset
        n_total = int(np.count_nonzero(c.view("dead") == 0))
        if n_total == 0:
            return
        self.weight += n_total
        for cell_type, behaviors in rs.compiled.items():
            idx = np.flatnonzero(c.view("type_idx") == sim.type_index[cell_type])
            if idx.size == 0:
                continue
            for cb in behaviors:
                live = cb.applies(c, idx)
                if not live.any():
                    continue
                target = idx[live]
                full = cb.evaluate(rs.dic, c, target, sim.t)
                scale = max(abs(cb.base), abs(cb.max_value), abs(cb.min_value), 1e-30)
                for rule in (*cb.up, *cb.down):
                    off = self._minus(cb, rule).evaluate(rs.dic, c, target, sim.t)
                    d = float(np.abs(np.asarray(full) - np.asarray(off)).sum()) / scale
                    self.total[id(rule)] = self.total.get(id(rule), 0.0) + d

    def finish(self, model) -> list[float]:
        w = max(self.weight, 1.0)
        return [self.total.get(id(r), 0.0) / w for r in model.rules]


def attach_probe(sim, model, every=None):
    d = Displacement(sim)
    sim.every(every or max(model.time.max_time / 200.0, model.time.dt_phenotype), lambda _s: d.sample())
    return d


# -------------------------------------------------------------- statistics ---
def rank(x):
    order = np.argsort(np.asarray(x, dtype=float), kind="stable")
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(len(x), dtype=float)
    # average ties
    v = np.asarray(x, dtype=float)[order]
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[j + 1] == v[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = np.mean(r[order[i:j + 1]])
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def effect(base_runs, test_runs, keys, rel_floor=0.02):
    """|change in the mean| in units of the model's own seed-to-seed spread.

    The floor matters more than the statistic.  A count that happens to be
    identical across the baseline seeds has sigma = 0, and without a floor one
    extra cell in the knock-out reads as an infinite effect -- which is how a
    *sham* knock-out (nothing removed, fresh seeds) scored 12.5 sigma in the
    first run of this experiment.  Counts therefore get an absolute floor of
    one cell, everything else a relative floor.
    """
    b = np.array([[r[k] for k in keys] for r in base_runs])
    t = np.array([[r[k] for k in keys] for r in test_runs])
    mu = b.mean(axis=0)
    sd = np.maximum(b.std(axis=0, ddof=1), t.std(axis=0, ddof=1) if len(t) > 1 else 0.0)
    counts = np.array([not k.startswith("sub:") for k in keys])
    floor = np.where(counts, 1.0, np.maximum(rel_floor * np.abs(mu), 1e-12))
    sigma = np.maximum(sd, floor)
    z = np.abs(t.mean(axis=0) - mu) / sigma
    return float(z.max()), keys[int(np.argmax(z))], z


# ------------------------------------------------------------------- study ---
def _job(args):
    path, scenario, drop, seed, until = args
    t0 = time.time()
    return drop, seed, simulate(path, scenario, drop, seed, until), time.time() - t0


def study(name, path, until, reps, base_reps, workers, scenario=None):
    model = build(path, scenario)
    rules = model.rules
    if not rules or not model.cell_types:
        raise SystemExit(f"{name}: {path} loaded {len(model.cell_types)} cell types and "
                         f"{len(rules)} rules -- wrong file, or a pre-1.6 XML")
    print(f"\n=== {name}: {len(rules)} rules, {len(model.cell_types)} cell types, "
          f"{until:g} min, {base_reps} baseline + {reps} per knock-out, {workers} workers")

    print("  instrumented baseline run (one run, measures D_r) ...", flush=True)
    t0 = time.time()
    _, disp = simulate(path, scenario, None, 1000, until, probe=True)
    print(f"  ... {time.time() - t0:.1f}s")

    jobs = [(path, scenario, None, 1000 + r, until) for r in range(base_reps)]
    jobs += [(path, scenario, None, 9000 + r, until) for r in range(3 * reps)]   # shams
    for i in range(len(rules)):
        jobs += [(path, scenario, i, 1000 + r, until) for r in range(reps)]

    done = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for n, (drop, seed, res, dt) in enumerate(ex.map(_job, jobs), 1):
            done.setdefault(drop if drop is not None else ("base" if seed < 9000 else "sham"),
                            []).append((seed, res))
            if n % 10 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} runs, {time.time() - t0:.0f}s elapsed", flush=True)

    base = [r for _s, r in sorted(done["base"])]
    keys = sorted(base[0])
    shams = [r for _s, r in sorted(done.get("sham", []))]
    sham_z = [effect(base, shams[i * reps:(i + 1) * reps], keys)[0]
              for i in range(len(shams) // reps)]

    rows = []
    for i, rule in enumerate(rules):
        z, worst, _ = effect(base, [r for _s, r in sorted(done[i])], keys)
        rows.append(dict(i=i, z=z, worst=worst, D=disp[i], rule=f"{rule.cell_type} | {rule.signal} "
                                                               f"{rule.direction} {rule.behavior}"))
    return dict(name=name, until=until, reps=reps, base_reps=base_reps,
                baseline={k: float(np.mean([r[k] for r in base])) for k in keys},
                baseline_sd={k: float(np.std([r[k] for r in base], ddof=1)) for k in keys},
                sham_z=sham_z, rows=rows)


def report(res, thresh=None):
    rows = sorted(res["rows"], key=lambda r: -r["z"])
    n = len(rows)
    # the shams are knock-outs of nothing: whatever they score is noise, so the
    # bar for "this rule matters" is set above them, never below.
    thresh = thresh if thresh is not None else max(3.0, 1.5 * max(res["sham_z"] or [0.0]))
    hits = [r for r in rows if r["z"] > thresh]
    quiet = [r for r in rows if r["D"] < 1e-3]
    missed = [r for r in quiet if r["z"] > thresh]
    rho = spearman([r["D"] for r in rows], [r["z"] for r in rows])
    print(f"\n--- {res['name']}  ({n} rules)")
    print(f"sham knock-outs (nothing removed): z = "
          f"{', '.join(f'{s:.1f}' for s in res['sham_z']) or 'n/a'}   <- the false-positive scale")
    print(f"{'z':>7} {'D_r':>9}  rule")
    for r in rows:
        mark = "*" if r["z"] > thresh else " "
        print(f"{r['z']:>7.1f} {r['D']:>9.2e} {mark} {r['rule']}  [{r['worst']}]")
    print(f"H1: {len(hits)}/{n} rules move an outcome by >{thresh:.1f} sigma "
          f"({100 * len(hits) / n:.0f}%)")
    print(f"H2: Spearman(D_r, z) = {rho:+.2f}; rules with D_r<1e-3: {len(quiet)}, "
          f"of which {len(missed)} have z>{thresh:.1f} (false negatives)")
    return dict(n=n, hits=len(hits), rho=rho, quiet=len(quiet), missed=len(missed),
                sham=res["sham_z"], thresh=thresh)


# ------------------------------------------------------------- the product ---
SAFE_CUTOFF = 3e-4
"""Below this in-run displacement, no rule in the validated corpus mattered.

Calibrated, not guessed: over 100 rule-measurements (5 models x 2 horizons) a
cutoff of 1e-3 let two live rules through, and 3e-4 let none through while
still clearing 42% of the ruleset.  Re-run `--curve` after adding models.
"""


def lint(name, path, until=None, budget=20.0, seed=1000, scenario=None):
    """One instrumented run -> the rules that are doing nothing.

    This is what the experiment bought: the inert rules of a model, for the
    cost of running it once instead of once per rule.  It is a lower bound on
    inertness -- a rule above the cutoff is not thereby important -- and it is
    only valid for the horizon it was run at, because the set of rules that
    matter grows with the window (see the README).
    """
    until = until or pick_until(path, scenario, budget)
    model = build(path, scenario)
    _, disp = simulate(path, scenario, None, seed, until, probe=True)
    quiet = [(d, r) for d, r in zip(disp, model.rules) if d < SAFE_CUTOFF]
    print(f"{name}: {len(model.rules)} rules over {until:g} min, "
          f"{len(quiet)} never moved their behavior by {SAFE_CUTOFF:g}")
    for d, r in sorted(quiet, key=lambda q: q[0]):
        print(f"  {d:>9.1e}  {r.cell_type} | {r.signal} {r.direction} {r.behavior}")
    return quiet


def curve(dirs):
    """The screen's false-negative rate against its yield, over saved studies."""
    rows = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.json")):
            r = json.loads(f.read_text())
            bar = max(3.0, 1.5 * max(r["sham_z"] or [0.0]))
            rows += [(x["D"], x["z"] > bar) for x in r["rows"]]
    print(f"{len(rows)} rule-measurements, {sum(m for _d, m in rows)} of them live\n")
    print(f"{'cutoff':>9} {'screened out':>13} {'yield':>7} {'false negatives':>16}")
    for cut in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5):
        q = [m for d, m in rows if d < cut]
        print(f"{cut:>9.0e} {len(q):>13} {len(q) / len(rows):>6.0%} {sum(q):>16}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=list(CORPUS))
    ap.add_argument("--until", type=float, default=0.0,
                    help="run length in simulated minutes; 0 = pick one per model "
                         "from --budget seconds of wall time")
    ap.add_argument("--budget", type=float, default=8.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--base-reps", type=int, default=6)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out", default="experiments/results")
    ap.add_argument("--lint", action="store_true",
                    help="one instrumented run per model: list the rules that do nothing")
    ap.add_argument("--curve", nargs="*", default=None,
                    help="read saved studies and print the screen's operating curve")
    a = ap.parse_args()

    if a.curve:
        curve(a.curve)
        raise SystemExit(0)
    if a.lint:
        for name in a.models:
            lint(name, CORPUS[name], a.until or None, a.budget)
        raise SystemExit(0)

    Path(a.out).mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in a.models:
        until = a.until or pick_until(CORPUS[name], None, a.budget)
        res = study(name, CORPUS[name], until, a.reps, a.base_reps, a.workers)
        Path(a.out, f"{name}.json").write_text(json.dumps(res, indent=1))
        summary[name] = report(res)
    print("\n================ summary ================")
    for k, v in summary.items():
        print(f"{k:<20} {v['hits']}/{v['n']} rules matter   rho={v['rho']:+.2f}   "
              f"silent={v['quiet']} missed={v['missed']}   sham z={['%.1f' % s for s in v['sham']]}")

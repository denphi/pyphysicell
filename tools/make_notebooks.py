"""Generate one self-contained notebook per PhysiCell sample project.

Each notebook carries its model as plain Python (pyphysicell.codegen), so it
needs no PhysiCell checkout, no XML and no CSV -- only the pyphysicell package.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import pyphysicell as pc  # noqa: E402
from pyphysicell.rules import Ruleset  # noqa: E402
from pyphysicell.signals import Dictionary  # noqa: E402

PC = ROOT.parent / "PhysiCell"
OUT = ROOT / "notebooks"
BUDGET_SECONDS = 8.0          # wall-clock target for the demo run in each notebook
CEILING_SECONDS = 15.0        # ... and the most we will spend to cover one cycle


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


def measure(model, steps: int = 20) -> float:
    """Seconds per simulated minute, measured on this machine."""
    sim = pc.Simulation(model, seed=model.random_seed)
    if not sim.cells.n:
        pc.initial.apply(sim)
    t0 = time.perf_counter()
    for _ in range(steps):
        sim.step()
    dt = (time.perf_counter() - t0) / steps
    return dt / model.time.dt_diffusion


def cycle_timescale(model) -> float:
    """One turn of the fastest cycle in the model, in simulated minutes.

    A demo window shorter than this cannot show a division, which is why 17 of
    37 notebooks used to screen every parameter they had and find that none of
    them moved anything: `template` was given 270 minutes of a 1080-minute
    cycle.  Closed form, no simulation needed.
    """
    spans = [ct.cycle.mean_duration() for ct in model.cell_types]
    finite = [v for v in spans if v < 1e12]
    return 1.2 * min(finite) if finite else 60.0


def pick_duration(model, per_minute: float) -> tuple[float, str]:
    """A window that covers one characteristic timescale, if we can afford it."""
    floor = cycle_timescale(model)
    cap = min(max(model.time.max_time, floor), 7200)
    affordable = CEILING_SECONDS / max(per_minute, 1e-9)
    want = min(max(BUDGET_SECONDS / max(per_minute, 1e-9), floor), cap, affordable)
    want = max(float(round(want, 2)), model.time.dt_phenotype)
    cycles = any(ct.cycle.mean_duration() < 1e12 for ct in model.cell_types)
    asks = f"the {model.time.max_time:g} min the project's own config asks for"
    if not cycles:
        note = (f"This demo covers {want:g} min. Nothing in this model divides on its own, so "
                f"the window is set by motility and transport rather than by a cycle")
        note += (f", and it is longer than {asks}." if want > model.time.max_time
                 else f", against {asks}.")
    elif want < floor * 0.99:
        note = (f"This demo covers {want:g} min. One turn of this model's fastest cycle is "
                f"{floor / 1.2:.0f} min and its own config asks for {model.time.max_time:g} min, "
                f"so what follows is a smoke test of the machinery, not a result: at this "
                f"window the population cannot divide.")
    else:
        note = (f"This demo covers {want:g} min of {asks} -- roughly "
                f"{want / (floor / 1.2):.1f} turns of its fastest cycle.")
    return want, note


def _final(model, until: float, seed: int = 0) -> dict:
    """Final aggregates of one run -- the screen's unit of measurement."""
    sim = pc.Simulation(model, seed=seed)
    if not sim.cells.n:
        pc.initial.apply(sim)
    sim.run(until=until, record_every=until)
    import numpy as np
    pos = sim.cells.view("position")
    return {"live": float((sim.cells.view("dead") == 0).sum()),
            "dead": float((sim.cells.view("dead") == 1).sum()),
            "births": float(sim._births), "deaths": float(sim._deaths),
            "spread": float(np.sqrt(((pos - pos.mean(axis=0)) ** 2).sum(axis=1).mean()))
            if sim.cells.n else 0.0}


def candidates(model, busiest, runnable):
    """Parameters this model exposes that a sweep could vary."""
    out = []
    if runnable:
        r0 = runnable[0]
        key = (r0.cell_type, r0.signal, r0.behavior, r0.direction)
        head = (f"def change(m, value):\n"
                f"    # the rules are data: edit one and re-run\n"
                f"    for r in m.rules:\n"
                f"        if (r.cell_type, r.signal, r.behavior, r.direction) == {key!r}:\n")
        out.append(dict(label=f"the half-max of '{r0.signal} {r0.direction} {r0.behavior}'"
                              f" for {r0.cell_type}",
                        base=r0.half_max, src=head + "            r.half_max = value",
                        apply=lambda m, v, k=key: [setattr(r, "half_max", v) for r in m.rules
                                                   if (r.cell_type, r.signal, r.behavior,
                                                       r.direction) == k]))
        if r0.max_response:
            out.append(dict(label=f"the saturated response of '{r0.signal} {r0.direction} "
                                  f"{r0.behavior}' for {r0.cell_type}",
                            base=r0.max_response,
                            src=head + "            r.max_response = value",
                            apply=lambda m, v, k=key: [setattr(r, "max_response", v)
                                                       for r in m.rules
                                                       if (r.cell_type, r.signal, r.behavior,
                                                           r.direction) == k]))
    ct = model.cell_type(busiest)

    def cycle_knob(name):
        c = model.cell_type(name).cycle
        k = next((k for k, v in c.transition_rates.items() if v > 0), None)
        if k is None:
            return None
        return dict(label=f"the cycle transition rate of {name}", base=c.transition_rates[k],
                    src=(f"def change(m, value):\n"
                         f"    m.cell_type({name!r}).cycle.transition_rates[{k!r}] = value"),
                    apply=lambda m, v, b=name, kk=k:
                        m.cell_type(b).cycle.transition_rates.__setitem__(kk, v))

    # the busiest type is not always the one that cycles: ode_energy's screen
    # found nothing when the count happened to favour a type with no cycle
    seen = set()
    fastest = min(model.cell_types, key=lambda c: c.cycle.mean_duration()).name
    for nm in (busiest, fastest):
        knob = cycle_knob(nm) if nm not in seen else None
        seen.add(nm)
        if knob:
            out.append(knob)
    for dname, dm in ct.deaths.items():
        if dm.death_rate > 0:
            out.append(dict(label=f"the {dname} rate of {busiest}", base=dm.death_rate,
                            src=(f"def change(m, value):\n"
                                 f"    m.cell_type({busiest!r}).deaths[{dname!r}].death_rate"
                                 f" = value"),
                            apply=lambda m, v, b=busiest, d=dname:
                                setattr(m.cell_type(b).deaths[d], "death_rate", v)))
            break
    if ct.motility.enabled and ct.motility.speed > 0:
        out.append(dict(label=f"the migration speed of {busiest}", base=ct.motility.speed,
                        src=(f"def change(m, value):\n"
                             f"    m.cell_type({busiest!r}).motility.speed = value"),
                        apply=lambda m, v, b=busiest:
                            setattr(m.cell_type(b).motility, "speed", v)))
    if ct.mechanics.cell_cell_adhesion_strength > 0:
        out.append(dict(label=f"the cell-cell adhesion of {busiest}",
                        base=ct.mechanics.cell_cell_adhesion_strength,
                        src=(f"def change(m, value):\n"
                             f"    m.cell_type({busiest!r}).mechanics.cell_cell_adhesion_strength"
                             f" = value"),
                        apply=lambda m, v, b=busiest:
                            setattr(m.cell_type(b).mechanics, "cell_cell_adhesion_strength", v)))
    if model.substrates:
        s0 = model.substrate_names[0]
        out.append(dict(label=f"the diffusion coefficient of {s0}",
                        base=max(model.substrate(s0).diffusion_coefficient, 1.0),
                        src=(f"def change(m, value):\n"
                             f"    m.substrate({s0!r}).diffusion_coefficient = value"),
                        apply=lambda m, v, n=s0:
                            setattr(m.substrate(n), "diffusion_coefficient", v)))
        out.append(dict(label=f"the decay rate of {s0}",
                        base=max(model.substrate(s0).decay_rate, 0.001),
                        src=(f"def change(m, value):\n"
                             f"    m.substrate({s0!r}).decay_rate = value"),
                        apply=lambda m, v, n=s0: setattr(m.substrate(n), "decay_rate", v)))
    return out


def pick_sweep(model, busiest, runnable, until: float):
    """Measure which candidate actually moves the model, and sweep that one.

    Picking the first rule's half-max (what this generator used to do) produced
    a flat sweep in 18 of 37 notebooks: most rules in a published model are
    inert at its own parameters.  So screen the candidates -- base, half, double
    -- and keep the one with the largest relative spread in any outcome.
    """
    import copy as _copy

    cands = candidates(model, busiest, runnable)[:6]
    base = _final(model, until)

    def swing_of(c, factor):
        m = _copy.deepcopy(model)
        c["apply"](m, c["base"] * factor)
        got = _final(m, until)
        worst = 0.0
        for k in base:
            denom = max(abs(base[k] + got[k]) / 2, 1.0 if k != "spread" else 1e-3)
            worst = max(worst, abs(got[k] - base[k]) / denom)
        return worst

    best, span = (0.0, cands[0]), 4.0
    for factor in (4.0, 0.25, 25.0):   # widen only while nothing has moved
        scored = [(swing_of(c, factor), c) for c in cands]
        for sw, c in scored:
            print(f"    screen x{factor:<5g} {c['label'][:46]:<48} swing {sw:6.1%}")
        best = max(scored, key=lambda sc: sc[0])
        span = max(factor, 1.0 / factor)
        if best[0] > 0:
            break
    # sweep the span the screen actually validated: sweeping +-2x after proving
    # the effect at +-4x is how ode_energy screened at 200% and swept flat
    return best[1], best[0], span


def build(project: Path, scenario: Path | None = None, label: str | None = None) -> dict:
    model = pc.load(scenario) if scenario is not None else pc.load(project)
    dic = Dictionary(model)
    rs = Ruleset(model, dic, strict=False)
    try:
        rel = (scenario or project).relative_to(PC.parent)
    except ValueError:
        rel = (scenario or project).relative_to(ROOT.parent)
    name = label or project.name
    data_dir = OUT / "data"
    csv_name = ""
    if model.initial_cells:
        csv_name = f"{name}_cells.csv"
        pc.write_cells_csv(data_dir / csv_name, model.initial_cells)
    src = pc.codegen.to_python(model, src=str(rel), initial="csv", csv_path=csv_name)
    per_min = measure(model)
    until, window_note = pick_duration(model, per_min)
    record = max(until / 12.0, model.time.dt_phenotype)
    nx, ny, nz = model.domain.shape

    skipped = sorted({why for _r, why in rs.skipped})
    cells, placement = pc.initial.auto(model)

    # intracellular models are a compile-time addon in PhysiCell and are not
    # implemented here; say which one the project expects
    xml = (scenario or project / "config" / "PhysiCell_settings.xml").read_text()
    for kind, label in (("maboss", "a MaBoSS Boolean network (PhysiBoSS)"),
                        ("roadrunner", "an SBML ODE model (libRoadrunner)"),
                        ("dfba", "a flux-balance model (dFBA)")):
        if f'type="{kind}"' in xml:
            skipped.append(f"intracellular model: {label} -- the cell-level model runs, "
                           f"its intracellular layer does not")

    # the busiest cell type: hooks and sweeps are aimed at it
    counts: dict[str, int] = {}
    for *_xyz, nm in cells:
        counts[nm] = counts.get(nm, 0) + 1
    order = {n: i for i, n in enumerate(model.cell_type_names)}
    busiest = (min(counts, key=lambda k: (-counts[k], order.get(k, 0)))
               if counts else model.cell_type_names[0])
    sub0 = model.substrate_names[0] if model.substrates else None

    runnable = [r for r in model.rules
                if dic.supports_signal(r.signal) and dic.supports_behavior(r.behavior)]
    short = max(round(until * 0.75, 2), model.time.dt_phenotype)
    chosen, swing, span = pick_sweep(model, busiest, runnable, short)
    swept, change_src = chosen["label"], chosen["src"]
    sweep_values = [float(f"{chosen['base'] * f:.4g}") for f in (1 / span, 1.0, span)]
    has_port = (label or project.name) in pc.ports.available()
    seed_src = ("print(pc.ports.install(sim))      # the project's custom.cpp, in Python"
                if has_port else
                "if not sim.cells.n:\n    print(pc.initial.apply(sim))")

    cells_list = [
        md(f"# {name}\n"
           f"\n"
           f"PhysiCell's `{rel}` as a **self-contained notebook**. The model below is Python,\n"
           f"generated from that project's XML and rules CSV; nothing here reads a PhysiCell\n"
           f"file, builds an executable or parses an output folder.\n"
           f"\n"
           f"| | |\n|---|---|\n"
           f"| domain | {nx} x {ny} x {nz} voxels, dx={model.domain.dx:g} um, "
           f"{'2D' if model.domain.use_2D else '3D'} |\n"
           f"| substrates | {', '.join(model.substrate_names)} |\n"
           f"| cell types | {', '.join(model.cell_type_names)} |\n"
           f"| rules | {len(model.rules)} ({len(model.rules) - len(rs.skipped)} runnable here) |\n"
           f"| initial cells | {len(cells)}"
           + (f", from `data/{csv_name}` |\n" if csv_name else " |\n")
           + (f"| custom C++ | ported to Python in `pyphysicell.ports.{project.name}` |\n"
              if has_port else "")
           + "\n"
           f"**Sections 3 to 7 are the point of the library**: each one is something the\n"
           f"XML-and-executable workflow cannot do at all, or can only do by editing C++,\n"
           f"recompiling and re-running from t=0.\n"),
        code("import pathlib\n"
             "import numpy as np, matplotlib.pyplot as plt\n"
             "import pyphysicell as pc\n"
             "from pyphysicell import viz\n"
             "\n"
             "# the data this notebook needs sits next to it\n"
             "DATA = next(p for p in (pathlib.Path('data'), pathlib.Path('notebooks/data'))\n"
             "            if p.exists())\n"
             "print('kernels:', pc.BACKEND, '|', pc.MAX_THREADS, 'threads', '| data:', DATA)"),

        md("## 1. The model is an object\n\n"
           "Not a document to be parsed at start-up: a value you can read, edit, copy, hash\n"
           "and generate. Everything below came from the project's XML."),
        code(src),
        code("print(model.summary())\n"
             "print('content hash:', model.hash, '-- changes with any edit, so runs are traceable')"),

        md("## 2. Answers before you simulate\n\n"
           "These cost milliseconds and need no run at all. `check()` is dimensional analysis\n"
           "of the model as written; the cycle duration is closed-form from the phase graph;\n"
           "a rule's response curve is a pure function of its signal."),
        code("print(pc.check(model))"),
        code("for ct in model.cell_types:\n"
             "    d = ct.cycle.mean_duration()\n"
             "    print(f'{ct.name:<28} cycle {d/60:8.1f} h' if np.isfinite(d)\n"
             "          else f'{ct.name:<28} does not cycle on its own')"),
    ]

    if runnable:
        cells_list += [
            code("d = pc.Dictionary(model)\n"
                 "shown = [r for r in model.rules if d.supports_signal(r.signal)\n"
                 "         and d.supports_behavior(r.behavior)][:3]\n"
                 "fig, axes = plt.subplots(1, len(shown), figsize=(4.5 * len(shown), 3))\n"
                 "for ax, r in zip(np.atleast_1d(axes), shown):\n"
                 "    viz.plot_rule(r, d.base_behavior(r.cell_type, r.behavior), ax=ax)\n"
                 "    print(r.english())\n"
                 "plt.tight_layout()"),
        ]

    cells_list += [
        md("## 3. Run it, and keep the simulation\n\n"
           "`sim` stays alive afterwards: its cells and fields are NumPy arrays you can read\n"
           "and change, and the run can be continued. The executable exits and leaves files."
           + ("\n\n" + window_note if window_note else "")),
        code("sim = pc.Simulation(model, seed=model.random_seed)\n"
             + seed_src + "\n"
             f"scalars = sim.run(until={until:g}, record_every={record:g})\n"
             "print(sim, f'| {sim.wall_time:.1f} s')\n"
             "sim.dataframe().head(3)"),
        code("fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n"
             f"viz.plot_cells(sim, substrate={sub0!r}, ax=axes[0])\n"
             "viz.plot_populations(scalars, ax=axes[1])\n"
             "plt.tight_layout()"),

        md("## 4. Steer it while it runs\n\n"
           "Change the state mid-run and carry on from where you are. In the file workflow\n"
           "this is a new XML and a new run from t=0, or custom C++ and a recompile."),
        code(f"before = sim.microenvironment[{sub0!r}].mean()\n"
             f"sim.microenvironment.density[..., 0] = 0.0      # remove {sub0} everywhere\n"
             f"sim.run(until=sim.t + {short:g}, record_every={record:g})\n"
             # NOT an f-string holding the same quote it is delimited with:
             # that is PEP 701, Python 3.12+, and these notebooks have to parse
             # on 3.8 (nanoHUB's default kernel)
             f"after = sim.microenvironment[{sub0!r}].mean()\n"
             f"print(f'mean {sub0} {{before:.3g}} -> {{after:.3g}}')\n"
             "viz.plot_populations(sim.scalars)\n"
             "plt.axvline(sim.scalars.times[0] / 60, ls=':', c='k')\n"
             "plt.title('one simulation, perturbed part-way through');"),

        md("## 5. Two futures from one state\n\n"
           "`branch()` is a deepcopy of the whole simulation, because all of the state is\n"
           "Python. PhysiCell keeps its state in process globals, so a second simulation in\n"
           "the same process is impossible there and a counterfactual means re-running from\n"
           "the beginning with a different input file."),
        code("base = pc.Simulation(model, seed=model.random_seed)\n"
             + ("pc.ports.install(base)\n" if has_port else
                "if not base.cells.n:\n    pc.initial.apply(base)\n")
             + f"base.run(until={short:g}, record_every={record:g})\n"
             "\n"
             "control = base.branch()\n"
             "treated = base.branch()\n"
             "treated.microenvironment.density[..., 0] *= 0.1   # 90% knock-down\n"
             f"control.run(until=control.t + {short:g}, record_every={record:g})\n"
             f"treated.run(until=treated.t + {short:g}, record_every={record:g})\n"
             "\n"
             "fig, ax = plt.subplots(figsize=(7, 4))\n"
             "for name, s in (('control', control.scalars), ('treated', treated.scalars)):\n"
             "    total = np.sum([v for v in s.counts.values()], axis=0)\n"
             "    ax.plot(np.asarray(s.times) / 60, total, lw=2, label=name)\n"
             "ax.set_xlabel('time [h]'); ax.set_ylabel('live cells'); ax.legend(); "
             "ax.grid(alpha=0.3);"),

        md("## 6. Logic the rules grammar cannot express\n\n"
           "A CBHG rule is a Hill function of one cell's own signals: it cannot see where a\n"
           "cell is, or how it ranks against the rest of the population. A hook is arbitrary\n"
           "Python over the whole population, called once per phenotype step on arrays, so\n"
           "both become possible -- and still no C++."),
        code(f"probe = pc.Simulation(model, seed=model.random_seed)\n"
             + ("pc.ports.install(probe)\n" if has_port else
                "if not probe.cells.n:\n    pc.initial.apply(probe)\n")
             + "log = []\n"
             + "\n"
             f"@probe.hook('phenotype', cell_type={busiest!r})\n"
             "def spatial_and_population_relative(ctx):\n"
             "    p = ctx.signals['pressure']\n"
             "    crowded = p >= np.quantile(p, 0.9) if ctx.n > 1 else np.zeros(ctx.n, bool)\n"
             "    left = ctx.position[:, 0] < 0\n"
             "    # population-relative: the most crowded tenth stops cycling\n"
             "    ctx.behaviors['cycle entry'] = np.where(crowded, 0.0,\n"
             "                                            ctx.behaviors['cycle entry'])\n"
             "    # spatial: cells in the left half of the domain stop migrating\n"
             "    ctx.behaviors['migration speed'] = np.where(left, 0.0,\n"
             "                                                ctx.behaviors['migration speed'])\n"
             "    log.append((ctx.t, int(crowded.sum()), float(left.mean())))\n"
             "\n"
             f"probe.run(until={short:g}, record_every={record:g})\n"
             "c = probe.cells\n"
             f"mine = c.view('type_idx') == probe.type_index[{busiest!r}]\n"
             "left = mine & (c.view('position')[:, 0] < 0)\n"
             "right = mine & (c.view('position')[:, 0] >= 0)\n"
             "print(f'hook fired {len(log)} times, one call per phenotype step for all "
             "{mine.sum()} cells')\n"
             "print(f\"  mean migration speed   left {c.view('motility_speed')[left].mean():.3f}"
             "   right {c.view('motility_speed')[right].mean():.3f}\")\n"
             "print(f\"  mean cycle entry rate  crowded decile is held at 0; "
             "population mean {c.view('cycle_entry_rate')[mine].mean():.3e}\")"),
        code("fig, ax = plt.subplots(figsize=(6, 6))\n"
             "viz.plot_cells(probe, color_by='pressure', ax=ax)\n"
             "ax.axvline(0, ls='--', c='k', lw=1)\n"
             "ax.set_title('left half frozen by the hook; colour is pressure');"),

        md("## 7. A sweep in one process\n\n"
           f"Varying {swept}, with replicates. No files, no subprocesses, no recompilation --\n"
           "the models and the simulations are objects, so N of them are just N objects.\n\n"
           f"*That parameter was not chosen by hand.* The generator screened every knob this\n"
           f"model exposes -- rule half-maxes and saturated responses, cycle and death rates,\n"
           f"migration speed, adhesion, substrate diffusion and decay -- at multiples of their\n"
           f"published values, and\n"
           + (f"kept the one that moved an outcome most ({swing:.0%} spread at x{span:g}). Most\n"
              f"knobs in a published model move nothing at its own parameters, which is itself\n"
              f"worth knowing before you spend a cluster on a sweep."
              if swing > 0 else
              f"found that **none of them moves this model at all**, at x4, x0.25 or x25.\n"
              f"The sweep below is therefore flat by construction, and that is the result:\n"
              f"nothing this library implements drives this project"
              + (", whose behaviour comes from an intracellular engine that does not run here"
                 if any("intracellular" in k for k in skipped) else "")
              + ". A flat sweep is worth seeing; a sweep of a knob nobody checked is not.")),
        code(change_src + "\n\n"
             f"df, runs = pc.sweep(model, change, {sweep_values!r}, until={short:g},\n"
             "                    replicates=2, progress=True)\n"
             "df[['value', 'replicate', 'cells', 'births', 'deaths', 'spread_um',\n"
             "    'mean_pressure', 'wall_s']]"),
        code("fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
             "pc.plot_sweep(df, 'cells', ax=axes[0], label='live cells')\n"
             "pc.plot_sweep(df, 'spread_um', ax=axes[1], label='spread (RMS radius)')\n"
             f"axes[0].set_xlabel({swept!r}); axes[1].set_xlabel({swept!r})\n"
             "plt.tight_layout()"),

        md("## 8. Stop when something happens\n\n"
           "A question like *when* does this model cross a level is a stopping condition\n"
           "here, and a post-hoc search through saved snapshots otherwise.\n\n"
           "The level is read off the run in section 3 rather than guessed, so the condition\n"
           "is an event this model really reaches. An earlier version of this notebook picked\n"
           "a threshold a priori and ran to its limit in 19 of 37 projects, demonstrating\n"
           "nothing."),
        code("import numpy as np\n"
             "h = sim.scalars\n"
             "t = np.asarray(h.times, dtype=float)\n"
             "# sections 4-6 steered this same sim, so calibrate on the untouched part\n"
             f"keep = t <= {until:g}\n"
             "t = t[keep]\n"
             "series = {}\n"
             "if h.counts:\n"
             "    series['live cells'] = np.asarray([sum(v) for v in zip(*h.counts.values())],\n"
             "                                      dtype=float)[keep]\n"
             "series['dead cells'] = np.asarray(h.dead, dtype=float)[keep]\n"
             "series['cumulative births'] = np.asarray(h.births, dtype=float)[keep]\n"
             "series['cumulative deaths'] = np.asarray(h.deaths, dtype=float)[keep]\n"
             "for _s, _v in h.mean_substrate.items():\n"
             "    series[f'mean {_s}'] = np.asarray(_v, dtype=float)[keep]\n"
             "\n"
             "def moved(v):\n"
             "    scale = max(float(np.abs(v).max()), 1e-12)\n"
             "    return max(abs(v[-1] - v[0]), v.max() - v.min()) / scale\n"
             "\n"
             "name, y = max(series.items(), key=lambda kv: moved(kv[1]))\n"
             "y0, lo, hi, end = float(y[0]), float(y.min()), float(y.max()), float(y[-1])\n"
             "if end != y0:                      # 60% of the way to where it ended\n"
             "    rising, level = end > y0, y0 + 0.6 * (end - y0)\n"
             "else:                              # it came back: aim at the far extreme\n"
             "    rising = (hi - y0) >= (y0 - lo)\n"
             "    level = y0 + 0.6 * ((hi - y0) if rising else (lo - y0))\n"
             "live = moved(y) > 0\n"
             "print(f'{sum(moved(v) > 0 for v in series.values())} of {len(series)} recorded "
             "observables moved')\n"
             "if live:\n"
             "    crossed = t[np.argmax((y >= level) if rising else (y <= level))]\n"
             "    print(f'watching {name}: the baseline crossed {level:.4g} at t = {crossed:.1f} min')\n"
             "else:\n"
             "    print('nothing this model records changes over the demo window')"),
        code("readers = {'live cells': lambda s: float((s.cells.view('dead') == 0).sum()),\n"
             "           'dead cells': lambda s: float((s.cells.view('dead') == 1).sum()),\n"
             "           'cumulative births': lambda s: float(s._births),\n"
             "           'cumulative deaths': lambda s: float(s._deaths)}\n"
             "for _s in model.substrate_names:\n"
             "    readers[f'mean {_s}'] = (lambda n: lambda s: float(s.microenvironment[n].mean()))(_s)\n"
             "read = readers[name]\n"
             "\n"
             "watch = pc.Simulation(model, seed=model.random_seed)\n"
             + ("pc.ports.install(watch)\n" if has_port else
                "if not watch.cells.n:\n    pc.initial.apply(watch)\n")
             + f"limit = {until * 4:g}\n"
             f"watch.run(until=limit, record_every={record:g},\n"
             "          stop_when=(lambda s: (read(s) >= level) if rising else (read(s) <= level))\n"
             "                    if live else None)\n"
             "print(f'stopped at t = {watch.t:.1f} min of a {limit:.0f} min limit, "
             "{name} = {read(watch):.4g}')\n"
             "if live:\n"
             "    assert watch.t < limit, 'the stopping condition never fired'"),
    ]

    return {"cells": cells_list,
            "metadata": {"kernelspec": {"display_name": "open-physicell (.venv)",
                                        "language": "python", "name": "open-physicell"},
                         "language_info": {"name": "python", "version": "3.12"}},
            "nbformat": 4, "nbformat_minor": 5}


def main(names: list[str] | None = None):
    OUT.mkdir(exist_ok=True)
    projects = []
    for pat in ("sample_projects/*", "sample_projects_intracellular/*/*"):
        projects += [d for d in sorted(PC.glob(pat))
                     if (d / "config" / "PhysiCell_settings.xml").exists()]
    if names:
        projects = [p for p in projects if p.name in names]
    rows = []
    for p in projects:
        try:
            nb = build(p)
        except Exception as e:                    # noqa: BLE001
            print(f"{p.name:<24} SKIPPED: {type(e).__name__}: {e}")
            rows.append((p.name, None, str(e)))
            continue
        path = OUT / f"{p.name}.ipynb"
        path.write_text(json.dumps(nb, indent=1))
        print(f"{p.name:<24} -> {path.relative_to(ROOT)}")
        rows.append((p.name, path, None))
    return rows


def published(limit_clean: bool = True):
    """A notebook per scenario of the published models under ../published."""
    from pyphysicell.rules import Ruleset
    from pyphysicell.signals import Dictionary

    base = ROOT.parent / "published"
    OUT.mkdir(exist_ok=True)
    made = []
    for cfg in sorted(base.rglob("PhysiCell_settings*.xml")):
        path = str(cfg)
        if any(skip in path for skip in ("/sample_projects", "/unit_tests", "/tests/",
                                         "/output/", "backup")):
            continue
        project = cfg.parent.parent
        label = (str(cfg.relative_to(base)).replace("/config/PhysiCell_settings", "")
                 .replace(".xml", "").replace("grammar_samples/user_projects/", "")
                 .replace("/", "_").strip("_"))
        try:
            model = pc.load(cfg)
            if not model.cell_types:
                print(f"{label:<34} skipped: no cell definitions in the XML (pre-1.6 style)")
                continue
            rs = Ruleset(model, Dictionary(model), strict=False)
            if limit_clean and rs.skipped:
                print(f"{label:<34} skipped: {len(rs.skipped)} rules not supported")
                continue
            nb = build(project, scenario=cfg, label=label)
        except Exception as e:                        # noqa: BLE001
            print(f"{label:<34} FAILED: {type(e).__name__}: {e}")
            continue
        out = OUT / f"pub_{label}.ipynb"
        out.write_text(json.dumps(nb, indent=1))
        made.append(out)
        print(f"{label:<34} -> {out.relative_to(ROOT)}")
    return made


if __name__ == "__main__":
    if sys.argv[1:2] == ["--published"]:
        published()
    else:
        main(sys.argv[1:] or None)

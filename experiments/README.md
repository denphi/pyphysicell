# Which rules in a PhysiCell model actually carry the result?

A CBHG ruleset is a list of hypotheses. A paper reports the list; nobody reports
which entries the conclusion rests on. Finding out means running the model once
per rule, which in the file-driven workflow is one edited XML, one subprocess
and one output folder per rule, so it is rarely done.

Here a rule is data and a simulation is an object, so the whole study is a loop
over `model.rules`. That made three hypotheses cheap enough to test.

    python experiments/rule_importance.py --budget 20 --reps 4 --base-reps 8
    python experiments/rule_importance.py rules_sample --lint      # the product
    python experiments/rule_importance.py --curve experiments/results*

## Method

**Knock-out.** Delete the rule from the ruleset. *Not* flatten its
`max_response`: several rules share one behavior's `max_value`, so flattening
one silently rewrites the others.

**Outcome.** What these papers plot -- live count per cell type, dead count,
cumulative births and deaths, mean of each substrate -- at the final time.

**Effect.** `z = |change in the mean| / sigma`, where sigma is the model's own
seed-to-seed spread, floored at one cell for counts. The floor is not cosmetic:
a count that happens to be identical across the baseline seeds has sigma = 0,
and the first run of this experiment scored a *sham* knock-out at 12.5 sigma
because of it.

**Controls.** 6-8 replicate seeds for the baseline, and three sham knock-outs
per model -- fresh seeds, nothing removed -- which measure the false-positive
rate of the whole pipeline. Every sham in the final runs scored below 2.0, so
the 3-sigma bar sits above the noise rather than being asserted to.

**Predictor.** In one instrumented baseline run, each rule's *in-run
displacement*

    D_r = (1 / sum_t N_t) * sum_t sum_cells |b_with_r - b_without_r| / scale

how far, averaged over the population and the run, the rule actually moves the
behavior it writes. One extra evaluation per rule per sample; no extra runs.

## Results

100 rule-measurements: 54 rules across 5 models at a short horizon, 46 of them
again at 1440 min. 39 measurements came out live.

### H1 -- a published ruleset is mostly inert. Supported, with a pattern.

| model | rules that matter (short) | at 1440 min |
|---|---|---|
| `asymmetric_division` | 3/3 | -- |
| `hypoxia` | 5/8 | **7/8** |
| `tumor_immune_base` | 6/11 | 6/11 |
| `rules_sample` | 3/27 | **9/27** |
| `immune_function` | 0/5 | -- |

Small rulesets are fully load-bearing; the 27-rule showcase is two-thirds inert
even at the longer horizon. `immune_function`'s 0/5 is weak evidence either way:
that model ends with ~28 cells, and against a one-cell noise floor only large
effects are visible.

### H3 -- the set of rules that matter grows with the observation window. Confirmed, and strictly nested.

Across three models and two horizons, **8 rules were gained and 0 were lost**.
`rules_sample` goes 3/27 to 9/27 between 360 and 1440 minutes; `hypoxia` 5/8 to
7/8. Nothing that mattered early stopped mattering later.

The mechanism is visible in the lint output: at 360 min, fourteen of
`rules_sample`'s rules have `D_r` exactly 0, because they are written for M1
macrophages, M2 macrophages and exhausted T cells -- cell types that only exist
once something has transformed into them. A rule whose cell type has no members
cannot do anything yet.

**So every statement of this kind needs its horizon attached.** "Two-thirds of
this ruleset is inert" is not a property of the model; it is a property of the
model and a window.

### H2 -- the inert rules can be found in one run. Falsified as stated, then calibrated.

The claim was that `D_r ~ 0` implies no knock-out effect, so one instrumented
run replaces N+1 runs. At the short horizon it held perfectly: 28 rules below
1e-3, zero false negatives. At 1440 min it broke -- two rules below 1e-3 moved
an outcome:

    M0 macrophage | necrotic debris increases transform to M1  D=3.2e-4  z=6.3
    malignant epithelial | oxygen increases cycle entry        D=6.5e-4  z=4.3

Both are rates. Displacement is instantaneous; a small persistent difference in
a transformation or cycle rate compounds into a different population. So the
predictor's units are wrong for anything that integrates.

The operating curve says where the screen is still safe:

| cutoff | rules screened out | false negatives |
|---|---|---|
| 1e-2 | 64% | 9 |
| 1e-3 | 48% | 2 |
| **3e-4** | **42%** | **0** |
| 1e-4 | 36% | 0 |

`SAFE_CUTOFF = 3e-4` clears 42% of the corpus with nothing lost. Re-run
`--curve` after adding models; the cutoff is calibration, not a constant.

The ranking half of H2 fails outright and is not worth rescuing. Spearman
between `D_r` and effect is +0.35 to +0.84 and once -1.00, because displacement
measures how far a rule moves *its own* behavior and cannot know that cycle
entry has more leverage on a population than a transformation rate does. In
`hypoxia`, `oxygen increases cycle entry` has `D_r = 2.7e-3` and z = 13.7 while
`oxygen decreases transform to motile tumor` has a `D_r` 185 times larger and
z = 9.2.

**One run proves a rule is dead. It cannot rank the live ones.**

## What this is good for, and what it is not

`--lint` is usable now: it names the rules that are doing nothing at a given
horizon, for the cost of one run. That is a real answer to "where do I start
reading this 36-rule model", and a real warning before someone sweeps a
parameter that cannot move anything.

It is a lower bound on inertness, not a certificate of importance: above the
cutoff means *unscreened*, not *important*. And the whole study measures
agreement with this library's own engine. It does not establish that PhysiCell
would rank the rules the same way -- that still wants the one check that has
never been run, a trajectory comparison against the compiled executable.

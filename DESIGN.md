# open-physicell — Design Document

> **Name (working):** open-physicell (distribution `open-physicell`, Python package `physicell`, C++ namespace `openphysicell`). Both names are free on PyPI as of 2026-10-18. See open question 1.
> **Status:** Draft v0.2 — 2026-10-19. An MVP exists in [open-physicell/](open-physicell/):
> kernels, model, simulation, hooks, rules, interactions, fibres, ports, 37 executed notebooks,
> 75 tests. Sections carry **As built** notes where the code and the design differ; everything
> else is still a target.
> **Upstream:** [PhysiCell](https://github.com/MathCancer/PhysiCell, https://github.com/bsc-life/PhysiCell-X/) 1.14.2, BSD-3-Clause (checked out at `./PhysiCell`).
> **Parent project:** [open-nemo](../open-nemo/DESIGN.md). This document follows its conventions: numbered ADRs in §4, layered architecture in §5, **As built** notes added as code lands.

---

## 1. Purpose

`open-physicell` is a Python library for building, running, inspecting and comparing agent-based
multicellular simulations, using **PhysiCell's C++ as a kernel library** rather than as an
application you compile.

Today a PhysiCell study is a C++ project: copy a sample project into the source tree, edit
`custom_modules/custom.cpp`, run `make`, run `./project ./config/PhysiCell_settings.xml`, and read
the output directory back with a separate Python tool. The simulation is a batch job with a text
interface, the model lives partly in XML and partly in C++, and two studies are comparable only if
someone re-derives what each one's custom code did.

open-physicell replaces that with:

- a **declarative model object** in Python (domain, substrates, cell types, rules, initial
  conditions, hooks) that round-trips to PhysiCell's own XML + rules CSV,
- a **stepable simulation handle** — `sim.step()`, `sim.run(until=…)`, read state as NumPy arrays,
  change it, continue — instead of a run-to-completion executable,
- **no compiler for most work**: prebuilt wheels carry the kernels; a C++ toolchain is needed only
  for models that genuinely need custom C++ (§6.6),
- a **comparison layer**: models are content-hashed and versioned, runs carry provenance, and
  published models are reproduced as benchmarks that can be re-run and diffed against each other,
- and **useful pieces that are not a simulation at all** (§11.6): the signal/behavior grammar of a
  model, a rule's response curve, the cycle time a parameter set implies, a diffusion-only solve, a
  single-cell 0-D trajectory. Most of these exist in PhysiCell already and are reachable only by
  compiling a program that prints them and exits.

The relationship to PhysiCell mirrors open-nemo's relationship to NEMO5, with one difference that
shapes the whole design: **NEMO5 was replaced, PhysiCell is not.** PhysiCell is actively maintained,
correct, widely used and well-validated. open-physicell wraps it, tracks its releases, and must stay
numerically identical to it (§13.1). This is a frontend project, not a rewrite.

## 2. Lessons from PhysiCell and its ecosystem

| Lesson | Evidence | Consequence for open-physicell |
|---|---|---|
| The compile step is the entry barrier, not the physics. | Every workflow starts with `make template` and a C++ edit; the Galaxy version of PhysiCell Studio ships only a precompiled template model and therefore cannot run custom C++ at all. | Prebuilt wheels with the standard model set; the compiler is a tier-C feature (§6.6), not a prerequisite. |
| Most published models do not actually need custom C++ any more. | The CBHG rules engine (`core/PhysiCell_rules.h`, settings `rules_enabled`) and the signal/behavior dictionary (`core/PhysiCell_signal_behavior.h`) express most phenotype logic as data; in `sample_projects/rules_sample`, all four hook functions are `{ return; }` and the model is a 40-line `cell_rules.csv`. | Rules are first-class Python objects compiled into the C++ rules engine; zero Python in the inner loop for rules-only models. |
| The model is split across XML, CSV and C++, with no single object that is the model. | `config/PhysiCell_settings.xml` (domain, substrates, cell definitions, user parameters), `config/cell_rules.csv`, `custom_modules/custom.cpp` (`create_cell_types`, `setup_tissue`, hooks). | One `Model` object owns all three; it serializes to exactly those files, so a Python model can still be run by the stock executable (ADR-006). |
| PhysiCell's state is process-global. | `microenvironment`, `default_microenvironment_options` (`BioFVM/BioFVM_microenvironment.h:362`), `PhysiCell_settings`, `PhysiCell_globals`, `parameters` (`modules/PhysiCell_settings.h:229`), `cell_defaults` (`core/PhysiCell_cell.h:139`), `all_cells` (`core/PhysiCell_cell_container.h:121`), `diffusion_dt`/`mechanics_dt`/`phenotype_dt` (`core/PhysiCell_constants.h:156`). | One simulation per process, enforced; ensembles are process pools (ADR-003). This is a hard constraint, not a preference. |
| The time loop already has a clean seam. | `main.cpp` is `microenvironment.simulate_diffusion_decay(diffusion_dt)` then `Cell_Container::update_all_cells(t)`, with overloads taking all three dt's (`core/PhysiCell_cell_container.h:104-107`). | The kernel exposes exactly that seam as `step()`; open-physicell owns the loop, the saving and the stopping conditions. |
| Agents are an array of structs, and so is the microenvironment. | `Cell : Basic_Agent` with `std::vector<double> position`, one heap vector per agent; densities are `std::vector<std::vector<double>>`, one vector per voxel. | There is **no zero-copy NumPy view**. Arrays are gathered copies with explicit write-back, and the gather cost is budgeted and measured (ADR-007). |
| The signal/behavior dictionary is already a flat, named, vector-valued API. | `get_signals(Cell*) -> vector<double>`, `set_behaviors(Cell*, vector<double>)`, `find_signal_index(name)`; dictionaries are built at runtime from the substrates and cell types actually present. | It is the natural Python surface: an (N_cells × N_signals) array with named columns, not an object graph of 30 nested phenotype classes. |
| Thirteen function pointers are the whole extension surface. | `Cell_Functions` (`core/PhysiCell_phenotype.h:492`): `update_phenotype`, `custom_cell_rule`, `update_velocity`, `contact_function`, `cell_division_function`, `volume_update_function`, … | Hooks are named, typed and enumerable; a Python hook is bound by name to one of those slots (ADR-005). |
| Calling Python per cell per step will not work naively. | `update_all_cells` runs its phenotype and mechanics loops under `#pragma omp parallel for`; the GIL serializes any callback and would also pin the OpenMP threads. | Default hooks are **vectorized**: one Python call per substep over all cells as arrays. Per-cell callbacks exist as a measured, documented escape hatch (ADR-005). |
| Reproducibility is thread-count dependent. | Per-thread RNG state (`physicell_random_seeds`, `core/PhysiCell_utilities.h:84`) plus `<random_seed>` in XML; cell ordering under OpenMP is not fixed. | Determinism policy is explicit: bitwise only at one thread and one build; everything else is distributional (ADR-010). |
| The output format is a dead end for analysis. | MultiCellDS v2 XML + MATLAB `.mat` per save, written to a folder, re-read by a separate package. | Results stay in memory as arrays; MultiCellDS is written when asked, for interoperability with Studio and `pcdl` (ADR-009). |
| Downstream Python tools lag upstream. | PhysiCOOL's documentation lists PhysiCell 1.10.4 as its tested version; the upstream is 1.14.2. | The kernel is pinned to a PhysiCell commit, the pinned version is in the package metadata and in every result's provenance, and upgrading upstream is a CI job that reruns the golden tests (§13.1). |

### 2.1 Prior art, and the gap this fills

| Tool | What it does | What it does not do |
|---|---|---|
| **PhysiCell Studio** (2.48.0) | GUI to edit the XML and rules, launch a compiled executable, scrub frames, plot populations over time, export an mp4, and browse UQ-PhysiCell databases; runs on nanoHUB. The best on-ramp for new users, and further along the artifact path than its reputation suggests (§12.5). | Not a library: you cannot script it, sweep it, step it, or call it from an optimizer. Its analyses are reconstructed from the file dump after the fact (the population plot loads every `output*.xml` through pyMCDS). Custom C++ still means a compiler, and the Galaxy build forbids it. |
| **pcdl / physicelldataloader** | Loads MultiCellDS output into Python, AnnData, dataframes, movies. Mature and good. | Read-only, post-hoc, file-based. Nothing about defining, running, steering or comparing models. |
| **PhysiCOOL** | Reads/writes the XML and runs calibration loops around the executable. | Drives the executable as a subprocess with files; no in-memory state, no stepping, no hooks, no custom-code story, and version lag. |
| **UQ-PhysiCell** (1.2.10) | The serious ensemble tool: an INI model spec, samplers (LHS, Sobol, one-at-a-time), multi-task and MPI parallel runs, Bayesian optimization and ABC-SMC calibration, summary statistics and distances, and SQLite result databases that store **either precomputed QoIs or raw MCDS**. Studio can browse its databases. | Still drives the compiled executable through files and INI edits: no model object, no in-memory state, no stepping, no Python hooks, and the custom-C++ barrier is untouched. |
| **PhysiBoSS / PhysiMeSS / dFBA / libRoadrunner addons** | Intracellular models (Boolean, mechanics, FBA, SBML ODE) built into `addons/`. | Each is another compile-time configuration. No uniform way to turn one on from Python. |
| **PhysiCell-X** and MPI forks | Distributed single large simulations. | Orthogonal: scaling one run, not making many runs programmable. |

The gap is narrower than it looks from the outside, and it is worth being precise about it. Studio
and UQ-PhysiCell between them already cover GUI authoring, frame scrubbing, population plots, movie
export, sampling, calibration and a result database. **What none of them has is the thing underneath:
a model object and a live simulation.** Every one of them drives the compiled executable through
files, so every one of them inherits the compile step, the file round trip and the post-hoc
reconstruction of anything the run did not already write.

So the gap is: in-memory state, a programmable loop, hooks in Python, a first-class `Model`, and
provenance strong enough to compare two studies. That is also the part that is hardest to retrofit,
because it is determined by how the kernel is wrapped — and it is the part that would make Studio
and UQ-PhysiCell better if they sat on top of it, which is the relationship to aim for (§12.5,
open question 8).

## 3. Goals and non-goals

### Goals
1. **No compiler for most users.** `pip install open-physicell`, then run, modify and compare models.
2. **Numerically identical to PhysiCell.** A model run through open-physicell matches the stock
   executable on the same XML, seed and thread count, bitwise where §13.1 says bitwise.
3. **Programmable.** The simulation is an object: step it, read it, change it, branch it, stop it on
   a Python condition.
4. **Modular.** Substrates, cell types, rules, hooks, initial conditions and intracellular models are
   separately constructible, serializable and swappable.
5. **Comparable.** Two models, or the same model at two parameter points, can be run and diffed with
   a shared metric protocol and full provenance.
6. **Honest about cost.** Every Python-in-the-loop feature carries a measured overhead number in the
   docs.
7. **Useful in pieces.** The partial flows of §11.6 are first-class entry points, not internals, and
   the ones that need no kernel ship first.

### Non-goals (for v1)
- Rewriting PhysiCell's physics. Kernels stay upstream's; open-physicell's own C++ is glue.
- Distributed single simulations (MPI domain decomposition) **in our own kernels**. Ensembles are
  MPI-parallel; one simulation is one process with OpenMP. §8.3 analyzes what domain decomposition
  would take and ADR-013 reserves the seam for driving someone else's MPI kernel.
- A GUI. PhysiCell Studio exists and can keep editing the XML open-physicell reads and writes.
- GPU kernels. Possible later (§9.3), not in v1.
- Windows tier-C builds (compiling user C++ on Windows). Wheels and tiers A/B work there.

## 4. Key decisions

Each decision is a short ADR. A change is a new, dated ADR; an amended ADR keeps its text and names
the one that amends it. Where the code does not yet match, the section carries an **As built** note
with its date.

### ADR-001 — Python is the API; PhysiCell C++ is the kernel library

**Decision:** the public, versioned interface is the Python package. The C++ layer is upstream
PhysiCell plus a thin `openphysicell` shim, bound with **nanobind**, and is internal: its signatures
may change in any release.

**Rationale:** this is open-nemo's ADR-013 adopted on day one instead of after v0.1. Users are
modelers; the expensive loops are already C++; writing every component twice doubles the validation
surface. Anyone who wants the C++ API has it — it is PhysiCell, unmodified.

**Consequences:** no installed headers, no exported CMake package in v1. Contributors add features in
Python against the protocols of §7, and move a loop into C++ only when a profile justifies it.

### ADR-002 — Upstream is vendored unmodified, as a pinned submodule

**Decision:** PhysiCell is a git submodule pinned to a release tag. open-physicell adds files, it does
not edit them. Anything that cannot be done from outside becomes (a) a small, separately listed patch
in `patches/`, each with an upstream issue or PR link, applied at build time, or (b) a feature
request upstream.

**Rationale:** a fork would make "identical to PhysiCell" unverifiable and would strand the project
at one version. The patch list is the honest measure of how much the design depends on changes
upstream has not accepted; it must stay short and shrink.

**Known patch candidates (to be confirmed by the spike, §18.0):**
- a destructor/reset path that returns the globals **and the 607 function-level statics** (ADR-014)
  to a clean state in-process (otherwise ADR-003's subprocess isolation is the only option, which is
  acceptable);
- making `main`'s loop body reachable without `main` (believed unnecessary: `simulate_diffusion_decay`
  and `update_all_cells` are already public);
- optional SoA mirrors of position/velocity for cheap gathers (§9.2, a performance item only).

### ADR-003 — One simulation per process; ensembles are process pools

**Decision:** a `Simulation` owns the process-global PhysiCell state. The library enforces one live
`Simulation` per process. By default `Simulation` is a **handle to a worker subprocess**; the
in-process mode (`Simulation(..., inprocess=True)`) is available for debugging and for single-run
scripts where the IPC cost of hooks matters.

**Rationale:** §2's globals make two simulations in one interpreter impossible without a fork of
upstream. Subprocess isolation also gives crash containment (a segfault in a tier-C plugin kills a
worker, not the notebook), clean teardown without a reset path, and a natural unit for ensembles.

**Consequences:**
- The hook transport (§6.5) must work across a process boundary: arrays go over shared memory
  (`multiprocessing.shared_memory`), control messages over a pipe. In-process mode uses the same
  code path with the copies elided.
- A hook that touches a large closure pays pickling once at setup, not per step.
- `Ensemble` (§8) owns a pool of workers, a scheduler and per-run checkpointing.

### ADR-004 — The step seam is `simulate_diffusion_decay` + `update_all_cells`

**Decision:** open-physicell owns the time loop in Python-visible terms. The kernel exposes
`step(dt_diffusion)` doing exactly what `main.cpp` does between saves, plus `advance_to(t)` that runs
many steps without returning to Python. Saving, SVG output, stopping conditions and progress
reporting are Python's, not the kernel's.

**Rationale:** it is the one seam that needs no upstream change, and it is where every interactive
feature (stepping, live plotting, steering, early stopping, checkpointing) comes from.

**Consequence:** `main.cpp` is not used at all. Its setup sequence (config load → microenvironment →
cell container → cell types → tissue) is reproduced by the shim in `setup()`, and a test asserts the
two produce identical state (§13.1).

### ADR-005 — Three tiers of model logic, and the GIL rule

**Decision:** phenotype and behavior logic can be expressed in three ways, in this order of
preference:

1. **Rules (CBHG).** Python `Rule` objects are compiled into the C++ rules engine via `add_rule` /
   `set_hypothesis_parameters`. No Python runs during the simulation. Full speed.
2. **Vectorized hooks.** A Python callable invoked **once per phenotype substep** with arrays over
   all cells: signals `(N, S)`, behaviors `(N, B)`, positions `(N, 3)`, plus typed extras. It returns
   (or writes into) a behaviors array; the kernel scatters it back. One GIL acquisition per substep,
   OpenMP untouched inside the kernel.
3. **Per-cell callbacks.** A Python function bound to a `Cell_Functions` slot for one cell type. The
   kernel collects the cells needing it and calls them **serially, outside the OpenMP region**.

**Rationale:** (2) is the design's core bet — that "modular in Python" means NumPy over the whole
population at a 6-minute phenotype step, not a Python call per cell per 0.01-minute diffusion step.
At a typical `dt_phenotype = 6` min and `dt_diffusion = 0.01` min, a vectorized hook runs 600× less
often than the diffusion solver, so even a millisecond-scale hook is noise. (3) is honest about the
cases that need it and loud about the cost.

**Consequences:**
- The docs state a measured overhead per tier, per population size, on the benchmark machines.
- Tier 3 is rejected by default above a configurable cell count unless explicitly allowed.
- Hooks are pure functions of their arguments plus their own state; they must not call back into
  `sim` (checked: the handle is not passed to them).

### ADR-006 — Central data object: the `Model`, which round-trips to PhysiCell files

**Decision:** one object, `Model`, holds domain, substrates, cell definitions, rules, initial
conditions, user parameters, save options and hook bindings. `Model.to_files(dir)` writes a valid
`PhysiCell_settings.xml` + `cell_rules.csv` (+ a generated `custom.cpp` for hooks that are expressible
in C++), and `Model.from_files(dir)` reads an existing project back.

**Rationale:** this is the comparison story and the trust story at once. A model that can be written
out and run by the stock executable is a model a PhysiCell user can check, a Studio user can open,
and a reviewer can reproduce without installing this library.

**Consequence:** every Python-only feature must either serialize into the PhysiCell files or be
flagged in `Model.portability_report()` as a reason the model is not portable.

### ADR-007 — Arrays are gathered copies with explicit write-back

**Decision:** `sim.cells.positions`, `sim.cells.signals`, `sim.substrate("oxygen")` and friends return
**new NumPy arrays**, gathered from AoS storage. Writes go through explicit calls
(`sim.cells.set_behaviors(arr)`, `field[...] = …; field.write()`), never through a returned array.
Vectorized hooks (ADR-005 tier 2) receive buffers the kernel owns and reuses, which are valid only for
the duration of the call.

**Rationale:** §2's layout makes views impossible. Pretending otherwise would produce silent
corruption; the API says plainly what is a copy.

**Consequences:**
- Gather cost is O(N_cells × N_columns) per access and is measured; `sim.cells.signals` on 10⁵ cells
  with ~10² signals is ~10⁷ doubles, ~80 MB — so column selection
  (`sim.cells.signals["oxygen", "pressure"]`) is the recommended form, and the full-matrix accessor
  warns above a threshold.
- A future SoA mirror upstream (ADR-002 patch candidate) would turn the hot ones into views; the API
  does not change if it lands.

### ADR-008 — Custom C++ becomes a cached, discoverable plugin

**Decision:** an existing PhysiCell project's `custom_modules/` can be compiled into a shared library
against the installed kernel headers and loaded at runtime:
`physicell.plugins.build("path/to/user_project") -> Plugin`, cached by content hash in a user cache
dir, discoverable by entry point so third parties can ship wheels of their own plugins. A plugin
registers named functions that `Model` binds to `Cell_Functions` slots.

**Compatibility tiers**, reported by `physicell.inspect(project_dir)`:
- **Tier A — XML + rules only.** Runs on the prebuilt wheel. No compiler.
- **Tier B — custom code that maps onto supported hooks or onto an `InitialCondition`.** `inspect`
  names each custom function, the slot it fills, and whether an equivalent Python hook or rule
  exists. Porting is a documented, mostly mechanical edit. The common case is `setup_tissue`: even
  `rules_sample`, whose hooks are all empty, places cells in C++ (uniform random placement of
  `number_of_cells` per type), which is `pc.initial.Uniform(...)` or an initial-conditions CSV.
- **Tier C — substantial custom C++** (new cell classes, new solvers, addon-dependent models). Needs
  a compiler once; then it is a normal Python object.

**Rationale:** the corpus of existing PhysiCell research is the reason to build this library at all
(§1, "compare research projects"). A design that cannot ingest existing projects is a design for
future models only.

**Risk:** a plugin ABI pins the compiler and the kernel build. **Mitigation:** plugins are built
against the installed wheel's headers with the wheel's recorded flags, the kernel records an ABI tag,
and a plugin built for another tag is rejected with a rebuild instruction rather than loaded.

**Amended by ADR-015 (2026-10-19):** the plugin is dropped. It only makes sense in the
facade architecture, where PhysiCell's object model still exists for a project's
`custom.cpp` to bind to. The kernels-and-Python architecture has no `Cell` to hand that
code, so the corpus is **ported**, not ingested.

### ADR-015 — Port the custom C++ instead of loading it (2026-10-19)

**Context:** measured, not estimated. Ignoring the empty `{ return; }` stubs the template
ships, the real custom logic in every sample project totals **980 lines** — a median of
about 65 lines per project, the largest being `interactions` at 277 and `celltypes3` at
219. It is arithmetic over cell state: Hill and linear responses of local substrate
levels, neighbour counts and pressure.

**Decision:** no C++ plugin. Projects are ported into `physicell.ports.<project>`, one
module per project, each reproducing that project's `custom.cpp` hooks and its
`setup_tissue`. Eleven are done: biorobots, cancer_biorobots, cancer_immune, celltypes3,
custom_division, heterogeneity, interactions, mechano, pred_prey_farmer, virus_macrophage,
worm. PhysiCell's own reusable pieces live in `physicell.standard`
(`update_cell_and_death_parameters_O2_based`, the Hill and linear response functions).

**Rationale:** a plugin ABI, a compiler dependency and a second extension mechanism cost
more to maintain than 980 lines cost to read. The ports are also the honest test of the
hook design: anything that would not port is a missing hook slot, and porting these
eleven is what produced the `custom_rule`, `migration_bias`, `contact`, `touch` and
`division` slots, plus attachments.

**Consequences:**
- The ingest argument weakens: a tier-C project must be read and ported, not dropped in.
  With rules at 100 % and five hook slots, that is 3 of 22 sample projects.
- What still cannot be ported is a different data model, not a longer function:
  PhysiMeSS's fibres (`instantiate_cell`, a custom `update_velocity`) need a new kernel.
- Intracellular models never needed PhysiCell C++: `libroadrunner`, `cmaboss`/`maboss`
  and `cobra` are on PyPI, so PhysiBoSS-style models are a dependency and a per-cell
  dispatch, not a compile step.

### ADR-009 — Results are arrays first, MultiCellDS on request

**Decision:** a run produces a `Result` holding time series and snapshots as arrays (`xarray` where
it helps), with `.provenance`, `.save()` (Zarr or HDF5 for fields, Parquet for cell tables) and
`.plot()`. MultiCellDS v2 XML/MAT output is a save option, written by upstream's own writer, for
Studio and `pcdl`. `pcdl` is an optional dependency for reading existing output, never required.

**Rationale:** mirrors open-nemo ADR-008. Writing 10⁵ cells to XML every 60 simulated minutes is a
large share of wall time in long runs and produces files nothing in this stack wants to read back.

**Provenance contents:** open-physicell version, pinned PhysiCell commit and version, kernel ABI tag
and compile flags, model content hash, resolved parameter values, RNG seed, OpenMP thread count,
partition and rank count (ADR-013), plugin hashes, platform, and the script or notebook cell that
ran it. §12 specifies the artifact tiers this produces and the rule that every rendering is
reproducible from the stored data without re-running.

### ADR-010 — Determinism policy

**Decision:**
- **Bitwise reproducible:** same wheel, same seed, `threads=1`, same model hash. Tested in CI.
- **Statistically reproducible only:** anything with `threads > 1`, because per-thread RNG draws
  (`physicell_random_seeds`) and cell ordering depend on the thread schedule.
- Every `Result` records seed and thread count; `Ensemble` defaults to **n replicates** rather than
  one run, and comparison metrics (§10.3) operate on distributions.

**Rationale:** silently non-reproducible results are worse than slow ones, and a comparison library
that compares single stochastic runs is a comparison library that lies.

### ADR-011 — Packaging: portable wheels with a bundled OpenMP runtime

**Decision:** wheels for Linux x86_64/aarch64 and macOS arm64, CPython 3.11–3.14, built with
scikit-build-core, bundling `libomp`/`libgomp` (via auditwheel/delocate). The default build target is a
portable baseline (`-O3`, no `-march=native`), **not** upstream's `-march=native`. An optional
`physicell build-kernel --native` rebuilds the kernel locally for the host CPU, and the docs carry the
measured difference on the benchmark models.

**Rationale:** `-march=native` in a wheel is not shippable; omitting it silently would be a hidden
regression against people's expectations from `make`. Measure it and give the opt-in.

**Risk:** macOS OpenMP packaging is the known-fragile part (Apple clang ships no OpenMP; upstream's
Makefile special-cases arm64). The spike must build a working macOS arm64 wheel before anything else
is promised.

### ADR-012 — The comparison layer is part of the library, not a script

**Decision:** `physicell.compare` provides: a **model registry** (content-addressed models with
metadata and a citation), a **metric protocol** (`Metric.compute(Result) -> xarray`), built-in
metrics (population over time by type, spatial statistics, substrate profiles, survival/extinction,
morphology), and a **comparison object** that aligns runs on time and reports differences with
replicate-aware statistics.

**Rationale:** this is the feature the parent project calls benchmarks (open-nemo §13, `book/15_benchmarks`)
and it is the deliverable that makes the library worth building for a research group rather than for
one study.

### ADR-013 — Two levels of MPI: ensembles now, domain decomposition behind a reserved seam (2026-10-18)

**Context:** PhysiCell is OpenMP-only. §8.3 analyzes every algorithm in the time loop for
distributed-memory decomposability. The summary: the microenvironment decomposes with standard
techniques, the agent model decomposes with standard MD-style techniques **except** for pointer-based
cell–cell links and cross-cell writes, and load balance is the practical limiter. Two existing
projects have done parts of it (BioFVM-X, BioFVM-B, PhysiCell-X) with published numbers that are
modest for agents and good for diffusion alone.

**Decision:**
- **MPI-E (ensembles) is supported in v1.** An `mpi4py` scheduler maps runs onto ranks; each rank
  runs whole simulations. Efficiency is ~1 by construction; this is where the group's compute
  actually goes (sweeps, replicates, calibration).
- **MPI-D (one simulation across ranks) is not implemented by open-physicell.** If it is wanted, the
  route is to drive an existing MPI kernel (the PhysiCell-X / BioFVM-B lineage) behind the same
  Python API, not to write a distributed PhysiCell ourselves.
- **The seam is reserved now, not later.** A `Partition` object exists from v1 with one
  implementation (`Partition.single()`), and the API is written so that `Partition.slabs(axis, n,
  comm)` can be added without breaking anything: state access is a collective by contract, hooks
  declare their communication needs, and `Result` records the partition.

**Rationale:** open-nemo's ADR-002 learned that a parallel layout retrofitted into an API is a
rewrite, and its §8 hierarchy is exactly the shape needed here (ensemble → run → threads → domain).
Reserving the seam costs a few lines of API discipline; not reserving it costs a redesign.

**Consequences:**
- `sim.cells.positions` is documented as **collective** — valid only when every rank calls it — with
  `local=True` for the rank-local subset. Under `Partition.single()` the two are identical, so the
  contract is free today and correct later.
- A vectorized hook (ADR-005) is defined as operating on **the cells of this rank**, with explicit
  reductions (`ctx.allreduce(...)`) for anything population-global. A hook that silently assumes it
  sees every cell is a hook that breaks under MPI-D; the docs say so from v1 and the test suite runs
  hooks under a two-rank fake partition.
- `Result.provenance` records the partition and rank count alongside seed and thread count (ADR-010).

### ADR-014 — The C++ facade is the product; Python is its first consumer (2026-10-18)

**Context:** ADR-001 and ADR-013 made Python the API and C++ an internal kernel detail. Two measured
facts argue for inverting that:

- **The ecosystem copy-pastes the driver loop.** There are 24 `main.cpp` files across
  `sample_projects/` and `sample_projects_intracellular/`, 4,282 lines in total, and they are nearly
  identical: ignoring whitespace, `biorobots` differs from `template` by 11 lines and `worm` by 9.
  Every project carries its own copy of the same ~250-line setup, save and time loop so it can change
  about ten lines.
- **The people whose adoption matters write C++.** A 6,000-line Python package is not something a
  research group with no developer will adopt. A self-contained pair of C++ files that deletes 24
  copies of a loop is something they can read in an afternoon and merge in a day.

**Decision:**
- The layer is a **C++ facade**, `openphysicell::Simulation`, over unmodified PhysiCell (ADR-002
  unchanged): explicit lifecycle, the step seam of ADR-004, flat-buffer state access, hook
  registration, and the tier-0/1 artifact writers of §12.3.
- **Python, the MCP server and any other consumer sit on top of it.** Python remains the API *for
  Python users* (ADR-001's intent) but is no longer the only interface, and it holds no logic that
  belongs in the facade.
- The facade is written to be **mergeable upstream** as `core/PhysiCell_simulation.{h,cpp}` plus a
  ten-line `main.cpp` template, and is offered that way (§18.3).

**Consequences and limits:**
- **One simulation per process still holds, and the facade cannot fix it from outside.** `core/`,
  `modules/` and `BioFVM/` hold 607 function-level `static` locals — 348 in
  `PhysiCell_signal_behavior.cpp` alone, the `static int index = find_behavior_index(...)` caching
  idiom — whose values derive from the first simulation's substrate and cell-type lists. A second
  in-process simulation with different substrates would read stale indices. ADR-003 stands; a
  `reset()` that clears them is an upstream patch (ADR-002), not facade work.
- Gather/scatter and hook dispatch live in C++, so the per-cell language-boundary crossing that
  ADR-005 worries about never happens: Python sees whole-population buffers.
- `cpp/shim/` in §16 becomes `cpp/facade/`, and it is public, documented and tested rather than
  internal glue.

```cpp
namespace openphysicell {

class Simulation {
 public:
  explicit Simulation(const Config& cfg);        // what main.cpp's first 60 lines do
  ~Simulation();                                  // teardown; one live instance per process

  void step();                                    // diffusion_dt: the ADR-004 seam
  void advance_to(double t);                      // many steps without returning to the caller
  double time() const;

  // state: flat, caller-owned buffers (ADR-007)
  std::size_t num_cells() const;
  void gather(std::span<const Column> cols, std::span<double> out) const;
  void scatter_behaviors(std::span<const int> idx, std::span<const double> values);
  const Field& substrate(std::string_view name) const;

  // extension points: one dispatcher per Cell_Functions slot, keyed by cell type
  void set_hook(Slot slot, std::string_view cell_type, Hook fn);

  // artifacts (§12.3)
  Scalars scalars() const;                        // tier 1, computed each phenotype step
  void write_snapshot(const SnapshotPolicy& p) const;
};

}  // namespace openphysicell
```

### ADR-016 — The built architecture: Python owns the state, C++ is stateless kernels (2026-10-19)

**Context:** ADR-014 proposed a C++ facade over unmodified PhysiCell. The MVP was built the other
way, and the two are incompatible: PhysiCell's process globals are exactly what makes a facade
cheap and in-process ensembles impossible.

**Decision:** the state lives in Python as struct-of-arrays NumPy (cells, fields, fibres); the C++
kernels are stateless functions over those buffers (`python/physicell/kernels.cpp`, ~550 lines,
C ABI, ctypes, no build step for the user). Everything else — cycle, volume, death, interactions,
rules, ports, analysis — is Python.

**What it bought, measured:** two simulations in one process; `branch()` as a deepcopy; `pc.sweep`
over models and replicates with no files or subprocesses; checkpoints by pickle; hooks that see
whole-population arrays. None of these is reachable through PhysiCell's globals.

**What it cost:** "it is PhysiCell" is no longer free. The microenvironment is still bit-for-bit
BioFVM (4.6e-14 against BioFVM itself as the oracle, §13.1), but the cell model is a
reimplementation checked against PhysiCell's formulas and against the published corpus, not against
a PhysiCell run. Closing that gap needs the side-by-side run §13.1 describes.

**Consequence for ADR-008/ADR-014:** the facade and its plugin remain the alternative if ingesting
C++ unchanged matters more than in-process state. They are not combinable.

### ADR-017 — Kernels keep a persistent worker pool (2026-10-19)

**Decision:** the non-OpenMP path uses one process-wide pool, not a pool per call.

**Why it is an ADR and not a detail:** the kernels are called hundreds of thousands of times in a
run. Creating 16 threads per call exhausted the process thread limit mid-sweep
(`thread constructor failed`) and cost more than the work: `rules_sample` went from 6.5 s to 3.2 s
per 60 simulated minutes, `worm` from 1.4 s to 0.26 s, with identical results.

### ADR-018 — Extended agents are their own container and kernels (2026-10-19)

**Decision:** fibres (PhysiMeSS's ECM) are segments, not spheres, so they get `physicell.fibres`
(a separate SoA) and two kernels — fibres pushing cells, cells pushing and turning fibres — using
the distance to the nearest point on the segment with PhysiCell's force law. Cells and fibres are
accumulated in separate passes so neither output array is written by two threads.

**Rationale:** this is the line between a hook and a kernel. A hook is arithmetic over existing
state; a new geometry is new state and new inner loops. Anything else that needs its own data model
(vessels, membranes) follows the same route.

### ADR-019 — Three visualization backends, matplotlib first (2026-10-19)

**Decision:** `viz3d.show(sim)` defaults to **matplotlib** (renders anywhere, survives nbconvert,
goes into a paper), `backend="speck"` uses **ipyspeck** (ambient occlusion and outlines, best
looking, needs a live kernel), `backend="html"` writes a **self-contained WebGL page** with
impostor-sphere shaders (one instanced quad per cell, the sphere solved in the fragment shader) that
can be saved and shared. `viz3d.snapshot()` renders the WebGL view to PNG through headless Chrome,
so an executed notebook keeps a real 3-D figure.

**Note on ipyspeck:** it is on npm only up to 0.6.0 while the Python package is 0.8.4, so the
standard widget-state HTML export cannot resolve its JavaScript; a headless Speck image needs a live
frontend (Voilà) driven through the DevTools protocol.

### ADR-020 — MPI: what PhysiCell-X teaches, and why our layout changes the cost (2026-10-19)

**What PhysiCell-X does** (github.com/bsc-life/PhysiCell-X, v0.1 on PhysiCell 1.9.0, 3-D only):
an MPI Cartesian topology decomposed **along x only**, precomputed left/right Moore-neighbour voxel
lists, a mechanics halo exchange each mechanics step, and migration of cells that cross a cut.

**The cost it reveals:** `core/PhysiCell_cell_mpi.cpp` is **1,568 lines of pack/unpack**, plus
`documentation/cell_pack_serialization.md`, which classifies every field of a `Cell` as definition,
state, mixed or alias/cache and explains how the receiver rebuilds the pointer-backed parts
(`restore_cell_from_definition`, `apply_cycle_model_safety_net`). That is the price of migrating an
object graph: a cell is not a value, so it cannot be sent.

**Why it is cheaper here:** a cell is a row index across contiguous arrays. Migration is a copy of
rows and a remap of the index-valued columns — the same remap `Cells.remove()` already performs for
attachments. The serialization layer is not smaller; it does not exist. The same is true of the
fields: BioFVM-B's first change was replacing BioFVM's vector-of-vectors with one contiguous array
(§8.3), which is what the microenvironment already is here.

**Decision (refines ADR-013):** unchanged in practice — MPI-E (ensembles over ranks) now, MPI-D not
implemented. What changes is the estimate: should MPI-D be wanted, the expensive part in the
PhysiCell-X experience is already absent, and the remaining work is the distributed tridiagonal
solve (§8.3), the halo exchange, and load balance, which is where PhysiCell-X's 40 % efficiency at
10 nodes actually goes.

### ADR-021 — A demo window is set by the model's timescale, not by a wall-clock budget (2026-10-19)

**What went wrong:** `tools/make_notebooks.py` sized each notebook's run from a seconds-of-wall-time
budget alone. `template` got 270 simulated minutes of a **1080-minute** flow-cytometry-separated
cycle, so no cell could divide; `physimess` inherited its own config's **1 simulated minute**. The
symptom was downstream and looked like something else: in 17 of 37 projects every parameter the
sweep screen tried came back at 0.0 % swing, and 19 of 37 stopping conditions never fired. Verified
against the engine rather than assumed — the template has 0 births at 360 min, 3 at 1440, 76 at
5760. Nothing was broken except the window.

**Decision:** `pick_duration` computes the cycle timescale closed-form from the phase graph
(`Cycle.mean_duration`, no simulation) and covers at least 1.2 turns of the fastest cycle, bounded
by a 15 s wall-clock ceiling and the project's own `max_time`. Every notebook states where it
stands, and says plainly when the ceiling won and the population cannot divide in the window shown.

**Consequence:** a demo that cannot show the phenomenon it demonstrates is worse than a slow one.
The same rule applies to any default duration the library picks on a user's behalf.

### ADR-022 — Rule importance is screened, with the horizon attached (2026-10-19)

**The question:** a CBHG ruleset is a list of hypotheses and nothing says which of them carry the
result. Answering it is N+1 runs, which the file-driven workflow makes expensive enough that nobody
does it. Here a rule is data and a simulation is an object, so it is a loop over `model.rules`.

**What was measured** (`experiments/`, 100 rule-measurements over 5 models and 2 horizons, with
sham knock-outs as controls): a published ruleset is mostly inert — 39 of 100 measurements moved any
reported outcome — and **the inert set shrinks as the window grows**: 8 rules gained and 0 lost
between the short and the 1440-minute horizon. Rules written for cell types that only exist after a
transformation cannot act until something has transformed.

**Decision:** ship the screen, not the ranking. `rule_importance.lint` runs one instrumented
simulation and reports each rule's in-run displacement; below `SAFE_CUTOFF = 3e-4` no rule in the
corpus mattered. The cutoff is calibration from the measured operating curve (1e-3 let two live
rules through), recomputed by `--curve` as models are added. The ranking half was falsified and is
not shipped: displacement measures how far a rule moves *its own* behavior and cannot know that
cycle entry outweighs a transformation rate.

**Consequence:** every claim of this form carries a horizon. "Two-thirds of this ruleset is inert"
is a property of a model *and a window*, never of a model. Anything the library reports about rule
importance says over how long it looked.

## 5. Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│ L5  Workflows & tools   (physicell.ensemble, .compare, .viz, CLI)    │
│     Ensemble · sweep · calibrate · registry · metrics · plots        │
├──────────────────────────────────────────────────────────────────────┤
│ L4  Simulation control  (physicell.sim)                              │
│     Simulation · step/run/branch · checkpoints · Result · provenance │
├──────────────────────────────────────────────────────────────────────┤
│ L3  Model & hooks       (physicell.model, .hooks, .rules)            │
│     Model · CellType · Substrate · Rule · Hook · InitialConditions   │
├──────────────────────────────────────────────────────────────────────┤
│ L2  State access        (physicell.state)                            │
│     CellView · SignalTable · BehaviorTable · SubstrateField · Mesh   │
├──────────────────────────────────────────────────────────────────────┤
│ L1  Kernel binding      (physicell._core, nanobind)                  │
│     setup · step · gather/scatter · dictionaries · rules · plugins   │
├──────────────────────────────────────────────────────────────────────┤
│ L0  Upstream PhysiCell  (vendored, unmodified)                       │
│     BioFVM · core · modules · addons                                 │
└──────────────────────────────────────────────────────────────────────┘
```

Rules:
- A layer depends only on layers below it.
- L0 is never edited (ADR-002). L1 is the only code that includes PhysiCell headers.
- L3–L5 are pure Python. A hot loop moves to C++ behind an unchanged Python interface only when a
  profile says so (ADR-001).
- The worker-process boundary (ADR-003) sits between L4 and L3: the model and the hooks are shipped
  to the worker; the handle, the ensemble and the analysis stay in the user's process.

**As built (2026-10-19)** — `open-physicell/python/physicell/`:

| module | what it is |
|---|---|
| `kernels.cpp`, `_kernels.py` | the C++ kernels (diffusion, gradients, secretion, cell list, forces, fibres) and the ctypes binding that compiles them on first import |
| `model.py`, `io_xml.py`, `codegen.py` | the `Model` object, the PhysiCell file readers (multi-scenario), and model-to-Python generation |
| `state.py`, `fibres.py` | cells and fibres as struct-of-arrays, with attachments and index remapping |
| `phenotype.py`, `interactions.py`, `rules.py`, `signals.py` | volume/cycle/death, phagocytosis-attack-fusion-transformation-damage, the CBHG engine, the signal/behavior dictionary |
| `simulation.py`, `sweep.py` | the loop, hooks, branch, checkpoints, stop conditions, in-process sweeps |
| `standard.py`, `ports/` | PhysiCell's standard functions, and eleven sample projects' `custom.cpp` ported |
| `initial.py`, `check.py`, `viz.py`, `viz3d.py` | placement, dimensional analysis, 2-D plots, three 3-D backends |

## 6. Core data model

### 6.1 `Model`

Immutable once built (`Model.evolve(**changes)` returns a new one), hashable, serializable.

```
Model
├── domain:      Domain(x, y, z bounds, dx/dy/dz, use_2D, virtual_wall)
├── time:        TimeSettings(dt_diffusion, dt_mechanics, dt_phenotype, max_time, units)
├── substrates:  [Substrate(name, D, decay, initial, dirichlet(per face), units)]
├── cell_types:  [CellType(name, cycle, death, volume, mechanics, motility,
│                          secretion, interactions, transformations, custom_data)]
├── rules:       Ruleset([Rule(cell_type, signal, direction, behavior,
│                             base, half_max, hill_power, apply_to_dead)])
├── initial:     InitialConditions(placements | csv | callable)
├── parameters:  UserParameters(name -> typed value)   # PhysiCell's <user_parameters>
├── hooks:       {slot: Hook}                          # §6.5
├── plugins:     [Plugin]                              # §6.6
└── save:        SaveOptions(full/SVG/intracellular intervals, formats, folder)
```

`CellType` is a flat, typed mirror of PhysiCell's `<cell_definition>` XML, not of the 20-class
`Phenotype` object graph. Values are plain floats with documented units; defaults come from
`cell_defaults` and are recorded, never implied.

### 6.2 Signals and behaviors

The dictionaries are built at runtime from the substrates and cell types in the model
(`setup_signal_behavior_dictionaries`), so they are a property of the model, not a constant:

```python
m.signals            # -> list[str], e.g. "oxygen", "oxygen gradient", "pressure",
                     #    "contact with malignant epithelial cell", "damage", "volume", …
m.behaviors          # -> list[str], e.g. "cycle entry", "apoptosis", "migration speed",
                     #    "oxygen secretion", "transform to M1 macrophage", …
m.signal("oxygen")   # -> Signal(index, name, units, synonyms)
```

`SignalTable` and `BehaviorTable` are the array views of these (§6.4). Named column indexing is the
primary interface; integer indices are available and stable within a run only.

### 6.3 Rules

A `Rule` is PhysiCell's CBHG hypothesis in one object, and the CSV row is its serialization:

```python
Rule("malignant epithelial cell", "oxygen", "increases", "cycle entry",
     base=0.003333, half_max=21.5, hill_power=4, apply_to_dead=False)
```

`Ruleset` supports `+`, filtering by cell type/signal/behavior, `to_csv()` / `from_csv()` (v0, v1 and
v3 dialects, `core/PhysiCell_rules.h`), `to_english()` (upstream's annotated English export) and a
`diff()` used by the comparison layer. Rules compile into the engine at setup; they are not
re-evaluated from Python.

### 6.4 State access (L2)

```python
sim.cells                       # CellView over the live population
sim.cells.n                     # int
sim.cells.positions             # (N, 3) float64 copy
sim.cells.types                 # (N,) int32, with .type_names
sim.cells.signals               # SignalTable: sig[:, "oxygen"] or sig["oxygen", "pressure"]
sim.cells.behaviors             # BehaviorTable, same indexing
sim.cells.state                 # cycle phase, elapsed time in phase, dead flags, volumes, …
sim.cells.custom                # custom_data columns by name
sim.cells.to_dataframe(cols)    # pandas / polars

sim.substrates["oxygen"]        # SubstrateField: (nz, ny, nx) copy, .mesh, .write()
sim.substrates.gradient("oxygen")
sim.mesh                        # voxel centers, spacing, bounds
```

All of these are copies (ADR-007). `CellView` carries the simulation time it was taken at and
invalidates itself after the next `step()`, so stale arrays raise instead of lying.

### 6.5 Hooks

```python
@physicell.hook("phenotype", cell_type="macrophage", mode="vectorized")
def polarize(ctx):
    o2 = ctx.signals["oxygen"]
    debris = ctx.signals["necrotic debris"]
    ctx.behaviors["transform to M1 macrophage"] = np.where(o2 > 10, 0.05 * debris, 0.0)
```

| Slot | `Cell_Functions` member | Modes | Called |
|---|---|---|---|
| `phenotype` | `update_phenotype` | vectorized, per-cell | each phenotype substep |
| `custom_rule` | `custom_cell_rule` | vectorized, per-cell | each mechanics substep |
| `velocity` | `update_velocity` | per-cell (vectorized under study) | each mechanics substep |
| `contact` | `contact_function` | per-cell | per attached pair |
| `division` | `cell_division_function` | per-cell | on division |
| `volume` | `volume_update_function` | per-cell | each phenotype substep |
| `migration_bias` | `update_migration_bias` | vectorized, per-cell | each mechanics substep |
| `orientation`, `membrane_distance`, `membrane_interactions` | the remaining pointers | per-cell | as upstream calls them |

`ctx` exposes the arrays for the cells of that type, the current time, the dt, the model parameters
and the RNG stream for this hook (seeded from the run seed and the hook name, so hooks are
reproducible independently of thread scheduling).

**Setup functions are a separate category.** `create_cell_types` and `setup_tissue` are not
`Cell_Functions` slots; they run once, before the loop. In open-physicell they are the `Model`'s cell
types and `InitialCondition` respectively, so a project whose only custom C++ is `setup_tissue` is
tier B, not tier C (ADR-008).

Hooks also come in a **step hook** form, which is not a `Cell_Functions` slot at all but a callback
between kernel steps in the Python loop. It is the simplest thing that works for treatment protocols,
measurements and stopping conditions, and it has zero kernel involvement:

```python
sim.on_time(every=60, fn=lambda s: s.substrates["drug"].set_uniform(0.0 if s.t > 1440 else 1.0))
sim.run(until=4320, stop_when=lambda s: s.cells.n > 50_000)
```

### 6.6 Plugins (tier C)

```python
plug = physicell.plugins.build("PhysiCell/sample_projects/cancer_immune",
                               addons=["PhysiBoSS"])     # compiles once, caches by hash
model = physicell.Model.from_files("…/cancer_immune", plugin=plug)
```

`build()` generates a CMake project that compiles the project's `custom_modules/` plus any requested
`addons/` against the installed headers, with the kernel's recorded flags, and produces a shared
library exporting a registration function. The plugin declares which functions it provides and which
`Model` fields it expects; `Model` binds them by name.

### 6.7 Units

PhysiCell's own: minutes, microns, and each substrate's declared concentration units. The Python API
takes plain floats in those units and documents them in every docstring. No `pint`, for the same
reason as the parent project.

## 7. Component protocols

Boundaries are `typing.Protocol` classes; the kernel sits behind them.

| Protocol | Role | Built-in implementations |
|---|---|---|
| `Hook` | model logic in Python (§6.5) | `VectorizedHook`, `PerCellHook`, `StepHook` |
| `InitialCondition` | place cells at t=0 | `FromCSV`, `Disc`, `Box`, `Lattice`, `FromResult` (resume/branch), `Callable` |
| `Metric` | `compute(Result) -> xarray` | population, spatial (pair correlation, Ripley K, clustering), substrate profiles, phase fractions, extinction time |
| `Sampler` | parameter points for an ensemble | grid, Latin hypercube, Sobol, from a dataframe, adaptive (returns points given results so far) |
| `Objective` | scalar or vector target for calibration | from metrics + reference data, with replicate-aware noise |
| `Storage` | where results go | memory, Zarr, HDF5, directory of MultiCellDS |
| `Scheduler` | where runs run | local pool, SLURM, nanoHUB/submit, cloud |
| `IntracellularModel` | per-cell internal model | PhysiBoSS (MaBoSS), libRoadrunner (SBML), dFBA, `PythonODE` (vectorized, via the phenotype hook) |

Only `Metric`, `Sampler`, `Objective` and `Storage` are expected to be implemented by users in v1;
the rest are listed so the seams are visible.

## 8. Execution and parallelism

### 8.1 Levels

```
Ensemble (Scheduler: local pool | MPI (mpi4py) | SLURM | nanoHUB submit | cloud)
 └─ Run = one worker process = one PhysiCell global state (ADR-003)
     │   (MPI-E: a run is one rank; ranks never talk during a run)
     └─ OpenMP threads inside the kernel (omp_num_threads)
         └─ [reserved] MPI-D: one run across a communicator (§8.3, ADR-013)
             └─ (future) vectorized kernels / GPU diffusion (§9.3)
```

`Ensemble(model, sampler, replicates=n, threads_per_run=k, workers=m)` with the invariant
`workers × threads_per_run ≤ available cores`, checked and reported rather than assumed. PhysiCell's
OpenMP scaling is sublinear past a handful of threads for typical cell counts, so for sweeps the
library defaults to **many workers × few threads** and says so.

### 8.2 Scheduling and fault tolerance

- Each run is independent; the scheduler is a simple work queue with per-run timeouts.
- Each finished run writes its `Result` and a row in the ensemble's point log, so an interrupted
  ensemble resumes and skips finished points — the same `resume` discipline as open-nemo §8.4.
- A crashed worker is a recorded failure for that point, not an ensemble failure; the point log
  stores the exit status and the last checkpoint.
- Checkpoints: a `Simulation` can `checkpoint()` to a file (cells + fields + RNG + time) and be
  restored into a new worker, which is also what `sim.branch()` uses for counterfactual runs from a
  common state.

### 8.3 Can PhysiCell's algorithms be divided by MPI, not just threads?

Yes — algorithm by algorithm, with two genuine blockers and one practical one. The analysis below is
from the 1.14.2 source; the time loop is `Cell_Container::update_all_cells` plus
`Microenvironment::simulate_diffusion_decay`.

Assume the natural scheme: **one geometric decomposition shared by the diffusion mesh and the
mechanics grid**, each rank owning a contiguous region, agents owned by the rank that owns the voxel
they sit in.

| Algorithm | Where it is | Communication it needs | Verdict |
|---|---|---|---|
| Diffusion/decay, sweeps **perpendicular** to the cut | `BioFVM_solvers.cpp:197+`, `#pragma omp parallel for` over lines | none — each tridiagonal line lies inside one rank | trivially distributable |
| Diffusion/decay, sweep **along** the cut | same | a distributed tridiagonal solve: pipelined/blocked Thomas, a Schur-complement reduced system, or pencil transposes | hard but solved; see BioFVM-B below |
| Dirichlet conditions, bulk sources/sinks | `apply_dirichlet_conditions`, `simulate_cell_sources_and_sinks` | none | local |
| Gradients | `compute_all_gradient_vectors`, central differences | 1-voxel halo exchange per substrate, once per mechanics step | routine |
| Secretion/uptake | `Secretion::advance` → `nearest_density_vector` | **none**: a cell only ever writes the voxel it sits in, which its own rank owns | local by construction |
| Phenotype, cycle, volume, death flagging, intracellular | `advance_bundled_phenotype_functions`, intracellular loop | none | embarrassingly parallel |
| Rules (CBHG) | `apply_ruleset` via the phenotype function | none — signals are per-cell and per-voxel | embarrassingly parallel |
| Mechanics forces | `standard_update_cell_velocity` → `add_potentials` over the **Moore 27-voxel** neighborhood of the mechanics grid | ghost-agent exchange one mechanics voxel deep (30 µm by default) each mechanics step: position, radius, type and the few phenotype fields the force law reads | standard MD halo; routine |
| Position update, motility | `Cell::update_position` | none, then **ownership migration** of cells that crossed a cut | routine |
| Division | `Cell::divide`, serial loop over `cells_ready_to_divide` | daughter may land across a cut (migrate), and IDs come from a process-static counter (`max_basic_agent_ID`, `BioFVM_basic_agent.cpp:57`) | needs rank-striped IDs; otherwise routine |
| **Attachments, springs, `state.neighbors`** | `Cell_State` holds raw `Cell*` (`core/PhysiCell_cell.h`) | a link that crosses a cut has no meaning: pointers are not portable | **blocker**: needs global IDs + ghost proxies, or a rule that attached clusters never split across ranks |
| **Phagocytosis, attack, fusion** | `standard_cell_cell_interactions`, `ingest_cell`/`attack_cell`/`fuse_cell` under `#pragma omp critical` | these **write into another cell**; across a cut that becomes a message queue applied by the owner, with a defined ordering | **blocker**: changes semantics, not just plumbing |
| Load balance | — | cells cluster (a spheroid grows in the middle of the domain) while voxels do not; a static slab decomposition puts most agents on few ranks | **practical limiter**: needs orthogonal recursive bisection / space-filling-curve rebalancing |
| Output (MultiCellDS) | `save_PhysiCell_to_MultiCellDS_v2` | one serial writer today | per-rank files + merge, or parallel HDF5 |
| RNG | per-thread seeds (`physicell_random_seeds`) | per-rank-and-thread streams | results become rank-count dependent (ADR-010 already says so) |

**The two blockers are the agent-coupling ones**, and they are exactly the features PhysiCell has
added most of since 1.8 (springs, attack, phagocytosis, fusion, contact functions). A 2015-era
PhysiCell would decompose almost mechanically; the 2026 one has a rich cross-cell write surface.

**What the prior art shows:**

| Project | What it did | Reported result |
|---|---|---|
| **BioFVM-X** (Saxena et al., CMSB 2021) | MPI+OpenMP BioFVM, 1-D x decomposition, `MPI_THREAD_FUNNELED` | runs multi-node; later work reports it "does not scale efficiently when increasing the number of compute nodes" |
| **BioFVM-B** (preprint, 2026) | contiguous 4-D array instead of vector-of-vectors, same 1-D partition, **blocked pipelined Thomas** with non-blocking MPI overlapping communication (NB ≈ 2 × nodes) | up to 34.8× over BioFVM and ~197× over BioFVM-X; ~35× strong scaling on 24 nodes; >120 M TDMA equations/s on 96 nodes; a single block degrades to ~20 % efficiency (i.e. the pipelining *is* the result) |
| **PhysiCell-X** (PerMedCoE) | MPI+OpenMP PhysiCell, decomposition along x only | predator–prey on 10 nodes / 480 cores: 4.06× speed-up, 40.3 % efficiency |

So: the microenvironment scales once someone rewrites the solver properly (BioFVM-B did), and the
**agent side is where the efficiency goes** — 40 % at 10 nodes, with a clustering-prone model, is the
published state of the art. BioFVM-B explicitly lists distributed agent computation as future work.

**When is MPI-D even needed?** Only when one node is not enough, which is a memory question before it
is a speed one. At `dx = 20 µm`: 1 mm³ = 50³ = 1.25 × 10⁵ voxels; 1 cm³ = 500³ = 1.25 × 10⁸ voxels.
Per substrate that is 8 B/voxel for the density (1 GB per substrate per cm³) and 24 B/voxel for the
gradient (3 GB). A `Cell` is 2,072 B of flat object on arm64 (measured) plus its heap vectors, so
call it 3–5 KB; 10⁶ cells is a few GB, 10⁸ cells is not happening on one node. Tissue-scale 3-D
(organ-level, BioFVM-B's liver case) needs MPI-D; the tumor-spheroid and tissue-slice models most
PhysiCell papers run do not, and for those **MPI-E with more replicates is strictly better value.**

### 8.4 What this means for open-physicell

- ADR-003 (one simulation per process) is **MPI-friendly, not MPI-hostile**: PhysiCell's globals are
  per-process, and under MPI-D a rank is a process holding one partition's globals. The same design
  that makes ensembles safe is the one MPI-D would need.
- open-physicell does **not** write a distributed PhysiCell (ADR-013). What it does is keep the
  `Partition` seam, the collective contract on state access, and the rank-local definition of hooks,
  so that a PhysiCell-X- or BioFVM-B-class kernel can be dropped in under the same Python API.
- A cheaper intermediate exists if the sizing question ever demands one: **MPI-D for the
  microenvironment only**, with agents kept on one rank. It fits diffusion-bound models (many
  substrates, large 3-D domain, modest cell counts) and avoids both agent-coupling blockers
  entirely. It is a Phase 6 investigation, not spike work.

### 8.5 Cluster and nanoHUB

`Scheduler` implementations keep the API identical between a laptop pool and a cluster. The nanoHUB
backend matters for this group specifically: a model is a serializable object with a content hash, so
submitting 500 points to a queue needs no project directory per point and no recompilation.

## 9. Performance

### 9.1 Where the time goes (to be measured, not assumed)

The first deliverable of the spike (§18.0) is a profile of the stock executable on three sample
projects (`template`, `rules_sample`, `cancer_immune`) at several population sizes, split into:
diffusion/decay (BioFVM LOD Thomas solvers), mechanics (`update_all_cells` velocity loop), phenotype
+ rules, secretion/uptake, and output writing. Every later optimization claim is relative to that
table.

### 9.2 What open-physicell adds, and its budget

| Addition | Expected cost | Budget |
|---|---|---|
| Python-owned loop (`step` per diffusion dt) | one nanobind call per step | < 1 % of step time; if not, `advance_to()` batches steps |
| Vectorized hook | gather + Python + scatter, once per phenotype dt | < 5 % at 10⁵ cells |
| Per-cell hook | serialized Python call per cell | measured and published; expected 10–100× the vectorized cost |
| State access from the user's process | gather + shared-memory copy | measured per accessor; column selection is the documented remedy |
| Portable build vs `-march=native` | compiler-dependent | measured per platform (ADR-011) |
| Array output instead of MultiCellDS | **negative** (saves time) | measured on long runs |

### 9.3 Possible kernel work, later and only if measured

- SoA mirrors for position/velocity/volume to make gathers views (needs ADR-002 patch).
- Batched vectorized evaluation of the rules engine.
- GPU diffusion for large 3-D domains (BioFVM's LOD solvers are tridiagonal sweeps, a natural fit);
  only worth it if §9.1 shows diffusion dominating at the sizes the group runs.
- A single-process, multi-simulation kernel (removing the globals upstream). Large, upstream-owned,
  and not needed if ADR-003 holds.

## 10. Models, data and comparison

### 10.1 Model registry

A model entry is a directory or package with: the serialized `Model` (TOML/JSON + the PhysiCell files
it round-trips to), a citation and DOI, the PhysiCell version it was published against, the reference
results it must reproduce, and the ids of the tests that check it. Content-addressed, so "the same
model" is a decidable question.

Seed corpus: upstream's 18 `sample_projects` and the `sample_projects_intracellular` ones — they are already a curated,
diverse, citable set, and they are the acceptance tests for the ingest tiers of ADR-008 at the same
time.

**As built (2026-10-19):** the corpus is real and measured. Beyond upstream's 18 sample projects,
`published/` holds the models of the rules-grammar paper
(`PhysiCell-Models/grammar_samples`) and four standalone repositories — **21 model scenarios**.
`tools/survey_published.py` reports them:

- **every rule runs in all 15 scenarios that have rules — 196 rules, none skipped**, including
  `pdac_therapy` (9 cell types, 36 rules) and `tumor_immune_extended` (7 types, 23 rules);
- 9 scenarios need neither a skipped rule nor custom C++; 6 more run their rules and carry C++ to
  port; 6 are pre-1.6 projects whose cell types are defined in C++ with no `<cell_definitions>` at
  all, so there is nothing for the reader to load.

Three of those projects ship several scenarios each (`PhysiCell_settings_go.xml`, `_grow.xml`, …),
which is why `pc.scenarios(project)` exists and `pc.load` refuses to guess between them.

### 10.2 Parameter provenance

Cell parameters in PhysiCell models are typically uncited numbers in XML. open-physicell does not fix
that, but it records it: a `CellType` field may carry `source=` and `validity=` metadata, and
`Model.parameter_report()` lists which values have a source and which do not. The honest output of
that report on existing projects is itself a result worth publishing.

### 10.3 Comparison

```python
cmp = physicell.compare.run([model_a, model_b], metrics=["population", "spatial.clustering"],
                            replicates=8, until=7200)
cmp.table()        # metric × model, mean ± CI over replicates
cmp.plot()         # aligned time series with replicate bands
cmp.diff_models()  # structural diff: substrates, cell types, rules, parameters, hooks
```

`diff_models()` is the piece no current tool has: a structural diff of two PhysiCell models, including
a rules diff in CBHG English, so "what is actually different between these two papers' models" has an
answer that is not a careful reading of two `custom.cpp` files.

## 11. Python API

### 11.1 Run an existing project, no compiler

```python
import physicell

report = physicell.inspect("PhysiCell/sample_projects/rules_sample")
print(report.tier, report.custom_functions)      # "A", []

model = physicell.Model.from_files("PhysiCell/sample_projects/rules_sample")
sim   = physicell.Simulation(model, seed=0, threads=4)
result = sim.run(until=7200, save_every=60)

result.population.plot()
result.save("rules_sample.zarr")
```

### 11.2 Build a model in Python, with a rule and a vectorized hook

```python
import numpy as np, physicell as pc

m = pc.Model(
    domain=pc.Domain(x=(-500, 500), y=(-500, 500), z=(-10, 10), dx=20, use_2D=True),
    time=pc.TimeSettings(dt_diffusion=0.01, dt_mechanics=0.1, dt_phenotype=6.0),
)
m.add(pc.Substrate("oxygen", D=1e5, decay=0.1, initial=38.0,
                   dirichlet={"xmin": 38.0, "xmax": 10.0}))
m.add(pc.CellType("tumor", cycle="live", motility=pc.Motility(speed=0.5, persistence=10)))
m.add(pc.CellType("macrophage", motility=pc.Motility(speed=1.0, persistence=5)))

m.rules += pc.Rule("tumor", "oxygen", "increases", "cycle entry",
                   base=0.0033, half_max=21.5, hill_power=4)
m.rules += pc.Rule("tumor", "pressure", "decreases", "cycle entry",
                   base=0.0033, half_max=1.0, hill_power=4)

@m.hook("phenotype", cell_type="macrophage")
def chemotax(ctx):                      # vectorized: arrays over all macrophages
    debris = ctx.signals["necrotic debris"]
    ctx.behaviors["migration speed"] = np.clip(0.2 + 40 * debris, 0, 2.0)
    ctx.behaviors["chemotactic response to necrotic debris"] = 1.0

m.initial = pc.initial.Disc("tumor", radius=100, n=500) + pc.initial.Uniform("macrophage", n=50)

sim = pc.Simulation(m, seed=0, threads=4)
with sim.live_plot(["oxygen"], color_by="cell_type"):   # notebook: updates as it runs
    sim.run(until=4320, save_every=30)

m.to_files("my_project/")        # runs under the stock PhysiCell executable too (ADR-006)
```

### 11.3 Step, inspect, intervene, branch

```python
sim = pc.Simulation(m, seed=0)
sim.run(until=1440)

cells = sim.cells
hypoxic = cells.signals["oxygen"] < 10
print(f"{hypoxic.mean():.1%} hypoxic at t={sim.t}")

treated = sim.branch()                            # checkpoint + new worker
treated.substrates["drug"].set_uniform(1.0)
treated.run(until=4320)
control = sim.run(until=4320)

pc.compare.results({"treated": treated.result, "control": control}).plot()
```

### 11.4 Sweep and calibrate

```python
ens = pc.Ensemble(
    m,
    sampler=pc.sample.LatinHypercube({
        "rules[tumor,oxygen,cycle entry].half_max": (5.0, 30.0),
        "cell_types[macrophage].motility.speed":    (0.2, 2.0),
    }, n=200),
    replicates=4, until=4320, workers=16, threads_per_run=2,
    storage=pc.storage.Zarr("sweep/"), scheduler=pc.sched.Local(),
)
runs = ens.run()                                  # resumable; one row per finished point

fit = pc.calibrate(m, objective=pc.Objective(metric="population",
                                             reference=observed_counts, noise="replicates"),
                   sampler=pc.sample.Adaptive(budget=400))
print(fit.best, fit.posterior_summary())
```

### 11.5 API rules

- `Model` and everything in it is immutable after construction; `evolve()` returns a new model with a
  new hash.
- Every long-running call returns a result object with `.data`, `.provenance`, `.save()`, `.plot()`.
- No hidden global state in Python: seed, thread count, scheduler and storage are explicit arguments.
- Anything returning cell or field data returns a copy, and says so in its docstring (ADR-007).
- Any call that writes artifacts can estimate their size first (`sim.estimate_output()`, §12.3).
- One simulation per process, enforced with a clear error (ADR-003).

### 11.6 Partial flows: what is useful without running a simulation

In NEMO, the Hamiltonian and the band structure are worth having on their own, long before anyone
runs transport. PhysiCell has the same shape, and the pieces are already in the C++ — they are just
locked behind a compile-and-run: `display_cell_definitions`, `display_signal_dictionary`,
`export_rules_csv_v3`, `save_annotated_English_rules_HTML` all exist upstream and are reachable only
by building a binary that prints them and exits.

| # | Product | What it needs | Kernel? | What it answers |
|---|---|---|---|---|
| 1 | **Signal/behavior grammar** of a model. `setup_signal_behavior_dictionaries` needs only the substrates and the cell definitions — no cells, no time loop. | `Model` | no (pure Python from the model) | "What can I write a rule about?" Validation, autocomplete, and a machine-readable grammar for assisted model authoring. |
| 2 | **Rule response curves.** `Hypothesis_Rule::evaluate(std::vector<double> signal_values, bool dead)` (`core/PhysiCell_rules.h`) is a **pure function of signal values** — no `Cell`, no microenvironment. | `Ruleset` | no | "What does this rule actually do?" Behavior vs signal, half-max and Hill sweeps, min/max clamping, and two papers' rules on one set of axes. **This is the band-structure analogue.** |
| 3 | **Cycle-model analysis.** A `Cycle_Model` is a graph of phases with transition rates and per-link `fixed_duration` flags. | `CellType` | no | Mean cycle time, the division-time distribution (a sum of exponentials for stochastic links), steady-state phase occupancy — closed form. "What doubling time do these numbers imply?" |
| 4 | **Single-cell 0-D mode.** `standard_volume_update_function` and `Geometry::update` are pure ODEs on `phenotype`; `Cycle_Model::advance_model` sets `phenotype.flagged_for_division` rather than calling into the container. A one-voxel microenvironment is supported (the Thomas setup special-cases size-1 dimensions). | kernel, trivial domain | yes (cheap) | A spaceless trajectory of volume, radius, phase and death for one cell type. Parameter intuition and cycle/volume calibration before any spatial run. |
| 5 | **Diffusion-only (BioFVM) mode.** The microenvironment solves with zero agents; BioFVM was a standalone published solver before PhysiCell existed. | kernel | yes | Substrate field design, boundary/Dirichlet studies, dx and dt choices, and plain reaction–diffusion problems that have nothing to do with cells. |
| 6 | **Dimensional-analysis checks.** Diffusion length √(D/λ) against `dx`; the cell-scale uptake time constant 1/(S+U) scaled by V_cell/V_voxel (`set_internal_uptake_constants`) against `dt_diffusion`; displacement per mechanics step against the mechanics voxel size. | `Model` | no | `model.check()` — warnings before a node-hour is spent. Nothing in the ecosystem does this today. |
| 7 | **Force-law evaluation.** `Cell::add_potentials` is pairwise and pure given two cells' positions, radii and adhesion/repulsion parameters. | `CellType` pair | yes (thin) | Force vs distance, equilibrium spacing, pressure at a given packing — the parameters users most often get wrong. |
| 8 | **Initial-condition generation.** `fill_circle` / `fill_annulus` / `fill_rectangle` (hex packing) and the CSV loaders in `modules/PhysiCell_geometry.h`. | `Model` | no (pure geometry, NumPy) | Build and preview initial conditions, and write the CSV that both open-physicell and the stock executable read. |
| 9 | **Model files: read, validate, diff, write.** | files | no | PhysiCOOL's niche, plus the structural and rules diff of §10.3. |
| 10 | **Output reading and rendering.** MultiCellDS → arrays; SVG from a snapshot. | files | no | `pcdl`'s niche; included so an analysis session needs one import, not three. |

**Why this matters for the project, not just for users:**

- **Six of the ten need no kernel at all.** Items 1, 2, 3, 6, 8 and 9 are pure Python over the
  `Model` object. They can ship before the kernel binding works at all, and they are what make the
  library useful on day one to someone who has never compiled PhysiCell.
- **They de-risk the whole plan.** If the spike (§18.0) comes back badly on hook overhead, the
  model / analysis / comparison half of the library is still worth having and still stands alone.
- **They are the natural comparison surface.** Comparing two papers' *models* — their rules, implied
  cycle times, uptake timescales — is often more informative than comparing two stochastic outputs,
  and it costs nothing to compute (§10.3).

**Where the analogy with NEMO stops.** A band structure is a publishable physics result. These
products are mostly model inspection, sanity checking and calibration support: in an agent-based
model the science is in the emergent dynamics, which does need the full run. So the partial flows
buy productivity, trust and comparability — they do not replace simulating, the way bands sometimes
replace transport.

```python
import physicell as pc

m = pc.Model.from_files("PhysiCell/sample_projects/rules_sample")

m.signals.search("debris")              # 1: what can I write a rule about?
m.behaviors.search("transform")

rule = m.rules["M0 macrophage", "necrotic debris", "transform to M1 macrophage"]
rule.curve(span=(0, 0.02)).plot()       # 2: pure function of the signal value

m.cell_types["malignant epithelial cell"].cycle.mean_duration()    # 3: closed form
m.cell_types["malignant epithelial cell"].cycle.graph()            #    phases + rates

traj = pc.single_cell(m, "malignant epithelial cell", oxygen=38.0, until=4320)  # 4
traj.volume.plot()

field = pc.diffusion_only(m, until=60)  # 5: BioFVM with no cells
field["oxygen"].slice(z=0).plot()

m.check()   # 6: real output for this model --
            #   oxygen:                  L = sqrt(D/lambda) = 1000 um = 50 dx   ok
            #   pro-inflammatory factor: L = 100 um = 5 dx                      marginal
            #   necrotic debris:         decay 0, D = 1 um^2/min -> spreads
            #                            ~54 um per day: effectively a local signal

pc.mechanics.force_curve(m.cell_types["tumor"], m.cell_types["tumor"]).plot()   # 7
pc.initial.Disc("tumor", radius=100).preview()                                  # 8
```

## 12. Artifacts: what a run should produce

### 12.1 What PhysiCell produces today, and what it costs

Per save, upstream writes `outputNNNNNNNN.xml` (MultiCellDS), `_cells.mat` (~90 columns per cell,
more with substrates and custom data), a microenvironment `.mat`, an SVG cross-section, and
optionally `_cell_neighbor_graph.txt`, `_attached_cells_graph.txt`,
`_spring_attached_cells_graph.txt` and `_boolean_intracellular.csv`. The movie is a Makefile target:
`make jpeg` (ImageMagick) → `make movie` (ffmpeg over `snapshot%08d.jpg`).

Three problems with that as a research artifact set:

- **Volume.** `PhysiCell_Settings` defaults to `max_time = 60*24*45` (45 days) with a 60-minute save
  interval: 1,080 saves. At 10⁵ cells and ~90 columns that is ~72 MB of cell data per save, ~78 GB
  for the run, plus one SVG per frame (one `<circle>` per cell, so tens of MB of XML each).
- **The cheap thing is missing.** There is no population-over-time series unless legacy saves are
  on, and then it is a five-column TSV (time, #cells, #births, #deaths, wall time) written by
  `log_output`. So the most-plotted quantity in the literature is normally reconstructed by loading
  every snapshot back — which is a large part of why `pcdl` exists.
- **Renderings are not reproducible from the data.** Coloring is a C++ function pointer
  (`SVG_plot(..., cell_coloring_function, substrate_coloring_function)`, bound in `main.cpp`), so
  changing how a frame is colored means editing C++, recompiling and **re-running** — or
  post-processing the `.mat` in another tool entirely.

### 12.2 Is a video what people want?

Mostly no. A video is the **communication** artifact — talks, abstracts, outreach, and genuinely
useful for eyeballing whether a model is behaving sanely — and open-physicell should make it one
line. But it is the worst possible research artifact: it cannot be queried, re-colored, re-scaled,
zoomed, subsetted, diffed against another run, or have error bars put on it, and it is irreversibly
lossy. You can always make the video from the data; you can never make the data from the video.

What the field's own tooling says people actually use: `pcdl` converts output to dataframes and
AnnData, `PhysiCOOL` converts runs into calibration objectives, and the figures in PhysiCell papers
are overwhelmingly time series, dose–response curves and quantified snapshots, with one or two
rendered panels for intuition. **The quantitative artifacts are the output; the picture is a
figure panel; the video is a slide.**

So the governing rule is: **data → derived → rendered, and every rendering must be reproducible from
the stored data without re-running the simulation.** That rule is precisely what upstream cannot
satisfy today, and satisfying it is most of the value of this section.

### 12.3 The artifact tiers

| Tier | Artifact | Cadence | Size | Who consumes it |
|---|---|---|---|---|
| 0 | **Run record** — provenance (ADR-009), the resolved `Model`, its content hash, status, timings, warnings, captured kernel stdout | once | KB | reviewers, the registry, ensembles, you in six months |
| 1 | **Scalars over time** — population by type and by cycle phase, births, deaths, mean/median/quantiles of selected signals, total substrate mass, domain aggregates, wall-time per step | every phenotype step (cheap) | KB–MB | **most paper figures**, stopping conditions, calibration objectives |
| 2 | **Snapshots** — cell table (Parquet/Arrow, selected columns) + fields (Zarr, selected substrates) | rare, explicit | MB–GB | spatial analysis, rendering, restart, anything not anticipated in tier 1 |
| 3 | **Derived metrics** — spatial statistics, survival/extinction, morphology, per-type trajectories; tidy tables, recomputable from tiers 1–2 | on demand | KB | comparison (§10.3), ensembles |
| 4 | **Renderings** — static figures (SVG/PDF), a self-contained interactive scrubber (HTML), video (mp4/gif) | on demand, from tier 2 | MB | humans |
| 5 | **Interop exports** — MultiCellDS (Studio, `pcdl`), AnnData, VTK/XDMF (ParaView), initial-conditions CSV | on demand | varies | the rest of the ecosystem |
| 6 | **Ensemble artifacts** — the sweep table (one row per run: parameters + metrics + provenance hash), the comparison report | per ensemble | MB | the actual deliverable of a sweep or calibration study |

Tiers 0 and 1 are **always on**: together they are kilobytes, and they are what people plot. Tier 2
is opt-in with an explicit policy, because it is the only tier that can fill a disk:

```python
sim.run(
    until=64800,
    scalars=pc.save.Scalars(every=6, signals=["oxygen", "pressure"], quantiles=[0.1, 0.5, 0.9]),
    snapshots=pc.save.Snapshots(
        every=360,                                   # 6 h, not 1 h
        cells=["position", "cell_type", "cycle_phase", "dead", "total_volume", "pressure"],
        fields=["oxygen"],                           # not all five substrates
        subsample=None,                              # or 0.1 for a random 10% of cells
    ),
)
print(sim.estimate_output())   # "run record 4 kB · scalars 2.1 MB · snapshots 180 x 31 MB = 5.6 GB"
```

`estimate_output()` before launching mirrors open-nemo's `solver.estimate(H)`: the artifact budget is
a decision you make up front, not a surprise you find on the filesystem.

### 12.4 Rules

1. **Renderings are never the source of truth.** Every figure, scrubber and video is generated from
   tier 2 by Python code that is recorded with it, so re-coloring is a function call, not a rerun.
2. **Everything written is readable back by this library**, and carries the run hash. A directory of
   artifacts with no provenance is a bug.
3. **The default visual artifact is the interactive scrubber, not the movie.** One self-contained
   HTML file: scrub time, switch the color-by field, toggle substrate layers, click a cell to see its
   row. It is shareable, it is a published artifact page on nanoHUB or claude.ai, and it answers the
   questions a video cannot. `result.animate("run.mp4")` stays one line for when a slide needs one.
4. **Figures carry their provenance**: run hash, model hash, and the snippet that drew them, in the
   file metadata, so a figure in a paper can be traced to a run and regenerated.
5. **The ensemble's artifact is a table, not a folder of runs.** One row per run — parameters,
   metrics, status, hashes — designed to be appended to a results database (nanoHUB ResultsDB /
   sim2l style) so that a sweep is minable later and by other people.
6. **MultiCellDS is an export, not the working format** (ADR-009). Writing it every hour of
   simulated time is a large share of wall time on long runs and produces files nothing in this
   stack reads back.

### 12.5 What Studio and UQ-PhysiCell already produce, and the division of labor

This section's conclusions are not speculative: two tools in the ecosystem have converged on the
same answer, which is good evidence it is the right one.

- **PhysiCell Studio 2.48** already offers a frame scrubber with a Play button, a **population plot**
  over time by cell type / cycle model / phase / dead, frame saving, `snap.csv` export, and
  `movie.mp4` through matplotlib + ffmpeg. Its video is one button among many — the quantitative
  panel sits right next to it.
- **UQ-PhysiCell 1.2.10** stores, per run, **either precomputed QoIs or raw MCDS** in a SQLite
  database, keyed by (SampleID, ReplicateID) with the sample's parameters. That choice is exactly
  §12.3's tier 1 versus tier 2, made independently by people running real sweeps.

Where they are forced to be inefficient is the part open-physicell can fix. Studio's population plot
calls `pyMCDS` on **every `output*.xml` in the folder** and holds them all to count cells per type:
a quantity of a few hundred bytes per frame is recovered by re-reading tens of gigabytes, because
nothing wrote it during the run. That is tier 1 missing, not a flaw in Studio.

**Division of labor to aim for:**

| Layer | Owner |
|---|---|
| Authoring a model in a GUI, browsing frames, one-off movies | **Studio** — it is good at this, it runs on nanoHUB, and it reads the XML open-physicell writes (ADR-006) |
| Sampling, calibration, UQ databases | **UQ-PhysiCell** where it fits — adopt its database and samplers rather than inventing a parallel set (open question 9) |
| The model object, the live simulation, hooks, partial flows, tier 0–1 artifacts written **during** the run, provenance and model diffs | **open-physicell** |

The test of this design is that Studio's population plot could become a read of a tier-1 table
instead of a thousand-file reconstruction, and a UQ-PhysiCell sweep could run against an in-process
simulation instead of a subprocess per sample.

### 12.6 Logging

Python `logging` with a per-run prefix; kernel stdout captured per worker and attached to the run
record rather than printed into the notebook; `Timers` per phase (diffusion, mechanics, phenotype,
hooks, I/O) emitted into tier 1 so performance regressions show up in the same place as the science.

## 13. Testing and validation

### 13.1 Golden-run equivalence (the project's core test)

For each sample project, at `threads=1` and fixed seed:

1. build and run the stock executable on the project's own XML,
2. run the same project through open-physicell with the same XML, seed and thread count,
3. assert cell-by-cell and voxel-by-voxel equality of the final state, and of every intermediate
   save, to bitwise tolerance.

This test is what makes "it is PhysiCell" a claim rather than a hope. It runs in CI on every commit
for a fast subset and nightly for all projects, and it is the gate on upgrading the pinned upstream
version.

### 13.2 The rest

| Level | What | Where |
|---|---|---|
| Unit | gather/scatter round trips, dictionary lookups, rules CSV round trips, XML round trips, model hashing | `tests/unit` |
| Contract | one-simulation-per-process enforcement, stale-view invalidation, hook arity and dtypes, plugin ABI rejection | `tests/contract` |
| Equivalence | §13.1, plus Python hook vs equivalent C++ custom function on the same model | `tests/equivalence` |
| Reproducibility | same seed + 1 thread ⇒ bitwise; multi-thread ⇒ distributional tests with fixed tolerance over n replicates | `tests/repro` |
| Validation | reproduce the reference results of each registered model (§10.1) | `tests/validation`, nightly |
| Performance | §9.1 profile and §9.2 budgets, tracked over time | `benchmarks/`, nightly |

A feature is not done until it has a test at the level that can falsify it, and a Python hook feature
is not done until there is an equivalence test against the C++ it replaces.

**As built (2026-10-19):** 75 tests, ~90 s.

| suite | covers |
|---|---|
| `test_vs_biofvm.py` | the golden test: our diffusion against BioFVM itself, 2-D and 3-D, **4.6e-14** |
| `test_kernels.py` | force law vs `add_potentials`, Adams-Bashforth, the BioFVM secretion solver, Hill responses |
| `test_codegen.py` | generated Python rebuilds a hash-identical model, all 18 sample projects; the cells CSV round trip |
| `test_regressions.py` | the defects found in review (cycle ring closure, volume targets at division, recycled slots, silent cell dropping, …) |
| `test_capabilities.py` | two simulations in one process, branch, checkpoint, hooks on every slot, stop conditions, sweeps, custom data, interactions |
| `test_ports.py` | every ported project installs and runs, plus behavioural checks (workers carry cargo, predators starve, stem cells differentiate) |
| `test_initial.py` | the placement heuristics and PhysiCell's own constructor defaults |

Not yet tested, and the main gap: the cell model against a PhysiCell *run*. §13.1's golden-run
equivalence covers the microenvironment only.

## 14. Packaging and CI

- **Build:** scikit-build-core + CMake over the vendored upstream sources; nanobind bindings.
- **Required at runtime:** Python ≥ 3.11, NumPy. Nothing else.
- **Optional:** `pandas`/`polars`, `xarray`, `zarr`/`h5py`, `matplotlib`, `pcdl`, `mpi4py`, a C++
  toolchain (tier C only).
- **Wheels:** manylinux_2_28 x86_64 + aarch64, macOS arm64, Windows x86_64 (tiers A/B), CPython
  3.11–3.14, with the OpenMP runtime bundled (ADR-011).
- **CI:** GitHub Actions — build matrix, unit + contract tests on every push; §13.1 fast subset on
  every push; nightly full equivalence, validation and benchmarks; a scheduled job that bumps the
  upstream pin on a branch and reports which golden tests move.

## 15. Coding conventions

- Python: type-hinted, `ruff` + `mypy --strict` on the public API, NumPy-style docstrings, units in
  every docstring that takes a number.
- C++ (the shim only): C++17, `clang-format`, no owning raw pointers, exceptions translated to Python
  errors at the boundary. Upstream's own style is left alone.
- Every default value and modelling assumption carries its reason and its source next to it.
- Every public function that returns a copy says so in its first docstring line.

## 16. Repository layout

```
open-physicell/
├── DESIGN.md  ROADMAP.md  README.md  pyproject.toml  CMakeLists.txt
├── PhysiCell/                 # submodule, pinned, unmodified (ADR-002)
├── patches/                   # the short, justified patch list (ADR-002)
├── cpp/
│   ├── facade/                # openphysicell::Simulation (ADR-014): setup, step, gather/scatter,
│   │                          #   hook dispatch, scalars, plugin loader — mergeable upstream
│   └── bindings/              # nanobind module  -> physicell._core
├── python/physicell/
│   ├── model/                 # L3: Model, CellType, Substrate, Rule, initial conditions
│   ├── state/                 # L2: CellView, SignalTable, SubstrateField
│   ├── sim/                   # L4: Simulation, worker protocol, checkpoints, Result
│   ├── hooks/                 # hook decorators, contexts, transport
│   ├── plugins/               # tier-C build, cache, entry points
│   ├── ensemble/              # samplers, schedulers, storage, resume
│   ├── compare/               # registry, metrics, diffs
│   ├── io/                    # xml, rules csv, multicellds, zarr/hdf5, pcdl bridge
│   ├── viz/                   # plots, live view
│   └── cli.py                 # physicell inspect | run | build-kernel | sweep
├── models/                    # registered models (§10.1), starting with upstream's samples
├── tests/{unit,contract,equivalence,repro,validation}/
├── benchmarks/
└── docs/                      # Jupyter Book; the examples are the docs
```

**As built (2026-10-19):**

```
open-physicell/
├── pyproject.toml                 pip install -e ".[notebooks,dev]"
├── python/physicell/              the package (see §5), with kernels.cpp inside it
├── tools/                         survey, gaps, survey_published, make_notebooks, make_index,
│                                  run_notebooks.sh
├── tests/                         75 tests, including the BioFVM oracle (biofvm_reference.cpp)
├── notebooks/                     37 executed notebooks + data/*.csv + a generated README
└── .venv/                         the environment, with an `open-physicell` Jupyter kernel
../published/                      the published models surveyed in §10.1
```

## 17. Open questions

1. **Name.** `open-physicell` / `import physicell` is free on PyPI and mirrors the parent project, but
   it could read as official. Alternatives: `physicell-py`, `pcpy`, or a name that does not contain
   PhysiCell at all. Worth asking the PhysiCell team before publishing, both for the name and because
   their blessing would matter more than the name does.
2. **Subprocess by default, or in-process by default?** Subprocess is safer and makes ensembles
   natural; in-process is simpler to debug and has cheaper hooks. The spike should measure the hook
   transport cost before this is fixed.
3. **Generated `custom.cpp`.** How far should ADR-006's round trip go? Serializing a vectorized NumPy
   hook to C++ is not generally possible. Options: only serialize rules and parameters and mark hooked
   models non-portable; or define a small expression subset that does translate.
4. **Does upstream want any of this?** A `reset()` and SoA mirrors would simplify the design and
   benefit everyone. Worth proposing early; the design must work without them.
5. **`pcdl` dependency or built-in reader?** Depending on it buys a mature reader and a maintenance
   coupling. Current lean: optional dependency, built-in minimal reader.
6. **Scope of intracellular models in v1.** PhysiBoSS and libRoadrunner are compile-time addons today;
   exposing them through tier C is easy, exposing them as a Python-level `IntracellularModel` protocol
   is not. Likely v1: tier C only, protocol in v2.
7. **GPU.** Only after §9.1 says where the time goes.
8. **Adopt an MPI kernel, or stay single-node?** §8.3 says the honest options are MPI-E (free) or
   driving PhysiCell-X / BioFVM-B (someone else's maintenance burden, and both are behind upstream).
   Deciding needs one number from this group: the largest domain and cell count anyone here actually
   wants to run. Below ~10⁷ cells and ~10⁸ voxels the question does not arise.
9. **Build ensembles, or build on UQ-PhysiCell?** It already has samplers, MPI execution, BO and
   ABC-SMC calibration, summary statistics and a result database that Studio can read (§12.5).
   Reinventing that would split a small community's effort. Current lean: implement `Sampler`,
   `Objective` and `Storage` (§7) as adapters over `uq_physicell` where it fits, contribute an
   in-process execution backend to it, and keep only the model object and the live simulation as
   open-physicell's own. Needs a conversation with its authors before Phase 3 is designed.

## 18. Phasing, and whether to build it

### 18.0 Phase 0 — the two-week spike (do this before committing)

Three things can kill the design; nothing else should be built until they are answered. Each is a
yes/no with a number attached:

1. **Can the kernel be driven from Python at all?** Reproduce `main.cpp`'s setup in a shim, step the
   `template` project from Python, and pass §13.1's golden-run equivalence at `threads=1`.
2. **What does a hook cost?** Implement one vectorized phenotype hook, measure against the identical
   C++ function on `cancer_immune` at 10³, 10⁴ and 10⁵ cells. If the overhead is not in single-digit
   percent, ADR-005's bet is wrong and the design needs rethinking (likely toward compiling rules and
   expressions rather than calling Python).
3. **Does a macOS arm64 wheel build and work?** OpenMP packaging is the known-fragile part (ADR-011).

Deliverable: a one-page memo with three numbers and a go/no-go. The §9.1 profile falls out of it.

### 18.1 Later phases (sketch; details belong in ROADMAP.md)

| Phase | Scope |
|---|---|
| 1a | The kernel-free partial flows (§11.6 items 1, 2, 3, 6, 8, 9): `Model`, XML/rules round trip, grammar, rule curves, cycle analysis, `check()`, initial conditions, diff. Useful on its own, and it is the input to every later phase. |
| 1b | `Simulation` + state access + the kernel-backed partial flows (0-D, diffusion-only, force curves) + tier-A ingest of every sample project. |
| 2 | Hooks (vectorized, step, per-cell), checkpoints, branch, live plotting, `Result` + provenance. |
| 3 | `Ensemble`: samplers, local + SLURM + nanoHUB schedulers, resume, storage. |
| 4 | `compare`: registry, metrics, model diff; the first reproduced published models. |
| 5 | Tier-C plugins, addons (PhysiBoSS, PhysiMeSS, dFBA, RoadRunner), calibration. |
| 6 | Measured optimizations (§9.3), if the profile justifies them; evaluate an MPI-D kernel behind the `Partition` seam (§8.3) if the size question in open question 8 says it is needed. |

### 18.1b Status (2026-10-19)

Phases 1a, 1b and most of 2 exist: model object, round trip, checks, partial flows, simulation,
state access, hooks, branch, checkpoints, sweeps, interactions, attachments, custom data, fibres,
three visualization backends, 37 executed notebooks and the published-model survey. Not started:
the ensemble schedulers of Phase 3 beyond the in-process sweep, the comparison layer of Phase 4,
tier-C plugins (dropped, ADR-015) and intracellular models.

### 18.2 Is a library like this useful?

**Where it clearly wins**

- **The compiler goes away for most work.** Tier A covers rules-driven models, which is where
  PhysiCell itself has been heading since the CBHG engine landed.
- **Ensembles become ordinary.** Sweeps, replicates, calibration and active learning need a
  programmable run, and today every group rebuilds that out of subprocesses and output directories.
- **In-memory state changes what is possible.** Stepping, intervening mid-run, branching from a
  common state, stopping on a condition, and coupling to an ML model in the loop are not available at
  all through a file-based executable.
- **Comparing models becomes a diff instead of a reading exercise.** This is the part worth a paper,
  and the part the parent project already proved out as a benchmark book.
- **Deployment.** A serializable model with a content hash is directly submittable to a queue or a
  nanoHUB tool, with no per-point project directory and no recompilation.
- **The pieces are useful without the whole** (§11.6). Six of the ten partial flows need no kernel,
  so the library has a floor of value that does not depend on the hard parts working.

**Where it does not help, and the honest risks**

- **It does not make one simulation faster.** Best case it is PhysiCell's speed minus a small
  constant, and the portable build may cost a few percent against `-march=native`.
- **Hard custom-C++ modelers still need a compiler.** Tier C is a convenience for them, not a
  liberation.
- **Upstream drift is a permanent tax.** Every PhysiCell release can move the golden tests. The pinned
  submodule plus nightly upgrade job is the mitigation, and it is real work forever.
- **The global-state constraint is structural.** It forces the subprocess design and rules out
  multiple simulations per interpreter. If upstream ever removes it the design gets simpler; until
  then, every feature lives with it.
- **Ecosystem overlap is larger than it first appears.** Studio 2.48 and UQ-PhysiCell 1.2.10 are
  actively moving into the artifact and ensemble space (§12.5), and `pcdl` owns output loading. The
  honest positioning is a layer *underneath* them — model object, live simulation, artifacts written
  during the run — consuming `pcdl`, writing files Studio opens, and adapting to UQ-PhysiCell rather
  than competing with it. If the project instead rebuilds their features, it splits a small
  community's attention and deserves to lose.
- **Three function-pointer slots may not be enough for real models.** The ingest survey of the 18
  sample projects (Phase 1) is what will show whether tier B is a broad class or a narrow one.

**The deciding question was answered (2026-10-19).** It was: *how much existing PhysiCell research
is tier A or tier B?* The survey says most of it:

- across upstream's 22 sample projects and 21 published model scenarios, **every rule runs** --
  27/27 in `rules_sample`, 36/36 in `pdac_therapy`, 196 of 196 in the published corpus;
- the real custom C++ in the whole sample corpus is **980 lines**, median 65 per project, and all
  eleven projects that have any are ported (ADR-015);
- what genuinely resists is narrow: a different data model (PhysiMeSS fibres, now a kernel,
  ADR-018), intracellular engines (a PyPI dependency, not C++), and six pre-1.6 projects that
  define their cell types in C++ with no XML at all.

So the larger claim holds: the corpus is reachable. What has *not* been established is the one
thing §13.1 was written for -- that the cell model reproduces a PhysiCell run. The microenvironment
does, to 4.6e-14. Until the agent side is compared the same way, this is a faithful reimplementation
with good evidence, not a drop-in replacement, and the documentation should keep saying so.

## 19. References

- A. Ghaffarizadeh, R. Heiland, S. H. Friedman, S. M. Mumenthaler, P. Macklin, "PhysiCell: An open
  source physics-based cell simulator for 3-D multicellular systems," PLoS Comput. Biol. 14(2):
  e1005991 (2018).
- A. Ghaffarizadeh, S. H. Friedman, P. Macklin, "BioFVM: an efficient, parallelized diffusive
  transport solver for 3-D biological simulations," Bioinformatics 32(8), 1256 (2016).
- P. Macklin et al., "Agent-based modeling of the tumor microenvironment," and the PhysiCell
  user guide and blog (physicell.org) for the cell behavior hypothesis grammar.
- R. Heiland, D. Bergman, B. Lyons, G. Waldow, J. Cass, H. B. Mohammad Nia, M. Ruscone, V. Noël,
  A. Shutt, P. Macklin, "PhysiCell Studio: a graphical tool to make agent-based modeling more
  accessible," Gigabyte (2024).
- E. Bruch et al., `physicelldataloader` (pcdl), github.com/elmbeech/physicelldataloader.
- H. L. Rocha et al., UQ-PhysiCell — uncertainty quantification, sensitivity analysis and
  calibration for PhysiCell models (PyPI `uq-physicell` 1.2.10, github.com/heberlr/UQ_PhysiCell,
  uq-physicell.readthedocs.io).
- PhysiCell-Studio source, github.com/PhysiCell-Tools/PhysiCell-Studio (2.48.0, checked out at
  `./PhysiCell-Studio` for the §12.5 reading).
- I. Gonçalves et al., "PhysiCOOL: A generalized framework for model Calibration and Optimization Of
  modeLing projects," JOSS / physicool.readthedocs.io.
- G. Letort et al., "PhysiBoSS: a multi-scale agent-based modelling framework integrating physical
  dimension and cell signalling," Bioinformatics 35(7), 1188 (2019).
- V. Noël et al., "PhysiBoSS 2.0: a sustainable integration of stochastic Boolean and agent-based
  modelling frameworks," npj Syst. Biol. Appl. 7, 1 (2021).
- G. Saxena, M. Ponce-de-Leon, A. Montagud, D. Vicente Dorca, A. Valencia, "BioFVM-X: An MPI+OpenMP
  3-D Simulator for Biological Systems," Computational Methods in Systems Biology, LNCS 12881,
  Springer (2021), doi:10.1007/978-3-030-85633-5_18.
- "A Novel Scalable High Performance Diffusion Solver for Multiscale Cell Simulations" (BioFVM-B),
  arXiv:2602.05017 (2026) — blocked pipelined Thomas, contiguous 4-D storage, scaling numbers quoted
  in §8.3.
- PhysiCell-X, PerMedCoE code of the month (2023), hpc-portal.eu — MPI+OpenMP PhysiCell with
  x-direction decomposition; the 4.06× / 40.3 % figures in §8.3 come from these slides, not from a
  peer-reviewed scaling study.
- G. An, et al., and the MultiCellDS standard: S. H. Friedman et al., "MultiCellDS: a standard and a
  community for sharing multicellular data," bioRxiv (2016).
- open-nemo DESIGN.md (this project's parent; ADR conventions, layering, provenance and benchmark
  practice).

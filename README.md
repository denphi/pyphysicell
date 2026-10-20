# pyphysicell: PhysiCell models in Python, with C++ only as computation kernels

**Version:** 0.1.1

**Release date:** 20 October 2026

**Based on:** PhysiCell **1.14.2** (and the BioFVM vendored in it). This is the
version every algorithm, default value and parity test here was written
against. PhysiCell changes its defaults and its rules grammar between
releases, so a model built for a different version may read differently --
check `pyphysicell.PHYSICELL_VERSION` before trusting a comparison.

**Developed and maintained by:** Daniel Mejia.

**Supported by:** the Network for Computational Nanotechnology and
[nanoHUB](https://nanohub.org), Purdue University.

**Reference:**

	(a) This repository.

	(b) A Ghaffarizadeh, R Heiland, SH Friedman, SM Mumenthaler, and P Macklin.
	    "PhysiCell: an Open Source Physics-Based Cell Simulator for Multicellular
	    Systems." PLoS Comput. Biol. 14(2): e1005991, 2018.
	    DOI: 10.1371/journal.pcbi.1005991

	(c) A Ghaffarizadeh, SH Friedman, and P Macklin. "BioFVM: an efficient
	    parallelized diffusive transport solver for 3-D biological simulations."
	    Bioinformatics 32(8): 1256-8, 2016. DOI: 10.1093/bioinformatics/btv730

pyphysicell is an independent Python reimplementation of
[PhysiCell](http://physicell.org), the open-source agent-based simulator for
multicellular systems. PhysiCell is a C++ application: a model is an XML file,
a run is an executable, and a result is a folder of files. That is a good fit
for a cluster and a poor fit for a notebook, an interactive session or an
agent. Here the model is a Python object, the simulation stays alive in the
process, and C++ is kept for the inner loops only -- diffusion, mechanics,
forces -- reached through ctypes over NumPy buffers. No project is compiled,
no executable is built, no output folder is parsed.

**This is not PhysiCell and not an official PhysiCell product.** It tracks one
specific release, **PhysiCell 1.14.2**: its microenvironment matches that
version's BioFVM to 4.6e-14 and its model parsing matches that version on
5,826 values across 17 projects, but its agent-side trajectories have never
been compared against a compiled PhysiCell run, of any version. Nothing here
is guaranteed against 1.13 or earlier, or against a later release. See *What
is verified* below, and cite the PhysiCell papers, not this.

```python
import pyphysicell as pc

model = pc.load("PhysiCell/sample_projects/rules_sample")   # their project, unmodified
print(pc.check(model))                                      # dimensional analysis, milliseconds
sim = pc.Simulation(model, seed=0)
sim.run(until=720, record_every=30)                         # population curve recorded *during* the run
sim.microenvironment.density[..., 0] = 38.0                 # intervene; the state is yours
sim.run(until=900)
```

No C++ project is compiled, no executable is built, no output folder is parsed.

## What the library adds over an XML and an executable

| | PhysiCell workflow | here |
|---|---|---|
| ask what a model implies before running | read the XML | `pc.check(model)`, `cycle.mean_duration()`, rule response curves -- milliseconds |
| change something mid-run | new XML, run again from t=0 | edit `sim.cells` / `sim.microenvironment`, call `sim.run()` again |
| two futures from one state | two runs from the beginning | `sim.branch()` -- a deepcopy, because the state is Python |
| save and resume | — | `sim.save(path)` / `pc.Simulation.load(path)` |
| logic outside the rules grammar | custom C++, recompile | `@sim.hook("phenotype", cell_type=...)`, arrays, once per step |
| stop on a condition | run to max_time, search the output | `sim.run(..., stop_when=lambda s: ...)` |
| a parameter sweep | one subprocess and one output folder per point | `pc.sweep(model, change, values, replicates=n)` in one process |
| two simulations at once | impossible (process globals) | two objects |

Each is covered by a test in [tests/test_capabilities.py](tests/test_capabilities.py) and
demonstrated on every sample project in the notebooks.

## Notebooks

Every PhysiCell sample project is a **self-contained notebook** in
[notebooks/](notebooks/README.md): the model is Python, generated from the project's XML
and rules CSV by `pyphysicell.codegen`, so a notebook needs no PhysiCell checkout at all.
A test asserts that the generated code rebuilds a model with the same content hash as
the XML it came from, for all 18 sample projects.

Initial conditions are the one thing not inlined: they ship as CSV in
[notebooks/data/](notebooks/data/), read with `pc.read_cells_csv`, so a notebook needs
only itself, its data file and the package. (Inlining 1,300 cells cost `rules_sample`
124 KB of literal tuples; the file is 48 KB now.) A test walks every notebook and fails
if it references a data file that is not shipped.

```python
model = pc.load("PhysiCell/sample_projects/rules_sample")   # read a project
print(pc.codegen.to_python(model))                          # write it out as Python
```

Every project notebook has the same eight sections: the model as an object, answers
before simulating, a run, steering it mid-run, two futures from one state, a hook doing
something the rules grammar cannot express, an in-process sweep with replicates, and a
stopping condition. Sections 3-8 are the ones that are impossible, or mean editing C++,
in the file workflow.

## Architecture

```
Python          model (from PhysiCell XML + rules CSV) · state (NumPy, struct-of-arrays)
                time loop · rules engine · cycle/volume/death · analysis · plots
   |  flat arrays, no hidden state
C++ kernels     diffusion (LOD/Thomas) · gradients · secretion-uptake · cell list ·
                mechanics forces · Adams-Bashforth positions
                OpenMP when available, std::thread otherwise
                (python/pyphysicell/kernels.cpp, 330 lines)
```

Everything PhysiCell keeps in process globals (`microenvironment`, `all_cells`,
`cell_defaults`, …) lives in ordinary Python objects here, so several models can exist in
one interpreter, state is picklable, and nothing is hidden between steps.

## What is verified

| Check | Result |
|---|---|
| Diffusion vs **BioFVM itself** (2-D, 400 voxels, 2 substrates, Dirichlet, 100 steps) | max relative difference **4.6e-14** |
| Diffusion vs **BioFVM itself** (3-D, 8000 voxels, 50 steps) | max relative difference **3.5e-14** |
| Pure decay vs analytic exponential | 2.5e-4 (the LOD splitting error, same as PhysiCell's) |
| Mass conservation, no decay, no-flux | 4.6e-14 |
| Mechanics force law vs `Cell::add_potentials`, 20 random separations | exact to 1e-13 |
| Position update vs `Cell::update_position` (Adams–Bashforth) | exact to 1e-15 |
| Secretion/uptake vs the BioFVM semi-implicit solver | exact to 1e-15 |
| Hill rule response vs `multivariate_Hill_response_function` | exact |

`tests/biofvm_reference.cpp` builds against the vendored PhysiCell checkout and is the
oracle for the first two rows: BioFVM's own solver, same case, binary comparison.

`tests/test_regressions.py` pins the defects found in review: the cycle ring must close
back to phase 0, cells must keep their size across divisions (division halves the volume
*and* the targets, the growth phase doubles the targets again), freed cell slots must be
cleared before reuse, internalized substrates are split at division, `apply_to_dead`
decides whether a rule touches a corpse, chained `run()` calls must not duplicate a
recorded time point, and `<chemotaxis>` must reach the motility bias.

```
.venv/bin/python -m pytest tests -q        # 16 passed
```

## What it does

- reads a PhysiCell project: `PhysiCell_settings.xml`, the CBHG rules CSV, `cells.csv`
- substrates: diffusion, decay, Dirichlet faces, gradients, cell secretion/uptake/export
- cells: volume ODE, cycle (any ring of phases: `live`, flow-cytometry, Ki67…),
  apoptosis and necrosis with swelling and lysis, mechanics, motility with chemotaxis,
  division with PhysiCell's daughter placement
- rules: the CBHG grammar, vectorized over the population (one NumPy pass per cell type
  per phenotype step, not one Python call per cell)
- `check()`: diffusion lengths against `dx`, motility step against the mesh, cycle
  durations in closed form from the phase graph
- `inspect()`: what fraction of a given project this can run, naming every rule it skips
- `codegen.to_python()`: a model written back out as a readable, self-contained script
- `initial.auto()`: ports a project's C++ `setup_tissue` placement from its own
  `number_of_*` and `*_radius` parameters, and says so in words
- cell definitions inherit like PhysiCell's: `parent_type`, else the `default` definition,
  else the values from PhysiCell's own constructors

## What it does not do (yet)

Measured, not guessed: `tools/gaps.py` reads every sample project and reports what
stands between it and its real model.

**Every rule in every sample project now runs** -- phagocytosis, attack and damage,
transformations, fusion, asymmetric division, `contact with <type>`, `custom:<name>`.
`rules_sample` went from 14 of 27 rules to 27 of 27.

What is left is C++ that projects write themselves. Five of the eight `Cell_Functions`
slots have hook equivalents, covering 54 of the 59 real (non-stub) assignments in the
sample projects -- the logic still has to be written in Python, but it no longer needs a
compiler:

| slot | assignments across projects | here |
|---|---|---|
| `update_phenotype` | 23 | `@sim.hook("phenotype")` |
| `custom_cell_rule` | 11 | `@sim.hook("custom_rule")` |
| `update_migration_bias` | 11 | `@sim.hook("migration_bias")` |
| `contact_function` | 8 | `@sim.hook("contact")` over touching pairs |
| `cell_division_function` | 1 | `@sim.hook("division")` over parent/daughter pairs |
| `instantiate_cell` | 3 | **no** -- needs a Cell subclass (PhysiMeSS fibres) |
| `update_velocity` | 1 | **no** -- custom mechanics (PhysiMeSS) |
| `pre/post_update_intracellular` | 2 | **no** -- PhysiBoSS, libRoadrunner, dFBA |

Also still missing, and known:

- **attachments and springs**, so `contact` hooks run over neighbours rather than attached
  pairs, and `cell attachment rate` / `cell detachment rate` do nothing
- basement-membrane functions, `is movable` as a behavior
- intracellular models of any kind (the cell-level model runs; its inner model does not)
- MultiCellDS output and SVG export (results stay as arrays and dataframes)

## Layout

```
python/pyphysicell/
  kernels.cpp                   the C++ computation kernels (C ABI, OpenMP/std::thread)
  _kernels.py                   builds and binds them on first import (ctypes, no build step)
  model.py  io_xml.py           the model object and the PhysiCell file readers
  state.py                      Microenvironment and Cells (struct-of-arrays)
  phenotype.py  rules.py        volume/cycle/death, and the CBHG engine
  signals.py                    the signal/behavior dictionary
  simulation.py                 the time loop and the tier-1 scalars
  check.py  viz.py              model checks and plots
tests/                          unit tests + the BioFVM golden test
notebooks/01_rules_sample.ipynb the demo
```

Requires Python ≥ 3.10, NumPy, a C++17 compiler; matplotlib and pandas for the notebook.

## Credit

This library is an independent reimplementation. It contains no PhysiCell
source: the algorithms were reimplemented in Python and C++ from the published
papers and the PhysiCell 1.14.2 source, and several sample projects' custom C++
modules were ported to Python (`python/pyphysicell/ports/`). Everything it can
do, it can do because the people below did the work first. If you use it,
cite their papers — not this package.

### PhysiCell and BioFVM — the simulator this reimplements

- **PhysiCell** — Ahmadreza Ghaffarizadeh, Randy Heiland, Samuel H. Friedman,
  Shannon M. Mumenthaler, Paul Macklin. *PhysiCell: an open source
  physics-based cell simulator for 3-D multicellular systems.* PLoS Comput.
  Biol. 14(2): e1005991 (2018). doi:10.1371/journal.pcbi.1005991
- **BioFVM** — Ahmadreza Ghaffarizadeh, Samuel H. Friedman, Paul Macklin.
  *BioFVM: an efficient, parallelized diffusive transport solver for 3-D
  biological simulations.* Bioinformatics 32(8): 1256–8 (2016).
  doi:10.1093/bioinformatics/btv730
- Copyright (c) 2015–2024, Paul Macklin and the PhysiCell Project, BSD-3-Clause.
  PhysiCell is developed by Paul Macklin's lab and a wider community of
  contributors: github.com/MathCancer/PhysiCell

### PhysiCell Studio — the source of this library's default values

- R. Heiland, D. Bergman, B. Lyons, G. Waldow, J. Cass, H. B. Mohammad Nia,
  M. Ruscone, V. Noël, A. Shutt, P. Macklin. *PhysiCell Studio: a graphical
  tool to make agent-based modeling more accessible.* Gigabyte (2024).

### Addons

- **PhysiMeSS** (ECM fibres — reimplemented here as kernels) — V. Noël,
  M. Ruscone, R. Shuttleworth, C. K. Macnamara. *PhysiMeSS — a new
  PhysiCell addon for extracellular matrix modelling.* bioRxiv
  2023.10.27.564365 (2023). doi:10.1101/2023.10.27.564365
- **PhysiBoSS** (*not implemented here*; the cell-level model of PhysiBoSS
  projects runs, its Boolean layer does not) — Gaelle Letort, Arnau Montagud,
  Gautier Stoll, Randy Heiland, Emmanuel Barillot, Paul Macklin, Andrei
  Zinovyev, Laurence Calzone. Bioinformatics 35(7): 1188–96 (2019).
  doi:10.1093/bioinformatics/bty766 — and **PhysiBoSS 2.0**: Miguel
  Ponce-de-Leon, Arnau Montagud, Vincent Noël, Annika Meert, Gerard Pradas,
  Emmanuel Barillot, Laurence Calzone, Alfonso Valencia. npj Syst. Biol. Appl.
  9(1): 54 (2023). doi:10.1038/s41540-023-00314-4
- **libRoadRunner** (*not implemented here*) — Endre T. Somogyi,
  Jean-Marie Bouteiller, James A. Glazier, Matthias König, J. Kyle Medley,
  Maciej H. Swat, Herbert M. Sauro. Bioinformatics 31(20): 3315–21 (2015).
  doi:10.1093/bioinformatics/btv363
- **MaBoSS** — the Boolean engine behind PhysiBoSS, by the Institut Curie
  Computational Systems Biology of Cancer group. Not vendored or used here.

### The cell behavior hypothesis grammar, and the published models replicated here

The rules engine implements the grammar from, and the `published/` corpus
replicates the models of:

- Johnson JAI, Bergman DR, Rocha HL, Zhou DL, Cramer E, Mclean IC, Dance YW,
  Booth M, Nicholas Z, Lopez-Vidal T, Deshpande A, Heiland R, Bucher E,
  Shojaeian F, Dunworth M, Forjaz A, Getz M, Godet I, Kurtoglu F, Lyman M,
  Metzcar J, Mitchell JT, Raddatz A, Solorzano J, Sundus A, Wang Y,
  DeNardo DG, Ewald AJ, Gilkes DM, Kagohara LT, Kiemen AL, Thompson ED,
  Wirtz D, Wood LD, Wu PH, Zaidi N, Zheng L, Zimmerman JW, Phillip JM,
  Jaffee EM, Gray JW, Coussens LM, Chang YH, Heiser LM, Stein-O'Brien GL,
  Fertig EJ, Macklin P. *Human interpretable grammar encodes multicellular
  systems biology models to democratize virtual cell laboratories.* Cell
  (2025). doi:10.1016/j.cell.2025.06.048 — preprint: *Digitize your Biology!
  Modeling multicellular systems through interpretable cell behavior*,
  doi:10.1101/2023.09.17.557982
- **collective-invasion / cell-ECM framework** — Metzcar J, Duggan BS,
  Fischer B, Murphy M, Heiland R, Macklin P. *A simple
  framework for agent-based modeling with extracellular matrix.* Bull. Math.
  Biol. 87(3): 43 (2025). doi:10.1007/s11538-024-01408-8
- **breastcancer-hypoxia** — Heber Lima da Rocha, with the Macklin and Gilkes
  labs; supported by the JKTG Foundation, BCRF, the IU Engineered nanoBIO Node
  and NCI U01CA232137.
- **pc3types**, **three_types** — Paul Macklin.
- The sample projects ported in `python/pyphysicell/ports/` are the work of the
  PhysiCell project and its contributors.

### Related Python work in this ecosystem, which this project learned from

- **pcdl / physicelldataloader** — Elmar Bucher et al.
- **UQ-PhysiCell** — Heber L. Rocha et al.
- **PhysiCOOL** — I. G. Gonçalves et al.
- **PhysiCell-X** (MPI) — Gaurav Saxena, Miguel Ponce-de-Leon, Arnau Montagud,
  David Vicente Dorca, Alfonso Valencia.

## License

BSD 3-Clause, the same license as PhysiCell — see [LICENSE](LICENSE). The
PhysiCell copyright notice is retained there, as condition 1 of that license
requires for the ported modules and the reimplemented algorithms.

Per condition 3 of that license, neither the PhysiCell name nor the names of
its contributors are used to endorse or promote this package. This is not an
official PhysiCell product and the PhysiCell authors are not responsible for
it; please do not send them bug reports about it.

## Acknowledgements

pyphysicell is developed and maintained by **Daniel Mejia**, with support from
the **Network for Computational Nanotechnology** and
**[nanoHUB](https://nanohub.org)**, Purdue University.

The PhysiCell project itself has been supported by the National Science
Foundation, the National Cancer Institute, the Breast Cancer Research
Foundation and the Jayne Koskinas Ted Giovanis Foundation, among others; see
the upstream repositories for their own acknowledgements.

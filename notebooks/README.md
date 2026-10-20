# Sample projects as notebooks

Every PhysiCell sample project, each as a **self-contained notebook**: the model is
Python generated from the project's XML and rules CSV, so nothing here reads a
PhysiCell file at run time. `pyphysicell.codegen` writes them and a test asserts the
generated code rebuilds a model with the same content hash as the XML it came from.

`00_tour.ipynb` is the exception and the place to start: it *loads* a project to show
the reader, the checks, the rules, stepping, intervening mid-run and re-colouring.

Initial conditions ship as CSV beside the notebooks, in `data/`, so a notebook needs
only itself, its data file and the `physicell` package -- never the PhysiCell tree.
Projects that place their cells in C++ have that placement ported instead, either in
`pyphysicell.ports.<project>` or by `pyphysicell.initial.auto`.

| notebook | dim | voxels | substrates | cell types | cells | rules (run/skipped) | executed |
|---|---|---|---|---|---|---|---|
| [asymmetric_division](asymmetric_division.ipynb) | 2D | 2,500 | 1 | 3 | 1 | 3/0 | yes, 61s |
| [biorobots](biorobots.ipynb) | 2D | 10,000 | 2 | 3 | 165 | — | yes, 140s |
| [cancer_biorobots](cancer_biorobots.ipynb) | 2D | 5,625 | 3 | 3 | 1,071 | — | yes, 202s |
| [cancer_immune](cancer_immune.ipynb) | 3D | 421,875 | 2 | 2 | 28,959 | — | yes, 189s |
| [celltypes3](celltypes3.ipynb) | 2D | 2,500 | 4 | 4 | 75 | — | yes, 171s |
| [custom_division](custom_division.ipynb) | 2D | 400 | 1 | 2 | 4 | — | yes, 141s |
| [episode](episode.ipynb) | 2D | 2,500 | 3 | 3 | 36 | — | yes, 124s |
| [heterogeneity](heterogeneity.ipynb) | 2D | 10,000 | 1 | 1 | 889 | — | yes, 157s |
| [immune_function](immune_function.ipynb) | 2D | 625 | 5 | 6 | 28 | 5/0 | yes, 160s |
| [interactions](interactions.ipynb) | 2D | 1,600 | 5 | 7 | 660 | — | yes, 120s |
| [mechano](mechano.ipynb) | 2D | 2,500 | 1 | 2 | 548 | — | yes, 141s |
| [physimess](physimess.ipynb) | 2D | 1,600 | 1 | 3 | 2,000 | — | yes, 8s |
| [pred_prey_farmer](pred_prey_farmer.ipynb) | 2D | 2,500 | 3 | 3 | 85 | — | yes, 114s |
| [rules_sample](rules_sample.ipynb) | 2D | 2,500 | 5 | 6 | 1,300 | 27/0 | yes, 76s |
| [template](template.ipynb) | 2D | 2,500 | 1 | 1 | 5 | — | yes, 103s |
| [virus_macrophage](virus_macrophage.ipynb) | 2D | 2,500 | 2 | 2 | 1,060 | — | yes, 151s |
| [worm](worm.ipynb) | 2D | 2,500 | 1 | 1 | 11 | — | yes, 138s |
| [cancer_invasion](cancer_invasion.ipynb) | 2D | 2,500 | 4 | 2 | 123 | — | yes, 273s |
| [physiboss_cell_lines](physiboss_cell_lines.ipynb) | 3D | 2,400 | 1 | 6 | 486 | — | yes, 77s |
| [template_BM](template_BM.ipynb) | 2D | 2,500 | 1 | 1 | 5 | — | yes, 146s |
| [cancer_metabolism](cancer_metabolism.ipynb) | 3D | 421,875 | 3 | 1 | 155 | — | yes, 141s |
| ecoli_acetic_switch | — | — | — | — | — | — | **no**: ParseError: mismatched tag: line 177, column 52 |
| [ode_energy](ode_energy.ipynb) | 2D | 2,500 | 3 | 1 | 313 | — | yes, 249s |

22 of 23 projects load.

## Where the cells come from

Most projects place their cells in C++ (`setup_tissue`), not in `cells.csv`, so the
notebook ports that placement with `pyphysicell.initial.auto`, driven by the project's
own `number_of_*` and `*_radius` user parameters. Each notebook prints exactly what
it did and says it is a port, not the original C++.

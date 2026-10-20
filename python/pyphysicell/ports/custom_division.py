"""Port of sample_projects/custom_division/custom_modules/custom.cpp."""

from __future__ import annotations


def install(sim) -> str:
    default_idx = sim.type_index["default"]
    ctype1_idx = sim.type_index["ctype1"]

    @sim.hook("division")
    def custom_division_function(ctx):
        # each daughter becomes "default" or "ctype1" with equal probability
        for k, daughter in enumerate(ctx.j):
            target = default_idx if ctx.rng.random() < 0.5 else ctype1_idx
            sim.convert_cell(int(daughter), target)
    return "daughters randomly become 'default' or 'ctype1'"

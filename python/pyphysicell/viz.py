"""Plots, drawn from state or from recorded scalars — never from saved images."""

from __future__ import annotations

import numpy as np


def _mpl_at_least(major: int, minor: int) -> bool:
    import matplotlib

    try:
        parts = matplotlib.__version__.split(".")
        return (int(parts[0]), int(parts[1])) >= (major, minor)
    except (ValueError, IndexError):      # a dev or vendored version string
        return True


def plot_cells(sim, color_by: str = "cell_type", substrate: str | None = None, ax=None,
               show_dead: bool = True, cmap="YlOrRd", alpha=0.9):
    """A 2-D snapshot: substrate as a background image, cells as circles."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import EllipseCollection

    ax = ax or plt.subplots(figsize=(6, 6))[1]
    d = sim.model.domain
    me = sim.microenvironment
    if substrate:
        field = me[substrate][0]
        im = ax.imshow(field, origin="lower", extent=(d.x_min, d.x_max, d.y_min, d.y_max),
                       cmap=cmap, alpha=0.85)
        plt.colorbar(im, ax=ax, fraction=0.046, label=substrate)

    c = sim.cells
    n = c.n
    if n:
        pos = c.view("position")
        diam = 2.0 * c.view("radius")
        if color_by == "cell_type":
            vals = c.view("type_idx").astype(float)
            colors = plt.get_cmap("tab10")(vals.astype(int) % 10)
        else:
            vals = sim.dictionary.signal_values(color_by, c, np.arange(n), sim.t)
            norm = (vals - vals.min()) / (np.ptp(vals) + 1e-16)
            colors = plt.get_cmap("viridis")(norm)
        if show_dead:
            dead = c.view("dead") == 1
            colors[dead] = (0.25, 0.25, 0.25, 1.0)
        # matplotlib renamed transOffset to offset_transform in 3.6, and the
        # older one swallows the unknown keyword into **kwargs before failing
        # in update() with an AttributeError, so this cannot be a try/except
        # on TypeError. nanoHUB's default kernel still ships 3.5.
        offset_kw = "offset_transform" if _mpl_at_least(3, 6) else "transOffset"
        coll = EllipseCollection(diam, diam, np.zeros(n), units="xy", offsets=pos[:, :2],
                                 facecolors=colors, edgecolors="k", linewidths=0.2,
                                 alpha=alpha, **{offset_kw: ax.transData})
        ax.add_collection(coll)
    ax.set_xlim(d.x_min, d.x_max)
    ax.set_ylim(d.y_min, d.y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    ax.set_title(f"t = {sim.t:.0f} min   n = {sim.cells.n}")
    return ax


def plot_populations(scalars, ax=None):
    """Tier-1 artifact: the population curves, recorded during the run."""
    import matplotlib.pyplot as plt

    ax = ax or plt.subplots(figsize=(7, 4))[1]
    t = np.asarray(scalars.times) / 60.0
    for name, counts in scalars.counts.items():
        if max(counts) > 0:
            ax.plot(t, counts, label=name, lw=2)
    if max(scalars.dead) > 0:
        ax.plot(t, scalars.dead, label="dead", lw=1.5, ls="--", color="0.4")
    ax.set_xlabel("time [h]")
    ax.set_ylabel("cells")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def plot_rule(rule, base_value: float, span=None, ax=None, n=200):
    """A rule's response curve: a pure function of the signal, no simulation."""
    import matplotlib.pyplot as plt

    ax = ax or plt.subplots(figsize=(5, 3))[1]
    hi = span[1] if span else 4.0 * rule.half_max
    lo = span[0] if span else 0.0
    s = np.linspace(lo, hi, n)
    ax.plot(s, rule.response(s, base_value), lw=2)
    ax.axvline(rule.half_max, ls=":", color="0.5")
    ax.set_xlabel(rule.signal)
    ax.set_ylabel(rule.behavior)
    ax.set_title(f"{rule.cell_type}: {rule.signal} {rule.direction} {rule.behavior}", fontsize=9)
    ax.grid(alpha=0.3)
    return ax

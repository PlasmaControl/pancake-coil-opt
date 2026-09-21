"""Every figure in the paper, drawn by this one file. Each figure reads
the .npz/.json data in paper/data and draws from it, so nothing here
re-runs an optimization or a Biot-Savart sum.
"""

import argparse
import json
import math
import pathlib
import shutil
import sys
import time

import matplotlib.colors
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np

SCRATCH = pathlib.Path(__file__).resolve().parent
REPOSITORY = SCRATCH.parent
DATA = SCRATCH / "data"
OUTPUT = SCRATCH / "plots"

sys.path.insert(0, str(SCRATCH))
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "scratch"))

RESOLUTION = "16x200"

WINDING_BACKGROUND = "#dcdcdc"


class MissingData(Exception):
    """Raised when a figure has no input at all, as opposed to one panel
    of several being unavailable."""


def save(fig, stem: str, dpi: int = 300) -> list[str]:
    """Write a figure as a PDF, the format the paper includes.

    dpi only affects any rasterized elements within the PDF (e.g. a
    pcolormesh panel), the PDF itself being vector, but it is carried per
    figure rather than fixed so these outputs stay pixel-identical to the
    per-figure scripts this file replaced.
    """
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{stem}.pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return [f"{stem}.pdf"]


def load_archive(name: str) -> dict | None:
    """Read one .npz from the data directory, or None if it is not there."""
    path = DATA / name
    if not path.exists():
        return None
    with np.load(path) as archive:
        return {field: archive[field] for field in archive.files}


def panel_extent(entries, extra_radius: float = 0.0) -> tuple[float, float]:
    """Axis limits covering every panel, so a difference between panels is
    a difference in geometry rather than in zoom.

    `extra_radius` accounts for figures that draw the conductor itself:
    the winding extends `coil.width` beyond the bore edge the arrays
    describe.
    """
    r_limit = 1.05 * max(float(np.abs(e["r_edges"]).max()) + extra_radius for e in entries)
    z_limit = 1.05 * max(float(np.abs(e["z_edges"]).max()) for e in entries)
    return r_limit, z_limit


def panel_size(r_limit: float, z_limit: float, target_width_in: float = 3.9):
    """Hold inches-per-meter constant across panels, derived from a target
    panel width rather than hard coded: this coil's r-span (~2.2 m) is an
    order of magnitude larger than the demo coils plots.py sizes for, and
    its own inches_per_meter would produce a ~26 in panel here.
    """
    inches_per_meter = target_width_in / (2 * r_limit)
    return target_width_in, 2 * z_limit * inches_per_meter


def draw_winding_map(ax, entry, values, *, cmap="viridis", norm=None,
                     vmin=None, vmax=None, outline=True, outline_width=0.7):
    """One winding panel: the field quantity on both mirrored halves, with
    the tape edges traced in black.

    shading="flat" against explicit cell EDGES is deliberate;
    shading="auto" would infer "nearest" and bleed half a cell into the
    bore.
    """
    r_edges, z_edges = entry["r_edges"], entry["z_edges"]
    ax.set_facecolor(WINDING_BACKGROUND)
    mesh = None
    for sign in (1, -1):
        mesh = ax.pcolormesh(
            sign * r_edges, z_edges, values, shading="flat", cmap=cmap,
            norm=norm, vmin=vmin, vmax=vmax, antialiased=False,
        )
    if outline:
        z_line = z_edges[:, 0]
        for sign in (1, -1):
            ax.plot(sign * r_edges[:, 0], z_line, color="black",
                    linewidth=outline_width, zorder=5)
            ax.plot(sign * r_edges[:, -1], z_line, color="black",
                    linewidth=outline_width, zorder=5)
    return mesh


def log_ticks(low: float, high: float) -> list[float]:
    """A 1-1.5-2-3-5-7 progression per decade, clipped to the data range.

    Over barely one decade matplotlib's LogLocator places a single label,
    leaving the axis effectively unlabelled, which is the case for every
    quantity in this paper.
    """
    decades = range(int(np.floor(np.log10(low))), int(np.ceil(np.log10(high))) + 1)
    return [m * 10.0 ** e for e in decades for m in (1, 1.5, 2, 3, 5, 7)]


def add_log_colorbar(fig, mesh, vmin: float, vmax: float, label: str,
                     rect=(0.895, 0.12, 0.018, 0.76)):
    """Colourbar in an explicit slot with readable ticks.

    fig.colorbar(ax=axes) fights tight_layout and lands the bar between
    the panels, hence add_axes. Ends are labelled to two decimals since
    they are real data values rather than round numbers; the interior
    ticks need one significant figure.
    """
    colorbar = fig.colorbar(mesh, cax=fig.add_axes(list(rect)))

    def position(value):
        return np.log10(value / vmin) / np.log10(vmax / vmin)

    ticks = [t for t in log_ticks(vmin, vmax)
             if vmin <= t <= vmax and 0.04 < position(t) < 0.96]
    ticks = [vmin, *ticks, vmax]
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([
        f"{t / 1e7:.2f}" if i in (0, len(ticks) - 1) else f"{t / 1e7:g}"
        for i, t in enumerate(ticks)
    ])
    colorbar.minorticks_off()
    colorbar.set_label(label)
    return colorbar


def comparison_grid(rows, columns, sources, quantity, *, stem, colour_label,
                    title_for, extra_title_rows=1):
    """The shared 2-D map figure: rows x columns of winding panels on one
    logarithmic colour scale, with placeholders for runs not yet landed.

    `sources` maps (row key, column tag) to a filename; `quantity` names
    the array to colour. Returns (outputs, missing).
    """
    data, missing = {}, []
    for row_key, _ in rows:
        for tag, _ in columns:
            entry = load_archive(sources[row_key, tag])
            if entry is None:
                missing.append(sources[row_key, tag])
                continue
            data[row_key, tag] = entry
    if not data:
        raise MissingData(f"none of the inputs for {stem} are present")

    vmin = min(float(e[quantity].min()) for e in data.values())
    vmax = max(float(e[quantity].max()) for e in data.values())
    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    r_limit, z_limit = panel_extent(data.values())
    panel_width_in, panel_height_in = panel_size(r_limit, z_limit)

    fig, axes = plt.subplots(
        len(rows), len(columns),
        figsize=(len(columns) * panel_width_in + 2.1,
                 len(rows) * panel_height_in + 1.5),
        squeeze=False,
    )

    mesh = None
    for row, (row_key, row_label) in enumerate(rows):
        for column, (tag, column_label) in enumerate(columns):
            ax = axes[row][column]
            entry = data.get((row_key, tag))
            if entry is None:
                ax.text(0.5, 0.5, f"{column_label}, {row_label}\npending",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=9, color="dimgray")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            mesh = draw_winding_map(ax, entry, entry[quantity], norm=norm)
            ax.set_xlim(-r_limit, r_limit)
            ax.set_ylim(-z_limit, z_limit)
            ax.set_aspect("equal")
            ax.set_title(title_for(entry, row_key, row_label, column_label),
                         fontsize=9 if extra_title_rows == 1 else 7.8)
            if row == len(rows) - 1:
                ax.set_xlabel("r [m]")
            if column == 0:
                ax.set_ylabel("z [m]")

    fig.subplots_adjust(left=0.07, right=0.87, bottom=0.08,
                        top=0.92 - 0.04 * (extra_title_rows - 1),
                        wspace=0.18, hspace=0.30 + 0.12 * (extra_title_rows - 1))
    add_log_colorbar(fig, mesh, vmin, vmax, colour_label)
    return save(fig, stem), missing



def collapsed_coil_schematic():
    """The self-intersection failure mode from the Constraints subsection.

    Builds its geometry rather than reading a saved run, but only the
    curve and its arc length, which is cheap. The parameters are a REAL
    reproduced optimizer output from
    minimize_average_lorentz_force_density_magnitude with no
    minimum_inner_radius floor (reproduce_midplane_collapse.py, try 5),
    not a fabricated illustrative curve: it genuinely collapses to
    r ~ 0.009 at z=0 and r ~ 0.0005 at z = +-0.887 z_top.
    """
    from pancakecoilopt.computer import basics

    curve_parameters = np.array([0.00936395, 0.38499528, -1.48194872, 1.82865414, -0.7437832])
    width = 0.19

    z_top = float(basics.find_z_bound_for_a_given_arc_length(1.9400000000000002, curve_parameters))
    z = np.linspace(-z_top, z_top, 1000)
    r_inner = np.array(basics.compute_curve(z, curve_parameters))
    r_outer = r_inner + width

    fig, ax = plt.subplots(figsize=(6.5, 8.0))
    copper, copper_edge = "#c98a4b", "#8a5a26"
    for sign in (1, -1):
        ax.fill_betweenx(z, sign * r_inner, sign * r_outer, facecolor=copper,
                         edgecolor=copper_edge, linewidth=0.8, zorder=1)

    ax.axvline(0.0, color="0.4", linestyle="-.", linewidth=1.0, zorder=0)
    ax.text(0.0, z_top * 1.04, "symmetry axis $r=0$", ha="center", va="bottom",
            fontsize=9, color="0.3")

    def radius_at(z_value):
        return float(basics.compute_curve(np.array([z_value]), curve_parameters)[0])

    pinches = [(0.0, radius_at(0.0)),
               (0.887 * z_top, radius_at(0.887 * z_top)),
               (-0.887 * z_top, radius_at(-0.887 * z_top))]
    labels = [f"$r(0) \\approx {pinches[0][1]:.3f}$\n(midplane)",
              f"$r(z) \\approx {pinches[1][1]:.4f}$\nat $z \\approx 0.89\\,z_{{\\mathrm{{top}}}}$",
              f"$r(z) \\approx {pinches[2][1]:.4f}$\nat $z \\approx -0.89\\,z_{{\\mathrm{{top}}}}$"]

    anchors = [pinches[0][1], pinches[1][0], pinches[2][0]]
    offsets = [0.10, 0.0, 0.0]
    for (z_p, _), label, anchor, offset in zip(pinches, labels, anchors, offsets, strict=True):
        ax.add_patch(plt.Circle((0.0, z_p), 0.05, fill=False,
                                edgecolor="tab:red", linewidth=1.6, zorder=4))
        ax.annotate(label, xy=(0.05, anchor), xycoords="data",
                    xytext=(0.34, anchor + offset), textcoords="data",
                    fontsize=10, color="tab:red", ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color="tab:red", linewidth=1.2))

    ax.set_xlabel("$r$")
    ax.set_ylabel("$z$")
    ax.set_xlim(-(r_outer.max() + 0.55), r_outer.max() + 0.55)
    ax.set_ylim(-z_top * 1.12, z_top * 1.12)
    ax.set_aspect("equal")
    fig.tight_layout()
    return save(fig, "collapsed_coil_schematic", dpi=250), []


def random_shape_convergence():
    """Discretization convergence at one fixed random feasible shape.

    Nothing is optimized: the shape is drawn at random, checked against
    the narrow constraints and held fixed while only the grid varies, so
    every curve is pure discretization error. Each panel rings the
    (16, 200) point used for the reported optimizations and annotates its
    difference from the finest grid computed, which stands in for the
    converged value.

    J_critical and Lorentz force density are evaluated AT the source
    points, so sweeping the source grid (nr, nz) is the whole story. The
    bore energy is a volume integral over a SECOND, independent grid laid
    across the bore, so being converged in (nr, nz) says nothing about
    being converged in that grid: its panels sweep the source grid with
    the bore grid fixed at the optimizer's 32x64, and the bore grid with
    the source fixed at (16, 200), one at a time.
    """
    seed, marked, reference = 0, (16, 200), (96, 200)
    families = [
        ("jc", f"jc_random_convergence_seed{seed}.json", "J_critical",
         [("mean_J_critical", "mean $J_c$", "A/m$^2$"),
          ("min_J_critical", "minimum $J_c$", "A/m$^2$")]),
        ("lorentz", f"lorentz_convergence_seed{seed}.json", "Lorentz force density",
         [("mean_force", "mean $F$", "N/m$^3$"),
          ("max_force", "maximum $F$", "N/m$^3$")]),
    ]

    def value_at(rows, nr, nz, field):
        return next((r[field] for r in rows if r["nr"] == nr and r["nz"] == nz), None)

    outputs, missing = [], []
    for key, filename, family, fields in families:
        path = DATA / filename
        if not path.exists():
            missing.append(filename)
            continue
        rows = json.loads(path.read_text())["results"]
        nz_values = sorted({r["nz"] for r in rows})
        nr_values = sorted({r["nr"] for r in rows})
        colours = plt.cm.viridis(np.linspace(0.1, 0.85, len(nz_values)))

        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
        for ax, (field, ylabel, unit) in zip(axes, fields, strict=True):
            for colour, nz in zip(colours, nz_values, strict=True):
                series = sorted((r for r in rows if r["nz"] == nz), key=lambda r: r["nr"])
                ax.plot([r["nr"] for r in series], [r[field] / 1e7 for r in series],
                        "o-", color=colour, markersize=4.5, label=f"nz={nz}")

            marked_value = value_at(rows, *marked, field)
            reference_value = value_at(rows, *reference, field)
            if reference_value is not None:
                ax.axhline(reference_value / 1e7, color="gray", linestyle="dashed",
                           linewidth=1.1, zorder=0,
                           label=f"reference (nr={reference[0]}, nz={reference[1]})")
            if marked_value is not None and reference_value is not None:
                difference = 100 * (marked_value - reference_value) / reference_value
                ax.plot(marked[0], marked_value / 1e7, "o", markersize=15,
                        markerfacecolor="none", markeredgecolor="crimson",
                        markeredgewidth=2.0, zorder=5,
                        label=f"used here (nr={marked[0]}, nz={marked[1]})")
                low, high = ax.get_ylim()
                near_top = (marked_value / 1e7 - low) / (high - low) > 0.6
                ax.annotate(
                    f"{difference:+.2f}% vs (nr={reference[0]}, nz={reference[1]})",
                    xy=(marked[0], marked_value / 1e7),
                    xytext=(14, -26) if near_top else (14, 16),
                    textcoords="offset points", fontsize=9.5, color="crimson",
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="crimson", linewidth=0.8),
                )
                print(f"  {family} {ylabel}: {difference:+.3f}% vs the finest grid")

            ax.set_xscale("log")
            ax.set_xticks(nr_values)
            ax.set_xticklabels([str(nr) for nr in nr_values])
            ax.minorticks_off()
            ax.grid(True, which="major", alpha=0.3)
            ax.set_xlabel("nr (radial divisions of the coil)")
            ax.set_ylabel(f"{ylabel} [$10^7$ {unit}]")
            ax.set_title(f"{family}: {ylabel}", fontsize=10)
            ax.legend(fontsize=8, loc="best")

        fig.tight_layout()
        outputs += save(fig, f"{key}_random_shape_convergence_seed{seed}")

    energy_filename = f"energy_convergence_seed{seed}.json"
    energy_path = DATA / energy_filename
    if not energy_path.exists():
        missing.append(energy_filename)
    else:
        payload = json.loads(energy_path.read_text())
        optimizer_grid = tuple(payload["optimizer_bore_grid"])
        source_reference = tuple(payload["source_reference"])
        source_rows = payload["source_sweep"]
        bore_rows = payload["bore_grid_sweep"]
        finest_bore = (max(r["grid_nr"] for r in bore_rows),
                        max(r["grid_nz"] for r in bore_rows))

        def energy_at(rows, x_key, group_key, x, group):
            return next((r["total_energy"] for r in rows
                        if r[x_key] == x and r[group_key] == group), None)

        panels = [
            (source_rows, "nr", "nz", marked, reference,
             "nr (radial divisions of the coil)",
             f"source discretization, bore grid fixed at "
             f"{optimizer_grid[0]}x{optimizer_grid[1]}"),
            (bore_rows, "grid_nr", "grid_nz", optimizer_grid, finest_bore,
             "grid_nr (radial divisions of the bore grid)",
             f"bore grid, source fixed at "
             f"{source_reference[0]}x{source_reference[1]}"),
        ]

        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
        for ax, (rows, x_key, group_key, marked_point, reference_point,
                 xlabel, title) in zip(axes, panels, strict=True):
            groups = sorted({r[group_key] for r in rows})
            x_values = sorted({r[x_key] for r in rows})
            colours = plt.cm.viridis(np.linspace(0.1, 0.85, len(groups)))

            for colour, group in zip(colours, groups, strict=True):
                series = sorted((r for r in rows if r[group_key] == group),
                                key=lambda r: r[x_key])
                ax.plot([r[x_key] for r in series],
                        [r["total_energy"] / 1e6 for r in series],
                        "o-", color=colour, markersize=4.5, label=f"{group_key}={group}")

            marked_value = energy_at(rows, x_key, group_key, *marked_point)
            reference_value = energy_at(rows, x_key, group_key, *reference_point)
            if reference_value is not None:
                ax.axhline(reference_value / 1e6, color="gray", linestyle="dashed",
                           linewidth=1.1, zorder=0,
                           label=f"reference ({x_key}={reference_point[0]}, "
                                 f"{group_key}={reference_point[1]})")
            if marked_value is not None and reference_value is not None:
                difference = 100 * (marked_value - reference_value) / reference_value
                ax.plot(marked_point[0], marked_value / 1e6, "o", markersize=15,
                        markerfacecolor="none", markeredgecolor="crimson",
                        markeredgewidth=2.0, zorder=5,
                        label=f"used here ({x_key}={marked_point[0]}, "
                              f"{group_key}={marked_point[1]})")
                low, high = ax.get_ylim()
                near_top = (marked_value / 1e6 - low) / (high - low) > 0.6
                ax.annotate(
                    f"{difference:+.2f}% vs {x_key}={reference_point[0]}, "
                    f"{group_key}={reference_point[1]}",
                    xy=(marked_point[0], marked_value / 1e6),
                    xytext=(14, -26) if near_top else (14, 16),
                    textcoords="offset points", fontsize=9.5, color="crimson",
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="crimson", linewidth=0.8),
                )
                print(f"  bore energy, {title}: {difference:+.3f}% vs the finest grid")

            ax.set_xscale("log")
            ax.set_xticks(x_values)
            ax.set_xticklabels([str(x) for x in x_values])
            ax.minorticks_off()
            ax.grid(True, which="major", alpha=0.3)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("stored bore energy [MJ]")
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=8, loc="best")

        fig.tight_layout()
        outputs += save(fig, f"energy_random_shape_convergence_seed{seed}")

    if not outputs:
        raise MissingData("no convergence sweeps found")
    return outputs, missing


def jc_solver_comparison():
    """J_critical maps, solver and platform against objective.

    The colour scale is logarithmic: J_c spans the operating point (~2e7)
    to ~2.9e8 at the outer edge, and on a linear ramp the whole winding
    renders near-black with only the thin high-J_c streak resolved,
    hiding the structure near the operating point that matters.
    """
    columns = [("COBYQA_cpu", "COBYQA (CPU)"), ("COBYQA_gpu", "COBYQA (GPU)"),
               ("trustconstr_gpu", "trust-constr (GPU)")]
    rows = [("average", "avg"), ("minimum", "min")]
    sources = {(row, tag): f"della_jc_{tag}_{row}_{RESOLUTION}.npz"
               for row, _ in rows for tag, _ in columns}

    def title(entry, row_key, row_label, column_label):
        grid = entry["J_grid"]
        return f"{column_label}, {row_label}\nmean={grid.mean():.2e}, min={grid.min():.2e}"

    return comparison_grid(
        rows, columns, sources, "J_grid",
        stem=f"jc_solver_comparison_{RESOLUTION}",
        colour_label="$J_{critical}$ [$10^7$ A/m$^2$], log scale",
        title_for=title,
    )


def trustconstr_jc_histogram():
    """Per-source-point J_critical distribution for the trust-constr GPU
    runs, average-objective coil against minimum-objective coil.

    Reads the same J_grid arrays the maps are drawn from, so the
    histogram is by construction the same distribution. The dashed line
    marks each coil's self-consistent operating current density, which
    equals that distribution's own minimum: the solve defines the
    operating current as the lowest J_critical anywhere on the winding.
    """
    runs = [("average", "maximize_average_J_critical", "tab:blue"),
            ("minimum", "maximize_minimum_J_critical", "tab:orange")]
    distributions, missing = {}, []
    for key, label, colour in runs:
        name = f"della_jc_trustconstr_gpu_{key}_{RESOLUTION}.npz"
        entry = load_archive(name)
        if entry is None:
            missing.append(name)
            continue
        distributions[key] = {
            "values": np.asarray(entry["J_grid"]).ravel(),
            "current_density": float(entry["current_density"]),
            "label": label, "colour": colour,
        }
    if not distributions:
        raise MissingData("no trust-constr J_critical runs found")

    all_values = np.concatenate([d["values"] for d in distributions.values()])
    low, high = float(all_values.min()), float(all_values.max())
    bins = np.logspace(np.log10(low), np.log10(high), 45)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for entry in distributions.values():
        values = entry["values"]
        ax.hist(values, bins=bins, alpha=0.55, color=entry["colour"],
                label=f"{entry['label']}\nmean={values.mean():.3e}, min={values.min():.3e}")
        ax.axvline(entry["current_density"], color=entry["colour"],
                   linestyle="dashed", linewidth=1.6)

    ax.set_xscale("log")
    ticks = [t for t in log_ticks(low, high) if low <= t <= high]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t / 1e7:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("$J_{critical}$ at each source point [$10^7$ A/m$^2$], log scale")
    ax.set_ylabel("number of source points")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(True, which="major", alpha=0.25)
    fig.tight_layout()
    return save(fig, f"trustconstr_gpu_J_critical_histogram_{RESOLUTION}"), missing


def lorentz_normalized():
    """Lorentz force density for the current-normalized objective F/J^2,
    the unoptimized coil against both solvers.

    The unoptimized coil takes the first column in place of a third
    solver run: a solver-versus-solver grid answers "do the solvers
    agree", which the numbers already settle, whereas the figure has to
    show what the optimization did to the winding, which needs the shape
    it started from on the same colour scale and axes.

    trust-constr's entries come from the runs with the inner-radius floor
    relaxed by 1e-7 m and an initial trust radius of 2.0. That constraint
    is active with zero slack at every feasible shape, which an
    interior-point method cannot work with, and the 0.01 radius inherited
    from the unnormalized objectives at ~1e8 left it returning a shape
    worse than the unoptimized coil. COBYQA never sees the radius
    setting, so its runs are unaffected.
    """
    columns = [("baseline", "unoptimized"), ("COBYQA_gpu", "COBYQA (GPU)"),
               ("trustconstr_gpu", "trust-constr (GPU)")]
    rows = [("average", "avg"), ("maximum", "max")]

    def source(row_key, tag):
        if tag == "baseline":
            return f"lorentz_baseline_{RESOLUTION}.npz"
        suffix = "_slack1e-07_tr2" if tag == "trustconstr_gpu" else "_slack1e-07"
        return f"della_lorentznorm_{tag}_{row_key}_{RESOLUTION}{suffix}.npz"

    sources = {(row, tag): source(row, tag) for row, _ in rows for tag, _ in columns}

    def title(entry, row_key, row_label, column_label):
        grid = entry["F_grid"]
        current = float(entry["current_density"])
        statistic = grid.mean() if row_key == "average" else grid.max()
        return (f"{column_label}, {row_label}\n"
                f"mean={grid.mean():.2e}, max={grid.max():.2e}\n"
                f"$F/J^2$={statistic / current ** 2:.3e}, $J$={current:.3e}")

    return comparison_grid(
        rows, columns, sources, "F_grid",
        stem=f"lorentz_normalized_{RESOLUTION}",
        colour_label="Lorentz force density [$10^7$ N/m$^3$], log scale",
        title_for=title, extra_title_rows=2,
    )


def magnetic_energy_bore_map():
    """Magnetic energy density over the BORE, not the winding.

    The region 0 <= r <= r_inner(z) is what
    maximize_magnetic_energy_in_bore integrates, and the winding is drawn
    over it as a solid band with representative turn lines so the
    conductor producing the field is visible alongside the field. The
    scale is linear: the energy density spans about 5x, where the winding
    quantities span 15x to 90x and need a log ramp.
    """
    from plots import build_a_tall_demo_coil_for_histogram_comparison
    from plots import draw_the_winding

    runs = [("baseline", "unoptimized baseline"), ("COBYQA_gpu", "COBYQA (GPU)"),
            ("trustconstr_gpu", "trust-constr (GPU)")]
    data, missing = {}, []
    for tag, _ in runs:
        entry = load_archive(f"energy_map_{tag}.npz")
        if entry is None:
            missing.append(f"energy_map_{tag}.npz")
            continue
        data[tag] = entry
    if not data:
        raise MissingData("no energy maps found")

    baseline_total = float(data["baseline"]["total_energy"]) if "baseline" in data else None
    for tag, label in runs:
        entry = data.get(tag)
        if entry is None:
            continue
        total = float(entry["total_energy"])
        line = f"  {label:<22} total={total:.6e} J"
        if baseline_total and tag != "baseline":
            line += f"   {100 * (total - baseline_total) / baseline_total:+.2f}% vs baseline"
        print(line)

    vmin = min(float(e["energy_density"].min()) for e in data.values())
    vmax = max(float(e["energy_density"].max()) for e in data.values())
    base_coil = build_a_tall_demo_coil_for_histogram_comparison()
    coils = {tag: base_coil.model_copy(update={"curve_parameters": list(e["parameters"])})
             for tag, e in data.items()}

    r_limit, z_limit = panel_extent(data.values(), extra_radius=base_coil.width)
    panel_width_in, panel_height_in = panel_size(r_limit, z_limit)

    present = [(tag, label) for tag, label in runs if tag in data]
    fig, axes = plt.subplots(
        1, len(present),
        figsize=(len(present) * panel_width_in + 2.1, panel_height_in + 1.5),
        squeeze=False,
    )

    mesh = None
    for ax, (tag, label) in zip(axes[0], present, strict=True):
        entry = data[tag]
        ax.set_facecolor("#f0f0f0")
        for sign in (1, -1):
            mesh = ax.pcolormesh(
                sign * entry["r_edges"], entry["z_edges"],
                entry["energy_density"] / 1e6, shading="flat", cmap="inferno",
                vmin=vmin / 1e6, vmax=vmax / 1e6, antialiased=False,
            )
        draw_the_winding(ax, coils[tag])
        ax.set_xlim(-r_limit, r_limit)
        ax.set_ylim(-z_limit, z_limit)
        ax.set_aspect("equal")
        ax.set_xlabel("r [m]")
        ax.set_title(f"{label}\nstored energy = {float(entry['total_energy']) / 1e6:.3f} MJ",
                     fontsize=9.5)
    axes[0][0].set_ylabel("z [m]")

    fig.subplots_adjust(left=0.07, right=0.87, bottom=0.09, top=0.88, wspace=0.20)
    colorbar = fig.colorbar(mesh, cax=fig.add_axes([0.895, 0.14, 0.020, 0.70]))
    colorbar.set_label("magnetic energy density [MJ/m$^3$]")
    return save(fig, "magnetic_energy_bore_map_16x200"), missing


def label_points_without_overlap(ax, xs, ys, texts, fontsize=7, color="0.25",
                                 marker_radius=5.0):
    """Annotate each (x, y) with its text, choosing an offset that collides
    with neither another label nor a data marker.

    Fixed offsets, even alternating ones, cannot separate points that sit on
    top of each other: in the Pareto inset the designs found at w = 0.1 and
    w = 0.2 differ by 1.9e-5 in h, so their labels land in the same place
    whatever constant offset they are given. This tries a ring of candidate
    positions per point and keeps the first that is clear, measuring real
    text extents rather than guessing.

    Three details are load-bearing and easy to "simplify" back into bugs.

    The candidates run out to about 30 pt in growing shells. Twelve points
    in an inset this small cannot all be placed in the first shell, and
    stopping there leaves labels unplaceable.

    Being clear of every other label and marker is NOT sufficient. A label
    whose centre is nearer to some other marker than to its own reads as
    that marker's label, which is how the w = 0.5 label first ended up
    apparently belonging to the w = 0.1 point. Hence the ownership test.

    When no candidate works the default position is kept and a note is
    printed, rather than the label being dropped: a crowded label is a
    cosmetic problem, a silently missing one is a wrong figure.

    Must be called AFTER the final tight_layout: the axes position, and so
    the display coordinates every extent is measured in, changes before then.
    """
    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    candidates = []
    for radius in (7, 12, 18, 25, 32):
        for angle in range(0, 360, 30):
            radians = math.radians(angle)
            candidates.append((radius * math.cos(radians),
                               radius * math.sin(radians)))

    occupied = [
        matplotlib.transforms.Bbox.from_bounds(
            px - marker_radius, py - marker_radius,
            2 * marker_radius, 2 * marker_radius)
        for px, py in (ax.transData.transform((x, y)) for x, y in zip(xs, ys))
    ]

    points = [ax.transData.transform((x, y)) for x, y in zip(xs, ys)]

    for owner, (x, y, text) in enumerate(zip(xs, ys, texts)):
        placed = None
        for dx, dy in candidates:
            annotation = ax.annotate(
                text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=fontsize, color=color,
                ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
                va="bottom" if dy > 0 else ("top" if dy < 0 else "center"))
            box = annotation.get_window_extent(renderer=renderer).expanded(1.1, 1.2)
            if not any(box.overlaps(other) for other in occupied):
                centre = ((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)
                nearest = min(
                    range(len(points)),
                    key=lambda i: (points[i][0] - centre[0]) ** 2
                    + (points[i][1] - centre[1]) ** 2)
                if nearest == owner:
                    placed = (annotation, box)
                    break
            annotation.remove()
        if placed is None:
            annotation = ax.annotate(
                text, xy=(x, y), xytext=(5, 4), textcoords="offset points",
                fontsize=fontsize, color=color)
            placed = (annotation,
                      annotation.get_window_extent(renderer=renderer))
            print(f"    note: no clear position for label {text!r}")
        occupied.append(placed[1])


def pareto_energy_homogeneity():
    """Stored bore energy against field inhomogeneity, swept by weight.

    The scalarization is f = -w U/U_ref + (1-w) h/h_ref, both terms
    divided by their unoptimized value so the weight means what it looks
    like it means: U is ~1e7 J and h is ~0.12, so an unnormalized w=0.5
    would be, in effect, pure energy.

    The front is steep, not concave, through the stretch a coarse weight
    ladder appears to skip. Differencing it and inverting the support
    relation w = 1/(1+s), s = du/dx, recovers each point's own weight to
    within 0.0054 over 0.74 <= w <= 0.95, so every design shown is
    selected by some weight and a coarse ladder merely steps over the
    transition, which the corrected volume weighting moved to
    0.73 -> 0.74.

    The low-weight end is a different matter and is drawn as one point.
    Solved the way the rest of the paper solves things, ten random
    restarts per weight and the best feasible result taken, the seven
    weights w <= 0.6 return seven different designs spanning 3.12% in h,
    with only 3 of 6 steps ordered. That spread dwarfs any real trade-off
    over the same range, and it is multimodality rather than convergence
    error: runs sharing a starting pool reproduce to about 1e-5, while
    runs with different pools differ by 0.2 to 0.5%.

    Three choices in the code below are deliberate.

    LOW_MAX = 0.6 is the collapse threshold and INSET_MAX = 0.73 is how
    far the inset reaches. They differ on purpose. Only w <= 0.6 comes
    from the ten-random-restart study and only those designs are
    indistinguishable; 0.65 to 0.73 come from the fine sweep and climb in
    order, so they stay as their own points on the main panel and appear
    in the inset as squares. Showing them is what lets a reader see that
    the low end is unresolved while the front just above it is real.

    The representative is chosen from w <= LOW_MAX alone, as the design
    minimizing the most of THOSE weights' objectives, so the single point
    stands for exactly what its label claims; the inset's extra points are
    displayed but not folded into the choice. Two designs win here, 0.016%
    apart in h against a 3.12% spread, so which one is picked is
    arbitrary at the scale of the figure. The count and spread of the
    contenders are printed on every run, because that is the number which
    decides whether collapsing is legitimate at all.

    The collapsed marker is coloured by the weight whose run found the
    design, not by LOW_MAX, which would invent a weight it does not have
    and would not match the same design's marker in the inset. It carries
    the star in both panels for the same reason: colour alone no longer
    identifies which of the inset's markers is the one sitting on the
    front.
    """
    path = DATA / "pareto_energy_homogeneity_COBYQA_gpu_16x200.json"
    if not path.exists():
        raise MissingData(path.name)
    payload = json.loads(path.read_text())
    U_ref = payload["baseline_energy"]
    h_ref = payload["baseline_homogeneity"]

    fine_path = DATA / "pareto_weighted_fine_COBYQA_gpu_16x200.json"
    front = json.loads(fine_path.read_text())["front"] if fine_path.exists() \
        else payload["front"]

    LOW_MAX = 0.6
    INSET_MAX = 0.73
    low_path = DATA / "pareto_low_random_COBYQA_gpu_16x200.json"
    low = json.loads(low_path.read_text())["front"] if low_path.exists() else []
    low = [{**r, "from_fine": False} for r in low]
    low += [{**r, "from_fine": True} for r in front
            if LOW_MAX < r["weight"] <= INSET_MAX]

    collapsed = None
    if low:
        pool = [r for r in low if r["weight"] <= LOW_MAX]
        scored = [(r["energy"] / U_ref, r["homogeneity"] / h_ref) for r in pool]
        wins = [0] * len(pool)
        for row in pool:
            w = row["weight"]
            wins[int(np.argmin([-w * u + (1 - w) * x for u, x in scored]))] += 1
        best = max(range(len(pool)), key=lambda i: wins[i])
        contenders = [i for i, n in enumerate(wins) if n]
        if len(contenders) > 1:
            gap_h = max(abs(pool[i]["homogeneity"] - pool[best]["homogeneity"])
                        for i in contenders) / pool[best]["homogeneity"] * 100
            gap_U = max(abs(pool[i]["energy"] - pool[best]["energy"])
                        for i in contenders) / pool[best]["energy"] * 100
            print(f"  {len(contenders)} designs win among w <= {LOW_MAX} "
                  f"(wins {[wins[i] for i in contenders]}); they differ by "
                  f"{gap_h:.3f}% in h and {gap_U:.3f}% in U")
        span_h = (max(r["homogeneity"] for r in pool)
                  - min(r["homogeneity"] for r in pool)) / min(
                      r["homogeneity"] for r in pool) * 100
        print(f"  the single point stands for {len(pool)} weights spanning "
              f"{span_h:.2f}% in h; the inset also shows "
              f"{len(low) - len(pool)} sweep weights up to {INSET_MAX:g}")
        collapsed = {**pool[best], "weight": LOW_MAX,
                     "source_weight": pool[best]["weight"]}

    upper = [r for r in front if r["weight"] > LOW_MAX]
    up_scored = [(r["energy"] / U_ref, r["homogeneity"] / h_ref) for r in upper]
    resolved, reassigned = [], 0
    for row in upper:
        w = row["weight"]
        pick = int(np.argmin([-w * u + (1 - w) * x for u, x in up_scored]))
        if upper[pick] is not row:
            reassigned += 1
            row = {**upper[pick], "weight": w}
        resolved.append(row)
    if reassigned:
        print(f"  reassigned {reassigned} weights above {LOW_MAX} to a better design")
    plotted = ([collapsed] if collapsed else []) + resolved

    weights = np.array([row["weight"] for row in plotted])
    colours = np.array([row.get("source_weight", row["weight"]) for row in plotted])
    energy = np.array([row["energy"] for row in plotted]) / 1e6
    homogeneity = np.array([row["homogeneity"] for row in plotted])
    dominated = np.array([
        any(e > energy[i] and h < homogeneity[i]
            for e, h in zip(energy, homogeneity, strict=True))
        for i in range(len(plotted))
    ])

    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    order = np.argsort(homogeneity[~dominated])
    ax.plot(homogeneity[~dominated][order], energy[~dominated][order],
            "-", color="0.65", linewidth=1.3, zorder=1)
    scatter = ax.scatter(homogeneity[~dominated], energy[~dominated],
                         c=colours[~dominated], cmap="viridis", s=95,
                         vmin=0.0, vmax=1.0,
                         edgecolor="black", linewidth=0.8, zorder=3)
    if collapsed is not None:
        ax.scatter([collapsed["homogeneity"]], [collapsed["energy"] / 1e6],
                   marker="*", s=260, facecolor="none", edgecolor="crimson",
                   linewidth=1.4, zorder=5)
    for row, e, h in zip(plotted, energy, homogeneity, strict=True):
        label = (f"$w\\leq{LOW_MAX:g}$" if row is collapsed
                 else (f"$w={row['weight']:g}$"
                       if row["weight"] in (0.7, 0.8, 1.0) else None))
        if label:
            offset = (4, -15) if row is collapsed else (
                6, -12 if row["weight"] >= 0.8 else 6)
            ax.annotate(label, xy=(h, e), xytext=offset,
                        textcoords="offset points", fontsize=9)

    if low:
        inset = ax.inset_axes([0.52, 0.10, 0.45, 0.42])
        low_h = np.array([r["homogeneity"] for r in low])
        low_U = np.array([r["energy"] for r in low]) / 1e6
        low_w = np.array([r["weight"] for r in low])
        from_fine = np.array([r.get("from_fine", False) for r in low])
        inset.scatter(low_h[~from_fine], low_U[~from_fine], c=low_w[~from_fine],
                      cmap="viridis", vmin=0.0, vmax=1.0, s=55,
                      edgecolor="black", linewidth=0.6, zorder=3)
        if from_fine.any():
            inset.scatter(low_h[from_fine], low_U[from_fine], c=low_w[from_fine],
                          cmap="viridis", vmin=0.0, vmax=1.0, s=48, marker="s",
                          edgecolor="0.35", linewidth=0.9, zorder=3)
        inset.scatter([collapsed["homogeneity"]], [collapsed["energy"] / 1e6],
                      marker="*", s=220, facecolor="none", edgecolor="crimson",
                      linewidth=1.4, zorder=4)
        inset.set_title(f"$w\\leq{INSET_MAX:g}$", fontsize=8)
        inset.tick_params(labelsize=7)
        inset.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        inset.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        inset.ticklabel_format(axis="x", useOffset=False)
        inset.margins(x=0.18, y=0.16)
        inset.grid(alpha=0.25)

    ax.set_xlabel("field inhomogeneity $h = \\sigma_B / \\langle B \\rangle$ over the bore")
    ax.set_ylabel("stored bore energy $U$ [MJ]")
    ax.grid(alpha=0.3)
    fig.colorbar(scatter, ax=ax, label="weight $w$ on the energy term")
    fig.tight_layout()
    if low:
        label_points_without_overlap(
            inset, low_h, low_U, [f"{w:g}" for w in low_w])

    print(f"  baseline U={U_ref/1e6:.4f} MJ  h={h_ref:.6f}")
    if low:
        span_h = (low_h.max() - low_h.min()) / low_h.min() * 100
        span_U = (low_U.max() - low_U.min()) / low_U.min() * 100
        print(f"  low end: {len(low)} weights, scatter {span_h:.2f}% in h, "
              f"{span_U:.2f}% in U; collapsed to h={collapsed['homogeneity']:.8f} "
              f"U={collapsed['energy']/1e6:.4f} MJ (from w={low[0]['weight']:.2f}..)")
    for row, is_dominated in zip(plotted, dominated, strict=True):
        print(f"  w={row['weight']:.2f}: U={row['energy']/1e6:.4f} MJ "
              f"({row['energy_vs_baseline']:+6.2f}%)  h={row['homogeneity']:.6f} "
              f"({row['homogeneity_vs_baseline']:+6.2f}%)"
              f"{'  dominated' if is_dominated else ''}")
    return save(fig, "pareto_energy_homogeneity_16x200"), []


def bc_trajectories():
    """COBYQA and trust-constr trajectories in the (B, C) slice, under the
    narrow constraints at the resolution the paper reports.

    Replaces an earlier figure computed at (8, 100) with only the bore
    pinned and no inner-radius floor or outer-radius cap, so that one did
    not show the narrow configuration at all and its hit rates do not
    carry over.

    Nelder-Mead is shown alongside them because its failure is the
    argument for excluding it. scipy's implementation accepts bounds but
    no nonlinear constraints, so it cannot be given the geometry: from ten
    starting points that were all buildable it finished with zero
    buildable endpoints, while reaching an objective value 7.6% above
    anything a buildable coil attains. It is not failing to optimize; it
    is succeeding on shapes that cannot be made.

    Only 7.9% of the plotted box satisfies the constraints, since with the
    bore pinned r(z) = A + Bz^2 + Cz^4 dips below the floor for any
    B < 0. The infeasible region is hatched, and the starting points were
    rejection-sampled from the feasible part so both solvers begin
    somewhere valid.
    """
    archive = load_archive("bc_trajectories_narrow_16x200.npz")
    path = DATA / "bc_trajectories_narrow_16x200.json"
    if archive is None or not path.exists():
        raise MissingData("bc_trajectories_narrow_16x200")
    payload = json.loads(path.read_text())

    b_values, c_values = archive["b_values"], archive["c_values"]
    landscape, feasible = archive["landscape"], archive["feasible"]
    best = payload["best_feasible"]
    shown = ["Nelder-Mead", "COBYQA", "trust-constr"]


    fig, axes = plt.subplots(2, 2, figsize=(11.0, 10.4), squeeze=False)
    colours = [plt.cm.tab10(i % 10) for i in range(len(payload["solvers"]["COBYQA"]["values"]))]

    mesh = None
    panel_axes = [axes[0][0], axes[0][1], axes[1][0]]
    for ax, solver in zip(panel_axes, shown, strict=True):
        entry = payload["solvers"][solver]
        mesh = ax.pcolormesh(c_values, b_values, landscape, shading="auto", cmap="viridis")
        ax.contour(c_values, b_values, landscape, levels=10, colors="white",
                   linewidths=0.3, alpha=0.45)
        ax.contour(c_values, b_values, feasible.astype(float), levels=[0.5],
                   colors="white", linewidths=2.4)
        ax.contour(c_values, b_values, feasible.astype(float), levels=[0.5],
                   colors="black", linewidths=0.9, linestyles="dashed")

        for trajectory, value, is_feasible, colour in zip(
            entry["trajectories"], entry["values"], entry["feasible"], colours, strict=True
        ):
            points = np.asarray(trajectory)
            ax.plot(points[:, 1], points[:, 0], "-", color=colour, linewidth=1.5, alpha=0.9)
            ax.plot(points[0, 1], points[0, 0], marker="o", markersize=8, color=colour,
                    markeredgecolor="black", zorder=5)
            hit = is_feasible and value >= best * 0.98
            ax.plot(points[-1, 1], points[-1, 0], marker="*" if hit else "X",
                    markersize=18 if hit else 12, color=colour,
                    markeredgecolor="black", markeredgewidth=1.1, zorder=6)

        buildable = sum(entry["feasible"])
        hits = sum(1 for v, ok in zip(entry["values"], entry["feasible"], strict=True)
                   if ok and v >= best * 0.98)
        summary = (f"{buildable}/{len(entry['values'])} buildable"
                   if buildable == 0 else
                   f"{hits}/{len(entry['values'])} reach the ridge")
        ax.set_title(f"{solver}  ({summary})", fontsize=11)
        ax.set_xlabel("$C$ ($z^4$ coefficient)")
        ax.set_ylabel("$B$ (leading curvature coefficient)")
        ax.set_xlim(c_values.min(), c_values.max())
        ax.set_ylim(b_values.min(), b_values.max())
        ax.set_aspect("equal")

    zoom = axes[1][1]
    rows, columns = np.where(feasible)
    b_low, b_high = b_values[rows.min()], b_values[rows.max()]
    c_low, c_high = c_values[columns.min()], c_values[columns.max()]
    inside = np.ma.masked_where(~feasible, landscape)
    zoom_mesh = zoom.pcolormesh(c_values, b_values, inside, shading="auto",
                                cmap="viridis")
    zoom.contour(c_values, b_values, inside, levels=12, colors="white",
                 linewidths=0.4, alpha=0.6)
    zoom.contour(c_values, b_values, feasible.astype(float), levels=[0.5],
                 colors="black", linewidths=1.2)

    for solver, marker in (("COBYQA", "*"), ("trust-constr", "P")):
        entry = payload["solvers"][solver]
        points = np.array([t[-1] for t, ok in
                           zip(entry["trajectories"], entry["feasible"], strict=True) if ok])
        if len(points):
            zoom.plot(points[:, 1], points[:, 0], marker, markersize=15 if marker == "*" else 10,
                      linestyle="none", markeredgecolor="black", markeredgewidth=0.9,
                      label=solver, zorder=6,
                      color="gold" if marker == "*" else "deepskyblue")
    zoom.set_xlim(c_low - 0.1, c_high + 0.1)
    zoom.set_ylim(b_low - 0.1, b_high + 0.1)
    zoom.set_aspect("equal")
    zoom.set_xlabel("$C$ ($z^4$ coefficient)")
    zoom.set_ylabel("$B$ (leading curvature coefficient)")
    zoom.set_title("buildable region only (7.9% of the plane)", fontsize=11)
    zoom.legend(fontsize=9, loc="lower left", framealpha=0.9)

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.115, top=0.955,
                        wspace=0.13, hspace=0.20)
    colorbar = fig.colorbar(mesh, cax=fig.add_axes([0.07, 0.045, 0.915, 0.020]),
                            orientation="horizontal")
    colorbar.set_label("average $J_{critical}$ [A/m$^2$] (upper-left, upper-right and "
                       "lower-left panels)")
    zoom_bar = fig.colorbar(zoom_mesh, ax=zoom, fraction=0.046, pad=0.02)
    zoom_bar.set_label("$J_{critical}$, buildable range", fontsize=9)

    print(f"  best feasible J_c = {best:.6e}; {100*feasible.mean():.1f}% of the box is feasible")
    for solver, entry in payload["solvers"].items():
        hits = sum(1 for v, ok in zip(entry["values"], entry["feasible"], strict=True)
                   if ok and v >= best * 0.98)
        note = "" if sum(entry["feasible"]) else "  (no buildable endpoint)"
        print(f"  {solver:>13}: {sum(entry['feasible'])}/10 feasible, "
              f"{hits}/10 within 2%, {entry['seconds']:.0f}s{note}")
    return save(fig, "bc_trajectories_narrow_16x200"), []



def restart_miss_rate_cobyqa():
    """Why the multi-start procedure uses ten restarts, in two panels that
    answer the question on two different problems.

    LEFT: the miss rate against restart count for COBYQA on the
    two-parameter (B, C) slice under the narrow constraints, estimated by
    resampling a pool of 400 independent single-start runs. It is close to
    zero everywhere: a single run already lands within 2% of the best
    98.8% of the time. This replaces an earlier curve computed with
    Nelder-Mead on an interpolation of an unconstrained landscape, which
    reported 68% at one restart, and it does not support using ten.

    RIGHT: which restart actually produced the reported result, across
    every multi-start run in this work. The full problem has five curve
    parameters, not two, and there the picture reverses: the winner is
    spread across eight of the eleven starting points, and the coil's own
    unperturbed shape never wins. A single run from the natural starting
    point would have found the best result in none of them.

    The two panels together are the argument: restarts are not needed
    because the optimizer is unreliable, which on a tractable problem it
    is not, but because the search space the results actually live in is
    multimodal.
    """
    archive = load_archive("restart_miss_rate_cobyqa.npz")
    if archive is None:
        raise MissingData("restart_miss_rate_cobyqa.npz")

    winners, total = {}, 0
    for path in sorted(DATA.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(payload, dict) and "winning_restart" in payload:
            winners[payload["winning_restart"]] = winners.get(payload["winning_restart"], 0) + 1
            total += 1

    fig, axes = plt.subplots(1, 1, figsize=(7.2, 5.0), squeeze=False)
    axes = axes[0]

    curves = [
        ("restart_miss_rate_cobyqa.npz",
         "2 parameters, average $J_c$", "tab:green", "o"),
        ("restart_miss_rate_5d_average.npz",
         "5 parameters, average $J_c$", "tab:blue", "s"),
        ("restart_miss_rate_5d_minimum.npz",
         "5 parameters, minimum $J_c$", "crimson", "D"),
    ]
    highest = 0.0
    for filename, label, colour, marker in curves:
        entry = load_archive(filename)
        if entry is None:
            continue
        N_values = entry["N_values"]
        rates = entry["miss_rates"] * 100
        highest = max(highest, float(rates.max()))
        axes[0].plot(N_values, rates, marker=marker, linestyle="-", color=colour,
                     linewidth=1.8, markersize=6, label=label)
    axes[0].set_xlabel("number of restarts $N$")
    axes[0].set_ylabel("miss rate (%)")
    axes[0].set_xticks(archive["N_values"])
    axes[0].set_ylim(-0.02 * highest, highest * 1.12)
    axes[0].grid(alpha=0.3)
    axes[0].axvline(10, color="0.5", linestyle="dotted", linewidth=1.2)
    axes[0].annotate("$N=10$ used\nthroughout", xy=(10, highest * 0.55),
                     xytext=(-6, 0), textcoords="offset points", fontsize=9,
                     color="0.35", ha="right")
    axes[0].legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    for filename, label, _colour, _marker in curves:
        entry = load_archive(filename)
        if entry is not None:
            rates = entry["miss_rates"] * 100
            print(f"  {label:<30}: N=1 {rates[0]:5.2f}%, N=4 {rates[3]:5.2f}%, "
                  f"N=10 {rates[7]:5.2f}%")
    print(f"  5-parameter runs: {total} total, winner spread over "
          f"{len(winners)} of 11 starting points, "
          f"unjittered original won {winners.get('unjittered_original', 0)}")
    return save(fig, "restart_miss_rate"), []



def lorentz_normalized_comparison():
    """Baseline against the current-normalized optimum, two columns.

    The companion to lorentz_normalized(), which compares solvers across
    three columns. This one drops the solver axis entirely and puts the
    unoptimized coil beside a single optimized design, so the figure
    answers "what did the optimization do to the winding" rather than "do
    the solvers agree". Both are in the paper because they answer
    different questions on the same data.

    Sources are the runs with the relaxed inner-radius floor, matching
    every other reported Lorentz number; the earlier standalone script
    read the pre-fix archives, which differ in the fourth significant
    figure on the maximum objective and would put this figure out of step
    with the tables.

    Rendering follows comparison_grid's conventions rather than
    reinventing them: shading="flat" against explicit cell edges,
    logarithmic colour scale with an explicit tick progression, both
    mirrored halves drawn, and one set of axis limits across all panels.
    """
    columns = [("baseline", "unoptimized"), ("norm", "minimize $F/J^2$")]
    rows = [("average", "avg"), ("maximum", "max")]

    def source(row_key, tag):
        if tag == "baseline":
            return f"lorentz_baseline_{RESOLUTION}.npz"
        return f"della_lorentznorm_COBYQA_gpu_{row_key}_{RESOLUTION}_slack1e-07.npz"

    sources = {(row, tag): source(row, tag) for row, _ in rows for tag, _ in columns}

    def title(entry, row_key, row_label, column_label):
        grid = entry["F_grid"]
        current = float(entry["current_density"])
        statistic = grid.mean() if row_key == "average" else grid.max()
        return (f"{column_label}, {row_label}\n"
                f"mean={grid.mean():.2e}, max={grid.max():.2e}\n"
                f"$F/J^2$={statistic / current ** 2:.3e}, $J$={current:.3e}")

    return comparison_grid(
        rows, columns, sources, "F_grid",
        stem=f"lorentz_normalized_comparison_{RESOLUTION}",
        colour_label="Lorentz force density [$10^7$ N/m$^3$], log scale",
        title_for=title, extra_title_rows=2,
    )


FIGURES = [
    ("collapsed_coil_schematic", collapsed_coil_schematic),
    ("bc_trajectories", bc_trajectories),
    ("restart_miss_rate", restart_miss_rate_cobyqa),
    ("random_shape_convergence", random_shape_convergence),
    ("jc_solver_comparison", jc_solver_comparison),
    ("trustconstr_jc_histogram", trustconstr_jc_histogram),
    ("lorentz_normalized", lorentz_normalized),
    ("lorentz_normalized_comparison", lorentz_normalized_comparison),
    ("magnetic_energy_bore_map", magnetic_energy_bore_map),
    ("pareto_energy_homogeneity", pareto_energy_homogeneity),
]

PLANNED: list[tuple[str, str]] = []

parser = argparse.ArgumentParser(description="Draw the paper's figures.")
parser.add_argument("--list", action="store_true", help="show the manifest without drawing")
parser.add_argument("--only", metavar="PATTERN", help="restrict to matching figures")
parser.add_argument("--copy", metavar="DIR", help="copy the PDFs into DIR")
arguments = parser.parse_args()

selected = [(name, function) for name, function in FIGURES
            if arguments.only is None or arguments.only in name]
if not selected:
    raise SystemExit(f"no figure matches {arguments.only!r}")

if arguments.list:
    for name, function in selected:
        summary = (function.__doc__ or "").strip().splitlines()[0]
        print(f"{name:<34} {summary}")
    print("\nplanned, no function yet:")
    for name, description in PLANNED:
        print(f"  {name:<34} {description}")
    raise SystemExit(0)

print(f"drawing {len(selected)} of {len(FIGURES)} figures\n")
produced, failures, incomplete = [], [], []
for name, function in selected:
    started = time.time()
    try:
        outputs, missing = function()
    except MissingData as error:
        print(f"[skip] {name:<34} no data: {error}")
        failures.append((name, str(error)))
        continue
    produced += outputs
    elapsed = time.time() - started
    if missing:
        incomplete.append((name, missing))
        print(f"[warn] {name:<34} {elapsed:5.1f}s  INCOMPLETE, missing {len(missing)}: {missing[0]}")
    else:
        print(f"[ok  ] {name:<34} {elapsed:5.1f}s  {', '.join(outputs)}")

if arguments.copy:
    destination = pathlib.Path(arguments.copy).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    for name in produced:
        shutil.copy2(OUTPUT / name, destination / name)
    print(f"\ncopied {len(produced)} PDFs into {destination}")

if incomplete:
    print("\nbuilt but INCOMPLETE (a panel is a placeholder, not data):")
    for name, missing in incomplete:
        print(f"  {name}: {', '.join(missing)}")

if PLANNED:
    print("\nplanned, no function yet:")
    for name, description in PLANNED:
        print(f"  {name:<34} {description}")

if failures:
    print(f"\n{len(failures)} could not be drawn:")
    for name, message in failures:
        print(f"  {name}: {message}")
    raise SystemExit(1)
print(f"\nall {len(selected)} drawn")

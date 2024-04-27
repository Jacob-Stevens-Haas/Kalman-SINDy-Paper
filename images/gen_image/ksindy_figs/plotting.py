from dataclasses import dataclass
from pathlib import Path
from types import EllipsisType as ellipsis
from typing import Any, NewType, Optional, Sequence, Callable, cast
from warnings import warn

import gen_experiments.plotting as genplot
import matplotlib as mpl
import matplotlib.pyplot as plt
import mitosis
import numpy as np
from gen_experiments.gridsearch import find_gridpoints
from gen_experiments.gridsearch.typing import (
    KeepAxisSpec,
    GridLocator,
    GridsearchResult,
    GridsearchResultDetails,
    SavedGridPoint,
)
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec, SubplotSpec

from ._typing import Float1D, Float2D
from .data import TRIAL_DATA, load_mitosis_5

CMAP = mpl.color_sequences["tab10"]
CTRUE = CMAP[0]
CMEAS = CMAP[1]
CEST = CMAP[2]
CGRAY = (0.85, 0.85, 0.85)
plt.rcParams["text.usetex"] = True


SelectSome = ellipsis | tuple[int | slice, int]
SelectMatch = dict[str, int | str]
NamedMatch = tuple[str, SelectMatch]
SelectStatement = tuple[tuple[SelectMatch, ...], tuple[SelectSome, ...]]
NamedSelect = tuple[tuple[NamedMatch, ...], tuple[SelectSome, ...]]
ExpKey = NewType("ExpKey", str)


if hasattr(mitosis, "__version__"):
    loadfunc = load_mitosis_5
else:
    loadfunc = mitosis.load_trial_data


@dataclass
class PlotData:
    """Container for 1-D plottable data"""

    t: Float1D
    dt: Float1D
    x: Float1D
    dx: Float1D
    x_dot: Float1D
    z: Float1D
    dz: Float1D
    x_hat: Float1D
    x_dot_hat: Float1D
    theta: Float2D
    coef: Float1D
    amp: float


def data_plot(
    t: Float1D, x: Float1D, z: Float1D, *, ylims: tuple[float, float] | None = None
) -> None:
    ax = plt.figure().add_axes((0, 0, 1, 1))
    ax.plot(t, x, color=CTRUE, label="true")
    ax.plot(t, z, ".", color=CMEAS, label="measured")
    ax.legend()
    if ylims is not None:
        ax.set_ylim(*ylims)


def smoothing_plot(
    ax: Axes,
    t: Float1D,
    x: Float1D,
    z: Float1D,
    dt: Float1D,
    dz: Float1D,
    x_hat: Float1D,
    x_dot_hat: Float1D,
    *,
    amp: float = 1,
    ylims: tuple[float, float] | None = None,
    **q_props: Any,
) -> None:
    ax.plot(t, x, color=CTRUE, label=r"$x$")
    ax.plot(t, z, ".", color=CMEAS, label=r"$z$")
    ax.plot(t, x_hat, "--", color=CEST, label=r"$\hat x$")
    # ax.quiver(t, z, dt, dz, color=CMEAS, **q_props)
    xyuv = (t, x_hat, dt * amp, dt * x_dot_hat * amp)
    ax.quiver(*xyuv, label=r"$\hat{\dot x}$", color=CEST, **q_props)
    ax.legend()
    ax.set_xticks(())
    if ylims is not None:
        ax.set_ylim(*ylims)


def lib_plot(
    ax: Axes,
    t: Float1D,
    dt: Float1D,
    x_hat: Float1D,
    x_dot_hat: Float1D,
    funcs_theta: Float2D,
    *,
    amp: float = 1,
    ylims: tuple[float, float] | None = None,
    **q_props: Any,
) -> None:
    """Create plot of function library vectors and smoothed derivative vectors."""
    color = q_props.get("color")
    if color is None:
        ax.plot(t, x_hat, "--", color=CGRAY)
        curr_props = q_props | {"color": CEST}
    else:  # i.e. if we're the history plot, don't plot our history
        curr_props = q_props | {"color": color}
    xyuv = (t, x_hat, dt * amp, x_dot_hat * dt * amp)
    ax.quiver(*xyuv, **curr_props)
    for i, func in enumerate(funcs_theta):
        if color is None:
            curr_props = q_props | {"color": CMAP[i + 3]}
        ax.quiver(
            t,
            x_hat,
            dt * amp,
            func * dt * amp,
            label=rf"$\theta_{i}(\hat x)$" if not color else None,
            **curr_props,
        )
    if ylims is not None:
        ax.set_ylim(*ylims)
    if color is None:
        ax.legend()
    ax.set_xticks(())


def shared_ylim(*args: Float1D) -> tuple[float, float]:
    data_lim = (cast(float, np.min(args)), cast(float, np.max(args)))
    diff = data_lim[1] - data_lim[0]
    margin = diff * 0.1
    return data_lim[0] - margin, data_lim[1] + margin


def soln_plot(
    ax: Axes,
    t: Float1D,
    dt: Float1D,
    x_hat: Float1D,
    x_dot_hat: Float1D,
    theta: Float2D,
    coef: Float1D,
    *,
    amp: float = 1,
    ylims: tuple[float, float] | None = None,
    **q_props: Any,
) -> None:
    lib_plot(ax, t, dt, x_hat, x_dot_hat, theta, amp=amp, color=CGRAY, **q_props)
    # ax.plot(t, x_hat, ".", color=CEST)
    ax.quiver(t, x_hat, dt * amp, x_dot_hat * dt * amp, color=CEST, **q_props)
    old_height = x_hat
    old_left = t
    du = dt / np.count_nonzero(coef) * amp
    # v_start = x_hat + np.add.accumulate(coef.reshape((-1, 1)) * theta, axis=0)
    for i, (weight, func) in enumerate(zip(coef, theta, strict=True)):
        if weight == 0:
            continue
        dv = weight * func * amp * du
        ax.quiver(
            old_left,
            old_height,
            du,
            dv,
            color=CMAP[i + 3],
            label=rf"$\xi_{i}\theta_{i}(\hat x)$",
            **q_props,
        )
        old_height = old_height + dv
        old_left = old_left + du
    if ylims is not None:
        ax.set_ylim(*ylims)
    ax.legend()
    ax.set_xticks(())


def make_composite_fig1(pdat: PlotData, q_props: dict[str, Any]) -> None:
    ylims = shared_ylim(pdat.x, pdat.z, pdat.x_hat)
    # data_plot(pdat.t, pdat.x, pdat.z, ylims=ylims)
    fig = plt.figure(figsize=(5, 8))
    fig.suptitle("SINDy Steps")
    axes = cast(list[Axes], fig.subplots(3, 1))
    smoothing_plot(
        axes[0],
        pdat.t,
        pdat.x,
        pdat.z,
        pdat.dt,
        pdat.dz,
        pdat.x_hat,
        pdat.x_dot_hat,
        amp=pdat.amp,
        ylims=ylims,
        **q_props,
    )
    ax0_text = "Measurements smoothed to \nrecover estimated trajectory"
    tx_props = {"fontsize": "large", "bbox": {"fill": False}}
    text_posit = (0.05, 0.2)
    axes[0].text(*text_posit, ax0_text, transform=axes[0].transAxes, **tx_props)
    lib_plot(
        axes[1],
        pdat.t,
        pdat.dt,
        pdat.x_hat,
        pdat.x_dot_hat,
        pdat.theta,
        amp=pdat.amp,
        ylims=ylims,
        **q_props,
    )
    ax1_text = "Function library evaluated \nalong estimated trajectory"
    axes[1].text(*text_posit, ax1_text, transform=axes[1].transAxes, **tx_props)
    soln_plot(
        axes[2],
        pdat.t,
        pdat.dt,
        pdat.x_hat,
        pdat.x_dot_hat,
        pdat.theta,
        pdat.coef,
        ylims=ylims,
        amp=pdat.amp,
        **q_props,
    )
    ax2_text = "Candidate functions regressed \nagainst estimated derivative"
    axes[2].text(*text_posit, ax2_text, transform=axes[2].transAxes, **tx_props)
    # Make arrows from one axis overlap next
    axes[1].set_zorder(-1)
    axes[2].set_zorder(-2)
    arrow_shape = (0.15, 0.13, 0, -0.2)
    arrow_props = {
        "clip_on": False,
        "width": 0.05,
        "head_width": 0.1,
        "head_length": 0.1,
        "zorder": 100,
    }
    axes[0].arrow(*arrow_shape, transform=axes[0].transAxes, **arrow_props)
    axes[1].arrow(*arrow_shape, transform=axes[1].transAxes, **arrow_props)
    fig.subplots_adjust(hspace=0.1, top=0.95)


def _setup_summary_fig(
    cells_or_shape: int | tuple[int, int],
    *,
    fig_cell: Optional[tuple[Figure, SubplotSpec]] = None,
) -> tuple[Figure, GridSpec | GridSpecFromSubplotSpec]:
    """Create neatly laid-out arrangements for subplots

    Creates an evenly-spaced gridpsec to fit follow-on plots and a
    figure, if required.

    Args:
        cells_or_shape: number of grid elements to create, or a tuple of
            rows/cols
        fig_cell: parent grid cell within which to to build a nested
            gridspec
    Returns:
        a figure and gridspec if fig_cell is not provided, otherwise,
        None and a sub-gridspec
    """
    if isinstance(cells_or_shape, int):
        n_sub = cells_or_shape
        n_rows = max(n_sub // 3, (n_sub + 2) // 3)
        n_cols = min(n_sub, 3)
    else:
        n_rows, n_cols = cells_or_shape
    figsize = (3 * n_cols, 3 * n_rows)
    if fig_cell is None:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(n_rows, n_cols)
        return fig, gs
    fig, cell = fig_cell
    return fig, cell.subgridspec(n_rows, n_cols)


def plot_experiment_across_gridpoints(
    experiment: tuple[str, ExpKey],
    metric: str,
    plot_axes: KeepAxisSpec,
    *args: tuple[str, SelectMatch, str | None],
    fig_cell: Optional[tuple[Figure, SubplotSpec]] = None,
    style: str,
    annotations: bool = True,
    shape: Optional[tuple[int, int]] = None,
) -> tuple[Figure, Sequence[str]]:
    """Plot a single experiment's test across multiple gridpoints

    Arguments:
        hexstr: hexadecimal suffix for the experiment's result file.
        metric: The metric who's argopt defines the gridpoint index to match
        plot_axes: Which gridsearch axes and indexes to match
        args: From which gridpoints to load data, described as a local
            name, the parameters for the trial, and the name of the series in
            the trial to match

            Matching logic is AND(OR(parameter matches), OR(index matches))
        style: either "test" or "train"
        shape: Shape of the grid
    Returns:
        Tuple of the plotted figure and names of each subplot
    """

    fig, gs = _setup_summary_fig(shape if shape else len(args), fig_cell=fig_cell)
    if fig_cell is not None and annotations:
        fig.suptitle("How do different smoothing compare on an ODE?")
    p_names = []
    ode_name, hexstr = experiment
    results = cast(GridsearchResultDetails, loadfunc(hexstr, trials_folder=TRIAL_DATA))
    for cell, (p_name, params, series_key) in zip(gs, args):
        if series_key:
            series_data = [results["series_data"][series_key]]
        else:
            series_data = results["series_data"].values()
        locator = GridLocator([metric], plot_axes, [params])
        if matches := find_gridpoints(
            locator,
            results["plot_data"],
            series_data,
            results["metrics"],
            results["scan_grid"]
        ):
            if len(matches) > 1:
                raise ValueError("More than one matches, unsure what to plot")
            p_names.append(p_name)
            for trajectory in matches:
                ax = _plot_train_test_cell(
                    (fig, cell), trajectory, style=style, annotations=False
                )
                if annotations:
                    ax.set_title(p_name)
                break
            else:
                warn(f"Did not find a parameter match for {p_name} experiment")
    if annotations:
        ax.legend()
        fig.suptitle(f"ODE: {ode_name}")
    return fig, p_names


def plot_point_across_experiments(
    named_params: NamedMatch,
    metric: str,
    plot_axes: KeepAxisSpec,
    *exps: tuple[str, ExpKey],
    series_key: str | None = None,
    fig_cell: Optional[tuple[Figure, SubplotSpec]] = None,
    style: str,
    annotations: bool = True,
    shape: Optional[tuple[int, int]] = None,
) -> tuple[Figure, Sequence[str]]:
    """Plot a single parameter's training or test across multiple experiments

    Arguments:
        params: name and parameters defining the gridpoint to match
        metric: The metric who's argopt defines the gridpoint to match
        exps (experiment_name, hexstr): From which experiments to load
            data, described as a local name and the hexadecimal suffix
            of the result file.
        series_key: Which series' argopt arrays to use for matching.
        fig_cell: Optionally, the figure and subplotspec area to add the plots.
        style: either "test" or "train"
        annotations: whether to add labels or leave figure clean
        shape: Shape of the grid
    Returns:
        Tuple of the plotted figure and names of each subplot
    """
    pname, params = named_params
    fig, gs = _setup_summary_fig(shape if shape else len(exps), fig_cell=fig_cell)
    if fig_cell is not None and annotations:
        fig.suptitle("How well does a smoothing method perform across ODEs?")

    for cell, (ode_name, hexstr) in zip(gs, exps):
        results = cast(
            GridsearchResultDetails, loadfunc(hexstr, trials_folder=TRIAL_DATA)
        )
        if series_key:
            series_data = [results["series_data"][series_key]]
        else:
            series_data = results["series_data"].values()
        locator = GridLocator([metric], plot_axes, [params])
        if matches := find_gridpoints(
            locator,
            results["plot_data"],
            series_data,
            results["metrics"],
            results["scan_grid"]
        ):
            if len(matches) > 1:
                raise ValueError("More than one matches, unsure what to plot")
            for trajectory in matches:
                ax = _plot_train_test_cell(
                    (fig, cell), trajectory, style=style, annotations=False
                )
                if annotations:
                    ax.set_title(ode_name)
                break
        else:
            warn(f"Did not find a parameter match for {ode_name} experiment")
    if annotations:
        ax.legend()
        fig.suptitle("How does a set of parameters perform across ODEs?")
    return fig, [name for name, _ in exps]


def _argmaxes_from_gsearch(
    results: GridsearchResultDetails,
) -> list[list[GridsearchResult[np.void]]]:
    sdata = results["series_data"].items()
    all_series: list[list[GridsearchResult[np.void]]] = []
    for _, single_series_all_axes in sdata:
        all_keep_axes: list[GridsearchResult[np.void]] = []
        for _, gridaxis_argmaxes in single_series_all_axes:
            all_keep_axes.append(cast(GridsearchResult[np.void], gridaxis_argmaxes))
        all_series.append(all_keep_axes)
    return all_series


def _plot_train_test_cell(
    fig_cell: tuple[Figure, SubplotSpec | int | tuple[int, int, int]],
    trajectory: SavedGridPoint,
    *,
    style: str,
    annotations: bool = False,
) -> Axes:
    """Plot either the training or test data in a single cell"""
    fig, cell = fig_cell
    if trajectory["data"]["x_test"].shape[1] == 2:
        ax = fig.add_subplot(cell)
        plot_func = genplot._plot_test_sim_data_2d
    else:
        ax = fig.add_subplot(cell, projection="3d")
        plot_func = genplot._plot_test_sim_data_3d
    if style.lower() == "training":
        plot_func = genplot.plot_training_trajectory
        plot_location = ax
        data = (
            trajectory["data"]["x_train"],
            trajectory["data"]["x_true"],
            trajectory["data"]["smooth_train"],
        )
    elif style.lower() == "test":
        plot_location = [ax, ax]
        data = (
            trajectory["data"]["x_test"],
            trajectory["data"]["x_sim"],
        )
    plot_func(plot_location, *data, labels=annotations)
    return ax


def plot_summary_metric(
    metric: str,
    grid_axis_name: str,
    *args: tuple[str, str],
    shape: Optional[tuple[int, int]] = None,
    title: bool = True,
    metric_fname: Optional[str] = None,
    grid_axis_fname: Optional[str] = None,
) -> None:
    """After multiple gridsearches, plot a comparison for all ODEs

    Plots the overall results for a single metric, single grid axis
    Args:
        metric: which metric is being plotted
        grid_axis_name: the name of the parameter varied.
        *args: each additional tuple contains the name of an ODE and
            the hexstr under which it's data is saved.
        shape: Shape of the grid
        title: whether to display title
        grid_axis_fname: A fancy name for the grid axis when printing.
    """
    if metric_fname is None:
        metric_fname = metric
    if grid_axis_fname is None:
        grid_axis_fname = grid_axis_name
    fig, gs = _setup_summary_fig(shape if shape else len(args) + 1)
    title_1 = "How well do the methods work on"
    title_2 = f" different ODEs as {grid_axis_fname}"
    title_3 = f" changes? ({metric_fname})"
    if title:
        fig.suptitle(title_1 + title_2 + title_3)
    for cell, (ode_name, hexstr) in zip(gs, args):
        results = cast(
            GridsearchResultDetails, loadfunc(hexstr, trials_folder=TRIAL_DATA)
        )
        grid_axis_index = results["grid_params"].index(grid_axis_name)
        grid_axis = results["grid_vals"][grid_axis_index]
        metric_index = results["metrics"].index(metric)
        ax = fig.add_subplot(cell)
        for s_name, s_data in results["series_data"].items():
            ax.plot(grid_axis, s_data[grid_axis_index][0][metric_index], label=s_name)
        ax.set_title(ode_name)
    legend_items = ax.get_legend_handles_labels()
    last_cell = gs[-1, -1]
    last_axes = fig.add_subplot(last_cell)
    tx_props = {"fontsize": "large", "bbox": {"fill": False}}
    if not title:
        legend_loc = (0.2, 0.2)
        title_text = f"{title_1}\n{title_2}\n{title_3}\n\n\n\n\n"
        last_axes.text(0.5, 0.5, title_text, ha="center", va="center", **tx_props)
    else:
        legend_loc = "center"
    last_axes.axis("off")
    last_axes.legend(*legend_items, loc=legend_loc)


def plot_summary_test_train(
    exps: Sequence[tuple[str, ExpKey]],
    params: Sequence[tuple[str, SelectMatch, str | None]],
    style: str,
    metric: str,
    plot_axes: KeepAxisSpec,
    row_cat: Optional[str] = None,
) -> Figure:
    """Plot a comparison of different variants across experiments

    Args:
        exps: From which experiments to load data, described as a local name
            and the hexadecimal suffix of the result file.
        params: which gridpoints to compare, described as a tuple of local name,
            parameters to match, and optionally a key to which series of
            argopt arrays.  If params matches multiple grid
            points for a single experiment, an error will be raised.
        style: "test" or "train"
        metric: From which metric's argmax to load data.
        row_cat: row category, either "exps" or "params"
    """
    if row_cat is None:
        row_cat = "exps"
    if row_cat == "exps":
        rows = exps
        cols = params
        col_names = [name for name, _, _ in params]
    elif row_cat == "params":
        rows = params
        cols = exps
        col_names = [name for name, _ in exps]
    else:
        raise ValueError("rows must be either 'exps' or 'params'")
    n_rows = len(rows)
    n_cols = len(cols)
    figsize = (0.5 + 3 * n_cols, 0.5 + 3 * n_rows)
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(n_rows + 1, 2, width_ratios=(1, 12 * n_cols))
    common_args = {"shape": (1, n_cols), "style": style, "annotations": False}

    def label_row(row_name: str, fig: Figure, grid: GridSpec, n_row: int) -> None:
        empty_ax = fig.add_subplot(grid[n_row + 1, 0])
        empty_ax.axis("off")
        empty_ax.text(
            0, 0.5, row_name, va="center", transform=empty_ax.transAxes, rotation=90
        )

    if row_cat == "exps":
        for n_row, (row_name, row_key) in enumerate(exps):
            common_args |= {"fig_cell": (fig, grid[n_row + 1, 1:])}
            plot_experiment_across_gridpoints(
                (row_name, cast(ExpKey, row_key)),
                metric,
                plot_axes,
                *params,
                **common_args,
            )
            label_row(row_name, fig, grid, n_row)
    else:
        for n_row, (row_name, row_key, series_key) in enumerate(params):
            common_args |= {"fig_cell": (fig, grid[n_row + 1, 1:])}
            plot_point_across_experiments(
                (row_name, cast(SelectMatch, row_key)),
                metric,
                plot_axes,
                *cast(Sequence[tuple[str, ExpKey]], cols),
                series_key=series_key,
                **common_args,
            )
            label_row(row_name, fig, grid, n_row)

    first_row = fig.get_axes()[:n_cols]
    for ax, col_name in zip(first_row, col_names):
        ax.set_title(col_name)
    grid.tight_layout(fig)
    return fig

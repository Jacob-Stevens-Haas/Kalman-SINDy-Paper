from dataclasses import dataclass
from types import EllipsisType as ellipsis
from typing import Any, Collection, Optional, Sequence, cast
from warnings import warn

import gen_experiments.plotting as genplot
import matplotlib as mpl
import matplotlib.pyplot as plt
import mitosis
import numpy as np
from gen_experiments.utils import SavedData, _amax_to_full_inds, _grid_locator_match
from matplotlib.axes._axes import Axes
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec, SubplotSpec
from matplotlib.pyplot import Figure

from ._typing import Float1D, Float2D
from .data import TRIAL_DATA

CMAP = mpl.color_sequences["tab10"]
CTRUE = CMAP[0]
CMEAS = CMAP[1]
CEST = CMAP[2]
CGRAY = (0.85, 0.85, 0.85)
plt.rcParams["text.usetex"] = True


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
    fig_cell: Optional[tuple[plt.Figure, SubplotSpec]] = None,
) -> tuple[plt.Figure, GridSpec | GridSpecFromSubplotSpec]:
    """Create neatly laid-out arrangements for subplots

    Creates an evenly-spaced gridpsec to fit follow-on plots and a
    figure, if required.

    Args:
        cells_or_shape: number of grid elements to create, or a tuple of
            rows/cols
        nest_parent: parent grid cell within which to to build a nested
            gridspec
    Returns:
        a figure and gridspec if nest_parent is not provided, otherwise,
        None and a sub-gridspec
    """
    if isinstance(cells_or_shape, int):
        n_sub = cells_or_shape
        n_rows = max(n_sub // 3, (n_sub + 2) // 3)
        n_cols = min(n_sub, 3)
    else:
        n_rows, n_cols = cells_or_shape
    figsize = [3 * n_cols, 3 * n_rows]
    if fig_cell is None:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(n_rows, n_cols)
        return fig, gs
    fig, cell = fig_cell
    return fig, cell.subgridspec(n_rows, n_cols)


def plot_experiment_across_gridpoints(
    hexstr: str,
    *args: tuple[str, dict] | ellipsis | tuple[int | slice, int],
    style: str,
    fig_cell: tuple[Figure, SubplotSpec] = None,
    annotations: bool = True,
    shape: Optional[tuple[int, int]] = None,
) -> tuple[Figure, Sequence[str]]:
    """Plot a single experiment's test across multiple gridpoints

    Arguments:
        hexstr: hexadecimal suffix for the experiment's result file.
        args: From which gridpoints to load data, described either as:
            - a local name and the parameters defining the gridpoint to match.
            - ellipsis, indicating optima across all metrics across all plot
                axes
            - an indexing tuple indicating optima for that tuple's location in
                the gridsearch argmax array
            Matching logic is AND(OR(parameter matches), OR(index matches))
        style: either "test" or "train"
        shape: Shape of the grid
    Returns:
        the plotted figure
    """

    fig, gs = _setup_summary_fig(shape if shape else len(args), fig_cell=fig_cell)
    if fig_cell is not None:
        fig.suptitle("How do different smoothing compare on an ODE?")
    p_names = []
    results = mitosis.load_trial_data(hexstr, trials_folder=TRIAL_DATA)
    amax_arrays = [
        [single_ser_and_axis[1] for single_ser_and_axis in single_series_all_axes]
        for _, single_series_all_axes in results["series_data"].items()
    ]
    parg_inds = {
        argind
        for argind, arg in enumerate(args)
        if isinstance(arg, tuple) and isinstance(arg[0], str)
    }
    indarg_inds = set(range(len(args))) - parg_inds
    pargs = [args[i] for i in parg_inds]
    indargs = [args[i] for i in indarg_inds]
    if not indargs:
        indargs = {...}
    full_inds = _amax_to_full_inds(indargs, amax_arrays)

    for cell, (p_name, params) in zip(gs, pargs):
        for trajectory in results["plot_data"]:
            if _grid_locator_match(
                trajectory["params"], trajectory["pind"], [params], full_inds
            ):
                p_names.append(p_name)
                ax = _plot_train_test_cell(
                    (fig, cell), trajectory, style, annotations=False
                )
                if annotations:
                    ax.set_title(p_name)
                break
        else:
            warn(f"Did not find a parameter match for {p_name} experiment")
    if annotations:
        ax.legend()
    return Figure, p_names


def _plot_train_test_cell(
    fig_cell: tuple[Figure, SubplotSpec | int | tuple[int, int, int]],
    trajectory: SavedData,
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


def plot_point_across_experiments(
    params: dict,
    point: ellipsis | tuple[int | slice, int] = ...,
    *args: tuple[str, str],
    style: str,
    shape: Optional[tuple[int, int]] = None,
) -> Figure:
    """Plot a single parameter's training or test across multiple experiments

    Arguments:
        params: parameters defining the gridpoint to match
        point: gridpoint spec from the argmax array, defined as either an
            - ellipsis, indicating optima across all metrics across all plot
                axes
            - indexing tuple indicating optima for that tuple's location in
                the gridsearch argmax array
        args (experiment_name, hexstr): From which experiments to load
            data, described as a local name and the hexadecimal suffix
            of the result file.
        style: either "test" or "train"
        shape: Shape of the grid
    Returns:
        the plotted figure
    """
    fig, gs = _setup_summary_fig(shape if shape else len(args))
    fig.suptitle("How well does a smoothing method perform across ODEs?")

    for cell, (ode_name, hexstr) in zip(gs, args):
        results = mitosis.load_trial_data(hexstr, trials_folder=TRIAL_DATA)
        amax_arrays = [
            [single_ser_and_axis[1] for single_ser_and_axis in single_series_all_axes]
            for _, single_series_all_axes in results["series_data"].items()
        ]
        full_inds = _amax_to_full_inds((point,), amax_arrays)
        for trajectory in results["plot_data"]:
            if _grid_locator_match(
                trajectory["params"], trajectory["pind"], [params], full_inds
            ):
                ax = _plot_train_test_cell(
                    [fig, cell], trajectory, style, annotations=False
                )
                ax.set_title(ode_name)
                break
        else:
            warn(f"Did not find a parameter match for {ode_name} experiment")
    ax.legend()
    return fig


def plot_summary_metric(
    metric: str,
    grid_axis_name: tuple[str, Collection],
    *args: tuple[str, str],
    shape: Optional[tuple[int, int]] = None,
) -> None:
    """After multiple gridsearches, plot a comparison for all ODEs

    Plots the overall results for a single metric, single grid axis
    Args:
        metric: which metric is being plotted
        grid_axis: the name of the parameter varied and the values of
            the parameter.
        *args: each additional tuple contains the name of an ODE and
            the hexstr under which it's data is saved.
        shape: Shape of the grid
    """
    fig, gs = _setup_summary_fig(shape if shape else len(args))
    fig.suptitle(
        f"How well do the methods work on different ODEs as {grid_axis_name} changes?"
    )
    for cell, (ode_name, hexstr) in zip(gs, args):
        results = mitosis.load_trial_data(hexstr, trials_folder=TRIAL_DATA)
        grid_axis_index = results["grid_params"].index(grid_axis_name)
        grid_axis = results["grid_vals"][grid_axis_index]
        metric_index = results["metrics"].index(metric)
        ax = fig.add_subplot(cell)
        for s_name, s_data in results["series_data"].items():
            ax.plot(grid_axis, s_data[grid_axis_index][0][metric_index], label=s_name)
        ax.set_title(ode_name)
    ax.legend()


def plot_summary_test_train(
    exps: Sequence[tuple[str, str]],
    params: Sequence[tuple[str, dict] | ellipsis | tuple[int | slice, int]],
    style: str,
) -> None:
    """Plot a comparison of different variants across experiments

    Args:
        exps: From which experiments to load data, described as a local name
            and the hexadecimal suffix of the result file.
        params: which gridpoints to compare, described as either:
            - a tuple of local name and parameters to match.
            - ellipsis, indicating optima across all metrics across all plot
                axes
            - an indexing tuple indicating optima for that tuple's location in
                the gridsearch argmax array
            Matching logic is AND(OR(parameter matches), OR(index matches))
        style
    """
    n_exp = len(exps)
    n_params = len(params)
    figsize = (3 * n_params, 3 * n_exp)
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(n_exp, 2, width_ratios=(1, 20))
    for n_row, (ode_name, hexstr) in enumerate(exps):
        cell = grid[n_row, 1]
        _, p_names = plot_experiment_across_gridpoints(
            hexstr, *params, style=style, fig_cell=(fig, cell), annotations=False
        )
        empty_ax = fig.add_subplot(grid[n_row, 0])
        empty_ax.axis("off")
        empty_ax.text(
            -0.1, 0.5, ode_name, va="center", transform=empty_ax.transAxes, rotation=90
        )
    first_row = fig.get_axes()[:n_params]
    for ax, p_name in zip(first_row, p_names):
        ax.set_title(p_name)
    fig.subplots_adjust(top=0.95)
    return fig

from dataclasses import dataclass
from typing import Any, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import Axes

from ._typing import Float1D, Float2D

CMAP = mpl.color_sequences["tab10"]
CTRUE = CMAP[0]
CMEAS = CMAP[1]
CEST = CMAP[2]


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
    ax.quiver(*xyuv, label=r"$\hat\dot x$", color=CEST, **q_props)
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
    ax.plot(t, x_hat, ".", color=CEST)
    xyuv = (t, x_hat, dt * amp, x_dot_hat * dt * amp)
    ax.quiver(*xyuv, color=CEST, **q_props)
    for i, func in enumerate(funcs_theta):
        ax.quiver(
            t,
            x_hat,
            dt * amp,
            func * dt * amp,
            color=CMAP[i + 3],
            label=rf"$θ_{i}(\hat x)$",
            **q_props,
        )
    if ylims is not None:
        ax.set_ylim(*ylims)
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
    ax.plot(t, x_hat, ".", color=CEST)
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
            label=rf"$ξ_{i}θ_{i}(\hat x)$",
            **q_props,
        )
        old_height = old_height + dv
        old_left = old_left + du
    if ylims is not None:
        ax.set_ylim(*ylims)
    ax.legend()
    ax.set_xticks(())


def make_all_plots(pdat: PlotData, q_props: dict[str, Any]) -> None:
    ylims = shared_ylim(pdat.x, pdat.z, pdat.x_hat)
    # data_plot(pdat.t, pdat.x, pdat.z, ylims=ylims)
    fig = plt.figure(figsize=(5, 8))
    fig.suptitle("SINDy process")
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
    axes[0].set_xlabel("(a)")
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
    axes[1].set_xlabel("(b)")
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
    axes[2].set_xlabel("(c)")
    fig.tight_layout()

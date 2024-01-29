from dataclasses import dataclass
from typing import Any
from typing import cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ._typing import Float1D
from ._typing import Float2D

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
    ax = plt.figure().add_axes((0, 0, 1, 1))
    ax.set_title("Kalman smoothing applied to measurements")
    ax.plot(t, x, color=CTRUE, label="true")
    ax.plot(t, z, ".", color=CMEAS, label="measured")
    ax.plot(t, x_hat, "--", color=CEST, label="kalman")
    # ax.quiver(t, z, dt, dz, color=CMEAS, **q_props)
    ax.quiver(t, x_hat, dt * amp, dt * x_dot_hat * amp, color=CEST, **q_props)
    ax.legend()
    if ylims is not None:
        ax.set_ylim(*ylims)


def lib_plot(
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
    ax = plt.figure().add_axes((0, 0, 1, 1))
    ax.plot(t, x_hat, ".", color=CEST, label="Kalman")
    ax.quiver(t, x_hat, dt * amp, x_dot_hat * dt * amp, color=CEST, **q_props)
    for i, func in enumerate(funcs_theta):
        ax.quiver(
            t,
            x_hat,
            dt * amp,
            func * dt * amp,
            color=CMAP[i + 3],
            label=f"$θ_{i}$",
            **q_props,
        )
    if ylims is not None:
        ax.set_ylim(*ylims)
    ax.legend()


def shared_ylim(*args: Float1D) -> tuple[float, float]:
    data_lim = (cast(float, np.min(args)), cast(float, np.max(args)))
    diff = data_lim[1] - data_lim[0]
    margin = diff * 0.1
    return data_lim[0] - margin, data_lim[1] + margin


def soln_plot(
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
    ax = plt.figure().add_axes((0, 0, 1, 1))
    ax.plot(t, x_hat, ".", color=CEST, label="Kalman")
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
            label=f"$ξ_{i}θ_{i}$",
            **q_props,
        )
        old_height = old_height + dv
        old_left = old_left + du
    if ylims is not None:
        ax.set_ylim(*ylims)
    ax.legend()


def make_all_plots(pdat: PlotData, q_props: dict[str, Any]) -> None:
    ylims = shared_ylim(pdat.x, pdat.z, pdat.x_hat)
    # data_plot(pdat.t, pdat.x, pdat.z, ylims=ylims)
    smoothing_plot(
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
    lib_plot(
        pdat.t,
        pdat.dt,
        pdat.x_hat,
        pdat.x_dot_hat,
        pdat.theta,
        amp=pdat.amp,
        ylims=ylims,
        **q_props,
    )
    soln_plot(
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
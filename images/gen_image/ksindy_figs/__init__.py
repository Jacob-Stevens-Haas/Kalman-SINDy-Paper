# %%
from dataclasses import dataclass
from typing import Any
from typing import cast
from typing import Callable
from typing import NewType
from typing import Optional
from typing import TypedDict
from typing import TypeVar

import kalman  # type: ignore
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
from numpy.typing import NBitBase
from scipy import sparse  # type: ignore
from sklearn.linear_model import Lasso

NpFlt = np.dtype[np.floating[NBitBase]]
NTimesteps = NewType("NTimesteps", int)
NSeries = NewType("NSeries", int)
Float1D = np.ndarray[NTimesteps, NpFlt]
Float2D = np.ndarray[tuple[NSeries, NTimesteps], NpFlt]
FloatND = np.ndarray[Any, NpFlt]
FloatOrArray = TypeVar("FloatOrArray", float, FloatND)


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


CMAP = mpl.color_sequences["tab10"]
CTRUE = CMAP[0]
CMEAS = CMAP[1]
CEST = CMAP[2]

np.random.seed(1)
NOISE_VAR = 0.2

# def f_dot(x: FloatOrArray) -> FloatOrArray:
#     return x ** 2 - 2 * x  # type: ignore

# def f(t: FloatOrArray) -> FloatOrArray:
#     return 2 / (1 + np.e ** (2 * t)) # type: ignore


def f_dot(x: FloatOrArray) -> FloatOrArray:
    return -4 - 3 * x  # type: ignore


def f(t: FloatOrArray) -> FloatOrArray:
    return 3 * np.e ** (-3 * t) - 4 / 3  # type: ignore


def gen_sin_data(
    dt: float, noise_var: float
) -> tuple[float, Float1D, Float2D, Float2D, Float2D]:
    t = np.arange(0, 6, dt)
    nt = len(t)
    x = np.stack([np.sin(t), np.cos(t)], axis=0)
    x_dot = np.stack([np.cos(t), -np.sin(t)])
    z = x + np.random.normal(size=x.shape, scale=np.sqrt(noise_var))
    return nt, t, x, x_dot, z


def gen_exp_data(
    dt: float, noise_var: float
) -> tuple[float, Float1D, Float1D, Float1D, Float1D]:
    t = np.arange(0, 1, dt, dtype=float)
    nt = len(t)
    x = f(t)
    x_dot = f_dot(x)
    z = x + np.random.normal(size=x.shape, scale=np.sqrt(noise_var))
    return nt, t, x, x_dot, z


class MathConfig(TypedDict):
    ode_type: str
    dt: float
    t_end: float


class PlotConfig(TypedDict):
    zoom_inds: slice
    q_props: dict[str, Any]


def run_example(m_conf: MathConfig, p_conf: PlotConfig) -> None:
    t, x, x_dot, z, x_hat, x_dot_hat = gen_and_solve(m_conf)
    zoom = p_conf["zoom_inds"]
    dt = fd_arrows(t.reshape((1, -1)))
    dx = fd_arrows(x)
    dz = fd_arrows(z)

    funcs_theta = ps.PolynomialLibrary(2).fit_transform(x_hat.T).T

    pdat = PlotData(
        t[zoom],
        dt[0, zoom],
        x[0, zoom],
        dx[0, zoom],
        x_dot[0, zoom],
        z[0, zoom],
        dz[0, zoom],
        x_hat[0, zoom],
        x_dot_hat[0, zoom],
        funcs_theta[:, zoom],
    )
    make_all_plots(pdat)
    model = ps.STLSQ(unbias=True).fit(funcs_theta.T, x_dot_hat.T)
    print(model.coef_)


def make_all_plots(pdat: PlotData) -> None:
    ylims = shared_ylim(pdat.x, pdat.z, pdat.x_hat)
    data_plot(pdat.t, pdat.x, pdat.z, ylims=ylims)
    smoothing_plot(
        pdat.t,
        pdat.z,
        pdat.dt,
        pdat.dz,
        pdat.x_hat,
        pdat.x_dot_hat,
        ylims=ylims,
        **q_props,
    )
    lib_plot(
        pdat.t, pdat.dt, pdat.x_hat, pdat.x_dot_hat, pdat.theta, ylims=ylims, **q_props
    )


def gen_and_solve(
    config: MathConfig,
) -> tuple[Float1D, Float2D, Float2D, Float2D, Float2D, Float2D]:
    if config["ode_type"] == "sin":
        nt, t, x, x_dot, z = gen_sin_data(0.3, NOISE_VAR)
    else:
        nt, t, x, x_dot, z = gen_exp_data(0.05, NOISE_VAR)
        x = cast(Float2D, x.reshape((1, -1)))
        x_dot = cast(Float2D, x_dot.reshape((1, -1)))
        z = cast(Float2D, z.reshape((1, -1)))
    H = sparse.lil_matrix((nt, 2 * nt))
    H[:, 1::2] = sparse.eye(nt)
    x_hats: list[Float1D] = []
    x_dot_hats: list[Float1D] = []
    for row_z in z:
        x_hat, x_dot_hat, _, _, _ = kalman.solve(row_z, H, t, NOISE_VAR, 1e-1)  # type: ignore
        x_hats.append(cast(Float1D, x_hat))
        x_dot_hats.append(cast(Float1D, x_dot_hat))
    x_hat = np.stack(x_hats)
    x_dot_hat = np.stack(x_dot_hats)

    return t, x, x_dot, z, cast(Float2D, x_hat), cast(Float2D, x_dot_hat)


def fd_arrows(
    x: Float2D,
) -> Float2D:
    deltas = [row[1:] - row[:-1] for row in x]
    return np.array(deltas)


# %%


# %%
def data_plot(
    t: Float1D, x: Float1D, z: Float1D, *, ylims: Optional[tuple[float, float]] = None
) -> None:
    ax = plt.figure().add_axes((0, 0, 1, 1))
    ax.plot(t, x, color=CTRUE, label="true")
    ax.plot(t, z, ".", color=CMEAS, label="measured")
    ax.legend()
    if ylims is not None:
        ax.set_ylim(*ylims)


# %%
def smoothing_plot(
    t: Float1D,
    z: Float1D,
    dt: Float1D,
    dz: Float1D,
    x_hat: Float1D,
    x_dot_hat: Float1D,
    *,
    ylims: Optional[tuple[float, float]] = None,
    **q_props: Any,
) -> None:
    ax = plt.figure().add_axes((0, 0, 1, 1))
    ax.set_title("Kalman smoothing applied to measurements")
    ax.plot(t, z, ".", color=CMEAS, label="measured")
    ax.plot(t, x_hat, "--", color=CEST, label="kalman")
    ax.quiver(t, z, dt, dz, color=CMEAS, **q_props)
    ax.quiver(t, x_hat, dt, x_dot_hat * dt, color=CEST, **q_props)
    ax.legend()
    if ylims is not None:
        ax.set_ylim(*ylims)


# %%
def lib_plot(
    t: Float1D,
    dt: Float1D,
    x_hat: Float1D,
    x_dot_hat: Float1D,
    funcs_theta: Float2D,
    ylims: Optional[tuple[float, float]] = None,
    **q_props: Any,
) -> None:
    """Create plot of function library vectors and smoothed derivative vectors."""
    ax = plt.figure().add_axes((0, 0, 1, 1))
    ax.plot(t, x_hat, ".", color=CEST, label="Kalman")
    ax.quiver(t, x_hat, dt, x_dot_hat * dt, color=CEST, **q_props)
    for i, func in enumerate(funcs_theta):
        ax.quiver(
            t, x_hat, dt, func * dt, color=CMAP[i + 3], label=f"$θ_{i}$", **q_props
        )
    if ylims is not None:
        ax.set_ylim(*ylims)
    ax.legend()


# %%
def shared_ylim(*args: Float1D) -> tuple[float, float]:
    data_lim = (cast(float, np.min(args)), cast(float, np.max(args)))
    diff = data_lim[1] - data_lim[0]
    margin = diff * 0.1
    return data_lim[0] - margin, data_lim[1] + margin


# %%
if __name__ == "__main__":
    m_conf = MathConfig({"ode_type": "sin", "dt": 1, "t_end": 2})
    zoom_start = 0
    zoom_end = zoom_start + 8
    zoom_inds = slice(zoom_start, zoom_end, 1)
    q_props = {"headwidth": 1.5, "headlength": 2, "angles": "xy"}
    p_conf = PlotConfig({"zoom_inds": zoom_inds, "q_props": q_props})
    run_example(m_conf, p_conf)

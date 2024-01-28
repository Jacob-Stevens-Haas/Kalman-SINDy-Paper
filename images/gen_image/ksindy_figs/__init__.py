# %%
from typing import Any
from typing import cast
from typing import Callable
from typing import NewType
from typing import Optional
from typing import TypeVar

import kalman  # type: ignore
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NBitBase
from scipy import sparse  # type: ignore
from sklearn.linear_model import Lasso

NpFlt = np.dtype[np.floating[NBitBase]]
NTimesteps = NewType("NTimesteps", int)
NSeries = NewType("NSeries", int)
Float1D = np.ndarray[NTimesteps, NpFlt]
Float2D = np.ndarray[tuple[NSeries, NTimesteps], NpFlt]
FloatOrArray = TypeVar("FloatOrArray", float, NpFlt)

CMAP = mpl.color_sequences['tab10']
CTRUE = CMAP[0]
CMEAS = CMAP[1]
CEST = CMAP[2]

np.random.seed(1)
NOISE_VAR = .2

def f_dot(t: FloatOrArray) -> FloatOrArray:
    return .3 * t **2 - 2 * t  # type: ignore 

def f(t: FloatOrArray) -> FloatOrArray:
    return 20 / (3 + 2 * np.e ** (2 * t)) # type: ignore

def generate_mock_data(dt: float, noise_var: float) -> tuple[float, Float1D, Float1D, Float1D, Float1D]:
    t = np.arange(0, 6, dt)
    nt = len(t)
    x = np.sin(t)
    x_dot = np.cos(t)
    z = x + np.random.normal(size=x.shape, scale=np.sqrt(noise_var))
    return nt, t, x, x_dot, z
def alt_mock_data(dt: float, noise_var: float) -> tuple[float, Float1D, Float1D, Float1D, Float1D]:
    t = np.arange(0, 6, dt, dtype=float)
    nt = len(t)
    x = f(t)
    x_dot = f_dot(t)
    z = x + np.random.normal(size=x.shape, scale=np.sqrt(noise_var))
    return nt, t, x, x_dot, z

# %%
# nt, t, x, x_dot, z = generate_mock_data(.3, NOISE_VAR)
nt, t, x, x_dot, z = alt_mock_data(.3, NOISE_VAR)
# %%
zoom_start = len(t) //3
zoom_inds = slice(zoom_start, zoom_start+10)
plt.plot(t[zoom_inds], z[zoom_inds], ".")

# %%
H = sparse.lil_matrix((nt, 2 * nt))
H[:, 1::2] = sparse.eye(nt)

x_hat, x_dot_hat, G, Qinv, loss = kalman.solve(z, H, t, NOISE_VAR, 1e-1)

# %%


# %%
def fd_arrows(
    x: Float1D,
    y: Float1D
) -> np.ndarray[tuple[NSeries, NTimesteps], np.dtype[np.floating]]:
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    return np.stack((dx, dy), axis=0)

# %%
fd = fd_arrows(t, x)
zd = fd_arrows(t, z)

q_props = {"headwidth": 1.5, "headlength": 2, "angles": "xy"}

# %%
def data_plot(
    t: Float1D,
    x: Float1D,
    z: Float1D,
    *,
    ylims: Optional[tuple[float, float]] = None
) -> None:
    ax = plt.figure().add_axes((0,0,1,1))
    ax.plot(t, x, color=CTRUE, label="true")
    ax.plot(t, z, ".", color=CMEAS, label="measured")
    ax.legend()
    if ylims is not None:
        ax.set_ylim(*ylims)



# %%
def smoothing_plot(
    t: Float1D,
    z: Float1D,
    zd: Float2D,
    x_hat: Float1D,
    x_dot_hat: Float1D,
    *,
    ylims: Optional[tuple[float, float]] = None,
    **q_props: Any
) -> None:
    dt = zd[0]
    ax = plt.figure().add_axes((0,0,1,1))
    ax.set_title("Kalman smoothing applied to measurements")
    ax.plot(t, z, ".", color=CMEAS, label="measured")
    ax.plot(t, x_hat, "--", color=CEST, label="kalman")
    ax.quiver(t, z, dt, zd[1], color=CMEAS, **q_props)
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
    **q_props: Any
) -> None:
    """Create plot of function library vectors and smoothed derivative vectors."""
    ax = plt.figure().add_axes((0,0,1,1))
    ax.plot(t, x_hat, ".", color=CEST, label="Kalman")
    ax.quiver(t, x_hat, dt, x_dot_hat * dt, color=CEST, **q_props)
    for i, func in enumerate(funcs_theta):
        ax.quiver(t, x_hat, dt, func * dt, color=CMAP[i+3], label=f"$θ_{i}$", **q_props)
    if ylims is not None:
        ax.set_ylim(*ylims)
    ax.legend()


# %%
def shared_ylim(*args: Float1D) -> tuple[float, float]:
    data_lim = (cast(float, np.min(args)), cast(float, np.max(args)))
    diff = data_lim[1] - data_lim[0]
    margin = diff * .1
    return data_lim[0] - margin, data_lim[1] + margin

# %%
if __name__ == "__main__":
    ylims = shared_ylim(x[zoom_inds], z[zoom_inds], x_hat[zoom_inds])
    data_plot(t[zoom_inds], x[zoom_inds], z[zoom_inds], ylims=ylims)
    funcs_theta = np.stack((np.ones_like(x_hat), x_hat, x_hat**2), axis=0)
    smoothing_plot(
        t[zoom_inds],
        z[zoom_inds],
        zd[:, zoom_inds],
        x_hat[zoom_inds],
        x_dot_hat[zoom_inds],
        ylims=ylims,
        **q_props
    )
    lib_plot(
        t[zoom_inds],
        fd[0, zoom_inds],
        x_hat[zoom_inds],
        x_dot_hat[zoom_inds],
        funcs_theta[:, zoom_inds],
        ylims=ylims,
        **q_props
    )

    model = Lasso(1e-1, fit_intercept=False).fit(funcs_theta.T, x_dot_hat)
    model.coef_

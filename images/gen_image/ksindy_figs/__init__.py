from typing import Any
from typing import cast
from typing import TypedDict

import kalman  # type: ignore
import numpy as np
import pysindy as ps
from scipy import sparse  # type: ignore
from sklearn.linear_model import Lasso

from ksindy_figs.data import gen_sin_data, gen_exp_data
from ksindy_figs.plotting import PlotData, make_all_plots
from ksindy_figs._typing import Float1D
from ksindy_figs._typing import Float2D


class MathConfig(TypedDict):
    ode_type: str
    dt: float
    t_end: float
    noise_var: float


class PlotConfig(TypedDict):
    zoom_inds: slice
    q_props: dict[str, Any]


def run_example(m_conf: MathConfig, p_conf: PlotConfig) -> None:
    t, x, x_dot, z, x_hat, x_dot_hat = gen_and_solve(m_conf)
    zoom = p_conf["zoom_inds"]
    dt = fd_vec(t.reshape((1, -1)))
    dx = fd_vec(x)
    dz = fd_vec(z)

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

    if len(pdat.dt) != len(pdat.t): raise ValueError("Slice is too small")

    make_all_plots(pdat, q_props)
    model = ps.STLSQ(unbias=True).fit(funcs_theta.T, x_dot_hat.T)
    print(model.coef_)


def gen_and_solve(
    config: MathConfig,
) -> tuple[Float1D, Float2D, Float2D, Float2D, Float2D, Float2D]:
    if config["ode_type"] == "sin":
        nt, t, x, x_dot, z = gen_sin_data(config["dt"], config["noise_var"])
    else:
        nt, t, x, x_dot, z = gen_exp_data(config["dt"], config["noise_var"])
        x = cast(Float2D, x.reshape((1, -1)))
        x_dot = cast(Float2D, x_dot.reshape((1, -1)))
        z = cast(Float2D, z.reshape((1, -1)))
    H = sparse.lil_matrix((nt, 2 * nt))
    H[:, 1::2] = sparse.eye(nt)
    x_hats: list[Float1D] = []
    x_dot_hats: list[Float1D] = []
    for row_z in z:
        x_hat, x_dot_hat, _, _, _ = kalman.solve(
            row_z, H, t, config["noise_var"], 1e-1  # type: ignore
        )
        x_hats.append(cast(Float1D, x_hat))
        x_dot_hats.append(cast(Float1D, x_dot_hat))
    x_hat = np.stack(x_hats)
    x_dot_hat = np.stack(x_dot_hats)

    return t, x, x_dot, z, cast(Float2D, x_hat), cast(Float2D, x_dot_hat)


def fd_vec(
    x: Float2D,
) -> Float2D:
    deltas = [row[1:] - row[:-1] for row in x]
    return np.array(deltas)


# %%
if __name__ == "__main__":
    m_conf = MathConfig({"ode_type": "sin", "dt": .1, "t_end": 6, "noise_var": .1})
    zoom_start = 0
    # zoom_end = m_conf["t_end"] // m_conf["dt"]
    zoom_end = zoom_start + 50
    zoom_step = 6
    zoom_inds = slice(zoom_start, zoom_end, zoom_step)
    q_props = {"headwidth": 1.5, "headlength": 2, "angles": "xy"}
    p_conf = PlotConfig({"zoom_inds": zoom_inds, "q_props": q_props})
    run_example(m_conf, p_conf)


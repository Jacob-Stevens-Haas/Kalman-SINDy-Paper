from pathlib import Path

import numpy as np
from gen_experiments.gridsearch import GridsearchResultDetails
from mitosis import load_trial_data

from ._typing import Float1D, Float2D, FloatOrArray

TRIAL_DATA = Path(__file__).resolve().parent.parent / "trials"

rng = np.random.default_rng(1)
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
    z = x + rng.normal(size=x.shape, scale=np.sqrt(noise_var))
    return nt, t, x, x_dot, z


def gen_exp_data(
    dt: float, noise_var: float
) -> tuple[float, Float1D, Float1D, Float1D, Float1D]:
    t = np.arange(0, 1, dt, dtype=float)
    nt = len(t)
    x = f(t)
    x_dot = f_dot(x)
    z = x + rng.normal(size=x.shape, scale=np.sqrt(noise_var))
    return nt, t, x, x_dot, z


def load_mitosis_5(
    hexstr: str, trials_folder: str | Path = TRIAL_DATA
) -> GridsearchResultDetails:
    """Load results if mitosis 0.5 is installed

    The mitosis 0.5 saves reusults as a list of results from each step.
    This experiment is meant to just have one step, however.
    """
    results = load_trial_data(hexstr, trials_folder=trials_folder)
    return results[0]

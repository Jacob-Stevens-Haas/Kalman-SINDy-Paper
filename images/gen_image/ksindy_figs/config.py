from collections.abc import Iterable, Sequence
from typing import TypeVar, cast

import numpy as np
import pysindy as ps
from numpy.typing import NDArray

from gen_experiments.data import _signal_avg_power
from gen_experiments.gridsearch.typing import (
    GridLocator,
    SeriesDef,
    SeriesList,
    SkinnySpecs,
)
from gen_experiments.plotting import _PlotPrefs
from gen_experiments.typing import NestedDict
from gen_experiments.utils import FullSINDyTrialData

T = TypeVar("T", bound=str)
U = TypeVar("U")


def ND(d: dict[T, U]) -> NestedDict[T, U]:
    return NestedDict(**d)


def _convert_abs_rel_noise(
    scan_grid: dict[str, NDArray[np.floating]],
    recent_results: FullSINDyTrialData,
) -> dict[str, Sequence[np.floating]]:
    """Convert abs_noise grid_vals to rel_noise"""
    signal = np.stack(recent_results["x_true"], axis=-1)
    signal_power = _signal_avg_power(signal)
    plot_grid = scan_grid.copy()
    new_vals = plot_grid["sim_params.noise_abs"] / signal_power
    plot_grid["sim_params.noise_rel"] = new_vals
    plot_grid.pop("sim_params.noise_abs")
    return cast(dict[str, Sequence[np.floating]], plot_grid)


# To allow pickling
def identity(x):
    return x


def quadratic(x):
    return x * x


def addn(x):
    return x + x


plot_prefs = {
    "test": _PlotPrefs(),
    "test-absrel": _PlotPrefs(
        True, False, GridLocator(..., {("sim_params.noise_abs", (1,))})
    ),
    "test-absrel2": _PlotPrefs(
        True,
        False,
        GridLocator(
            ...,
            (..., ...),
            (
                {"sim_params.noise_abs": 0.1},
                {"sim_params.noise_abs": 0.5},
                {"sim_params.noise_abs": 1},
                {"sim_params.noise_abs": 2},
                {"sim_params.noise_abs": 4},
                {"sim_params.noise_abs": 8},
            ),
        ),
    ),
    "absrel-newloc": _PlotPrefs(
        True,
        False,
        GridLocator(
            ["coeff_mse", "coeff_f1"],
            (..., (2, 3, 4)),
            (
                {"diff_params.kind": "kalman", "diff_params.alpha": "gcv"},
                {
                    "diff_params.kind": "kalman",
                    "diff_params.alpha": lambda a: isinstance(a, float | int),
                },
                {"diff_params.kind": "trend_filtered"},
                {"diff_params.diffcls": "SmoothedFiniteDifference"},
            ),
        ),
    ),
    "all-kernel": _PlotPrefs(
        True,
        False,
        GridLocator(..., (..., ...), ({"diff_params.kind": "kernel"},)),
    ),
}
sim_params = {
    "debug": ND({"n_trajectories": 1, "dt":.1, "t_end": 1, "noise_abs": 0.0}),
    "test": ND({"n_trajectories": 2}),
    "test-r1": ND({"n_trajectories": 2, "noise_rel": 0.01}),
    "test-r2": ND({"n_trajectories": 2, "noise_rel": 0.1}),
    "test-r3": ND({"n_trajectories": 2, "noise_rel": 0.3}),
    "10x": ND({"n_trajectories": 10}),
    "10x-r1": ND({"n_trajectories": 10, "noise_rel": 0.01}),
    "10x-r2": ND({"n_trajectories": 10, "noise_rel": 0.05}),
    "10x-plot-noise": ND({"n_trajectories": 10, "noise_rel": 0.1, "t_end": 8}),
    "test2": ND({"n_trajectories": 2, "noise_abs": 0.4}),
    "med-noise": ND({"n_trajectories": 2, "noise_abs": 0.8}),
    "med-noise-many": ND({"n_trajectories": 10, "noise_abs": 0.8}),
    "hi-noise": ND({"n_trajectories": 2, "noise_abs": 2}),
    "pde-ic1": ND({"init_cond": np.exp(-((np.linspace(-8, 8, 256) + 2) ** 2) / 2)}),
    "pde-ic2": ND({
        "init_cond": (np.cos(np.linspace(0, 100, 1024))) * (
            1 + np.sin(np.linspace(0, 100, 1024) - 0.5)
        )
    }),
}
diff_params = {
    "test": ND({"diffcls": "FiniteDifference"}),
    "autoks": ND({"diffcls": "sindy", "kind": "kalman", "alpha": "gcv"}),
    "test_axis": ND({"diffcls": "FiniteDifference", "axis": -2}),
    "test2": ND({"diffcls": "SmoothedFiniteDifference"}),
    "tv": ND({"diffcls": "sindy", "kind": "trend_filtered", "order": 0, "alpha": 1}),
    "savgol": ND({"diffcls": "sindy", "kind": "savitzky_golay"}),
    "sfd-nox": ND({"diffcls": "SmoothedFiniteDifference", "save_smooth": False}),
    "sfd-ps": ND({"diffcls": "SmoothedFiniteDifference"}),
    "kalman": ND({"diffcls": "sindy", "kind": "kalman", "alpha": 0.000055}),
    "kalman-empty2": ND({"diffcls": "sindy", "kind": "kalman", "alpha": None}),
    "kalman-auto": ND(
        {"diffcls": "sindy", "kind": "kalman", "alpha": None, "meas_var": 0.8}
    ),
    "kernel-default": ND({"diffcls": "sindy", "kind":"kernel"}),
}
feat_params = {
    "test": ND({"featcls": "Polynomial"}),
    "test2": ND({"featcls": "Fourier"}),
    "cubic": ND({"featcls": "Polynomial", "degree": 3}),
    "quadratic": ND({"featcls": "Polynomial", "degree": 2}),
    "quadratic-noconst": ND({"featcls": "Polynomial", "degree": 2, "include_bias": False}),
    "testweak": ND({"featcls": "weak"}),  # needs work
    "pde2": ND({
        "featcls": "pde",
        "function_library": ps.PolynomialLibrary(degree=2, include_bias=False),
        "derivative_order": 2,
        "spatial_grid": np.linspace(-8, 8, 256),
        "include_interaction": True,
    }),
    "pde4": ND({
        "featcls": "pde",
        "function_library": ps.PolynomialLibrary(degree=2, include_bias=False),
        "derivative_order": 4,
        "spatial_grid": np.linspace(0, 100, 1024),
        "include_interaction": True,
    }),
}
opt_params = {
    "test": ND({"optcls": "STLSQ"}),
    "test_low": ND({"optcls": "STLSQ", "threshold": 0.09}),
    "miosr": ND({"optcls": "MIOSR"}),
    "enslsq": ND(
        {"optcls": "ensemble", "opt": ps.STLSQ(), "bagging": True, "n_models": 20}
    ),
    "ensmio-ho-vdp-lv-duff": ND({
        "optcls": "ensemble",
        "opt": ps.MIOSR(target_sparsity=4, unbias=True),
        "bagging": True,
        "n_models": 20,
    }),
    "ensmio-hopf": ND({
        "optcls": "ensemble",
        "opt": ps.MIOSR(target_sparsity=8, unbias=True),
        "bagging": True,
        "n_models": 20,
    }),
    "ensmio-lorenz-ross": ND({
        "optcls": "ensemble",
        "opt": ps.MIOSR(target_sparsity=7, unbias=True),
        "bagging": True,
        "n_models": 20,
    }),
    "mio-lorenz-ross": ND({"optcls": "MIOSR", "target_sparsity": 7, "unbias": True}),
    "miosr-vdp-quad": ND({"optcls": "MIOSR", "target_sparsity": 3, "unbias": True}),
    "miosr-vdp-cub": ND({"optcls": "MIOSR", "target_sparsity": 4, "unbias": True}),
}

# Grid search parameters
metrics = {
    "test": ["coeff_f1", "coeff_mae"],
    "all-coeffs": ["coeff_f1", "coeff_mae", "coeff_mse"],
    "all": ["coeff_f1", "coeff_precision", "coeff_recall", "coeff_mae", "coeff_mse"],
    "lorenzk": ["coeff_f1", "coeff_precision", "coeff_recall", "coeff_mae"],
    "1": ["coeff_f1", "coeff_precision", "coeff_mse", "coeff_mae"],
}
other_params = {
    "debug": ND({
        "sim_params": sim_params["debug"],
        "diff_params": diff_params["kernel-default"],
        "feat_params": feat_params["test"],
        "opt_params": opt_params["test"],
    }),
    "test": ND({
        "sim_params": sim_params["test"],
        "diff_params": diff_params["test"],
        "feat_params": feat_params["test"],
        "opt_params": opt_params["test"],
    }),
    "test-pde1": ND({
        "sim_params": sim_params["pde-ic1"],
        "diff_params": diff_params["test_axis"],
        "feat_params": feat_params["pde2"],
        "opt_params": opt_params["test_low"],
    }),
    "test-pde2": ND({
        "sim_params": sim_params["pde-ic2"],
        "diff_params": diff_params["test_axis"],
        "feat_params": feat_params["pde4"],
        "opt_params": opt_params["test"],
    }),
    "tv1": ND({
        "sim_params": sim_params["test"],
        "diff_params": diff_params["tv"],
        "feat_params": feat_params["test"],
        "opt_params": opt_params["test"],
    }),
    "test2": ND({
        "sim_params": sim_params["test"],
        "feat_params": feat_params["test"],
        "opt_params": opt_params["test"],
    }),
    "test-kalman-heuristic2": ND({
        "sim_params": sim_params["test"],
        "diff_params": diff_params["kalman-empty2"],
        "feat_params": feat_params["test"],
        "opt_params": opt_params["test"],
    }),
    "lorenzk": ND({
        "sim_params": sim_params["test"],
        "diff_params": diff_params["kalman"],
        "feat_params": feat_params["test"],
        "opt_params": opt_params["test"],
    }),
    "exp1": ND({
        "sim_params": sim_params["10x"],
        "feat_params": feat_params["test"],
        "opt_params": opt_params["enslsq"],
    }),
    "cubic": ND({
        "sim_params": sim_params["test"],
        "feat_params": feat_params["cubic"],
        "opt_params": opt_params["test"],
    }),
    "exp2": ND({
        "sim_params": sim_params["10x"],
        "feat_params": feat_params["cubic"],
        "opt_params": opt_params["enslsq"],
    }),
    "abs-exp3": ND({
        "sim_params": sim_params["med-noise-many"],
        "feat_params": feat_params["cubic"],
        "opt_params": opt_params["ensmio-lorenz-ross"],
    }),
    "rel-exp3-lorenz": ND({
        "sim_params": sim_params["10x"],
        "feat_params": feat_params["cubic"],
        "opt_params": opt_params["ensmio-lorenz-ross"],
    }),
    "lor-ross-cubic": ND({
        "sim_params": sim_params["10x"],
        "feat_params": feat_params["cubic"],
        "opt_params": opt_params["ensmio-lorenz-ross"],
    }),
    "lor-ross-kernel": ND({
        "diff_params": diff_params["kernel-default"],
        "sim_params": sim_params["10x-plot-noise"],
        "feat_params": feat_params["cubic"],
        "opt_params": opt_params["ensmio-lorenz-ross"],
    }),
    "lor-ross-cubic-fast": ND({
        "sim_params": sim_params["test"],
        "feat_params": feat_params["cubic"],
        "opt_params": opt_params["mio-lorenz-ross"],
    }),
    "4nonzero-cubic": ND({
        "sim_params": sim_params["10x"],
        "feat_params": feat_params["cubic"],
        "opt_params": opt_params["ensmio-ho-vdp-lv-duff"],
    }),
    "4nonzero-kernel": ND({
        "diff_params": diff_params["kernel-default"],
        "sim_params": sim_params["10x-plot-noise"],
        "feat_params": feat_params["cubic"],
        "opt_params": opt_params["ensmio-ho-vdp-lv-duff"],
    }),
    "hopf-cubic": ND({
        "sim_params": sim_params["10x"],
        "feat_params": feat_params["cubic"],
        "opt_params": opt_params["ensmio-hopf"],
    }),
    "hopf-kernel": ND({
        "diff_params": diff_params["kernel-default"],
        "sim_params": sim_params["10x-plot-noise"],
        "feat_params": feat_params["cubic"],
        "opt_params": opt_params["ensmio-hopf"],
    }),
}
grid_params = {
    "test": ["sim_params.t_end"],
    "abs_noise": ["sim_params.noise_abs"],
    "abs_noise-kalman": ["sim_params.noise_abs", "diff_params.meas_var"],
    "tv1": ["diff_params.alpha"],
    "lorenzk": ["sim_params.t_end", "sim_params.noise_abs", "diff_params.alpha"],
    "duration-absnoise": ["sim_params.t_end", "sim_params.noise_abs"],
    "rel_noise": ["sim_params.t_end", "sim_params.noise_rel"],
    "kernel_noise": ["diff_params.lmbd"],
    "kernel_noise_scale": ["diff_params.lmbd", "diff_params.sigma"],
}
grid_vals: dict[str, list[Iterable]] = {
    "test": [[5, 10, 15, 20]],
    "abs_noise": [[0.1, 0.5, 1, 2, 4, 8]],
    "abs_noise-kalman": [[0.1, 0.5, 1, 2, 4, 8], [0.1, 0.5, 1, 2, 4, 8]],
    "abs_noise-kalman2": [[0.1, 0.5, 1, 2, 4, 8], [0.01, 0.25, 1, 4, 16, 64]],
    "tv1": [np.logspace(-4, 0, 5)],
    "tv2": [np.logspace(-3, -1, 5)],
    "small_even": [np.logspace(-2, 2, 5)],
    "small_even2": [np.logspace(-2, 2, 5), np.logspace(-2, 2, 5)],
    "lorenzk": [[1, 9, 27], [0.1, 0.8], np.logspace(-6, -1, 4)],
    "lorenz1": [[1, 3, 9, 27], [0.01, 0.1, 1]],
    "duration-absnoise": [[0.5, 1, 2, 4, 8, 16], [0.1, 0.5, 1, 2, 4, 8]],
    "rel_noise": [[0.5, 1, 2, 4, 8, 16], [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]],
}
grid_decisions = {
    "test": ["plot"],
    "plot1": ["plot", "max"],
    "lorenzk": ["plot", "plot", "max"],
    "plot2": ["plot", "plot"],
    "noplot": ["max", "max"],
}
diff_series: dict[str, SeriesDef] = {
    "kalman1": SeriesDef(
        "Kalman",
        diff_params["kalman"],
        ["diff_params.alpha"],
        [np.logspace(-6, 0, 3)],
    ),
    "kalman2": SeriesDef(
        "Kalman",
        diff_params["kalman"],
        ["diff_params.alpha"],
        [np.logspace(-4, 0, 5)],
    ),
    "auto-kalman": SeriesDef(
        "Auto Kalman",
        diff_params["kalman"],
        ["diff_params.alpha", "diff_params.meas_var"],
        [(None,), (0.1, 0.5, 1, 2, 4, 8)],
    ),
    "auto-kalman2": SeriesDef(
        "Auto Kalman",
        diff_params["kalman"],
        ["diff_params.alpha", "diff_params.meas_var"],
        [(None,), (0.01, 0.25, 1, 4, 16, 64)],
    ),
    "auto-kalman3": SeriesDef(
        "Auto Kalman",
        diff_params["kalman"],
        ["diff_params.alpha"],
        [("None",)],
    ),
    "auto-kalman4": SeriesDef(
        "Auto Kalman",
        diff_params["kalman"],
        ["diff_params.alpha"],
        [("gcv",)],
    ),
    "tv1": SeriesDef(
        "Total Variation",
        diff_params["tv"],
        ["diff_params.alpha"],
        [np.logspace(-6, 0, 3)],
    ),
    "tv2": SeriesDef(
        "Total Variation",
        diff_params["tv"],
        ["diff_params.alpha"],
        [np.logspace(-4, 0, 5)],
    ),
    "sg1": SeriesDef(
        "Savitsky-Golay",
        diff_params["sfd-ps"],
        ["diff_params.smoother_kws.window_length"],
        [[5, 7, 15]],
    ),
    "sg2": SeriesDef(
        "Savitsky-Golay",
        diff_params["sfd-ps"],
        ["diff_params.smoother_kws.window_length"],
        [[5, 8, 12, 15]],
    ),
    "kernel-line": SeriesDef(
        "Gaussian RBF vs noise",
        diff_params["kernel-default"],
        ["diff_params.lmbd"],
        [[1e-2, 1e-1, 1e0, 1e1, 1e2]]
    ),
    "kernel-box": SeriesDef(
        "Gaussian RBF, noise and scale",
        diff_params["kernel-default"],
        ["diff_params.lmbd", "diff_params.sigma"],
        [[1e-2, 1e-1, 1e0, 1e1, 1e2], [1e-2, 1e-1, 1e0, 1e1, 1e2]]
    ),
}
series_params: dict[str, SeriesList] = {
    "test": SeriesList(
        "diff_params",
        "Differentiation Method",
        [
            diff_series["kalman1"],
            diff_series["tv1"],
            diff_series["sg1"],
        ],
    ),
    "lorenz1": SeriesList(
        "diff_params",
        "Differentiation Method",
        [
            diff_series["kalman2"],
            diff_series["tv2"],
            diff_series["sg2"],
        ],
    ),
    "kalman-auto": SeriesList(
        "diff_params",
        "Differentiation Method",
        [
            diff_series["auto-kalman"],
            diff_series["tv2"],
            diff_series["sg2"],
        ],
    ),
    "kalman-auto2": SeriesList(
        "diff_params",
        "Differentiation Method",
        [
            diff_series["auto-kalman2"],
            diff_series["tv2"],
            diff_series["sg2"],
        ],
    ),
    "kalman-auto3": SeriesList(
        "diff_params",
        "Differentiation Method",
        [
            diff_series["auto-kalman3"],
            diff_series["tv2"],
            diff_series["sg2"],
        ],
    ),
    "kalman-auto4": SeriesList(
        "diff_params",
        "Differentiation Method",
        [
            diff_series["auto-kalman4"],
            diff_series["tv2"],
            diff_series["sg2"],
        ],
    ),
    "multikalman": SeriesList(
        "diff_params",
        "Differentiation Method",
        [
            diff_series["auto-kalman"],
            diff_series["kalman2"],
            diff_series["tv2"],
            diff_series["sg2"],
        ],
    ),
    "multikalman2": SeriesList(
        "diff_params",
        "Differentiation Method",
        [
            diff_series["auto-kalman4"],
            diff_series["kalman2"],
            diff_series["tv2"],
            diff_series["sg2"],
        ],
    ),
    "kernel1": SeriesList(
        "diff_params",
        "Differentiation Method",
        [diff_series["kernel-line"]]
    ),
    "kernel2": SeriesList(
        "diff_params",
        "Differentiation Method",
        [diff_series["kernel-box"]]
    ),
}


skinny_specs: dict[str, SkinnySpecs] = {
    "exp3": (
        ("sim_params.noise_abs", "diff_params.meas_var"),
        ((identity,), (identity,)),
    ),
    "abs_noise-kalman": (
        tuple(grid_params["abs_noise-kalman"]),
        ((identity,), (identity,)),
    ),
    "duration-noise-kalman": (
        ("sim_params.t_end", "sim_params.noise_abs", "diff_params.meas_var"),
        ((1, 1), (-1, identity), (-1, identity)),
    ),
    "duration-noise": (("sim_params.t_end", "sim_params.noise_rel"), ((1,), (-1,))),
}

lu = {
    "plot_prefs": plot_prefs,
    "sim_params": sim_params,
    "diff_params": diff_params,
    "feat_params": feat_params,
    "opt_params": opt_params,
    "metrics": metrics,
    "other_params": other_params,
    "grid_params": grid_params,
    "grid_vals": grid_vals,
    "grid_decisions": grid_decisions,
    "diff_series": diff_series,
    "series_params": series_params,
    "skinny_specs": skinny_specs
}

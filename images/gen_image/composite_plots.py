# %%
from typing import cast

import mitosis
from gen_experiments.utils import FullSINDyTrialData, strict_find_grid_match

import ksindy_figs.data as data
import ksindy_figs.plotting as plots
from ksindy_figs.plotting import ExpKey

exp_hexes = {
    "Cubic HO": ExpKey("0b044c"),
    "Duffing": ExpKey("b78632"),
    "Hopf": ExpKey("2255b2"),
    "Lotka-Volterra": ExpKey("128ce1"),
    "Rossler": ExpKey("7a3950"),
    "SHO": ExpKey("04b738"),
    "Van der Pol": ExpKey("1a23e7"),
}

# %%
plots.plot_summary_metric(
    "coeff_mae",
    "sim_params.t_end",
    *exp_hexes.items(),
    shape=(2, 4),
    title=False,
    metric_fname=r"$\xi$ MAE",
    grid_axis_fname="data duration",
)
pass
# %%
plots.plot_summary_metric(
    "coeff_f1",
    "sim_params.t_end",
    *exp_hexes.items(),
    shape=(2, 4),
    title=False,
    metric_fname=r"$\xi$ F1 score",
    grid_axis_fname="data duration",
)
pass
# %%
plots.plot_summary_metric(
    "coeff_mae",
    "sim_params.noise_rel",
    *exp_hexes.items(),
    shape=(2, 4),
    title=False,
    metric_fname=r"$\xi$ MAE",
    grid_axis_fname="relative noise",
)
pass
# %%
plots.plot_summary_metric(
    "coeff_f1",
    "sim_params.noise_rel",
    *exp_hexes.items(),
    shape=(2, 4),
    title=False,
    metric_fname=r"$\xi$ F1 score",
    grid_axis_fname="relative noise",
)
pass
# %%

noise_params = {"sim_params.t_end": 16, "sim_params.noise_abs": 1}
params_kalman = noise_params | {"diff_params.kind": "kalman"}
params_kalmanauto = noise_params | {"diff_params.kind": "kalman"}
params_tv = noise_params | {"diff_params.kind": "trend_filtered"}
params_savgol = noise_params | {"diff_params.diffcls": "SmoothedFiniteDifference"}

if params_kalmanauto==params_kalman:
    raise ValueError("Cannot distinguish between kalman and kalman auto hyperparameters.")
# %%
fig = plots.plot_summary_test_train(
    [*exp_hexes.items()],
    [("Kalman", params_kalman), ("KalmanAuto", params_kalmanauto), ("TV", params_tv), ("SavGol", params_savgol)],
    style="training",
    row_cat="params",   
)
pass
# %%
fig = plots.plot_summary_test_train(
    [*exp_hexes.items()],
    [("Kalman", params_kalman), ("KalmanAuto", params_kalmanauto), ("TV", params_tv), ("SavGol", params_savgol)],
    style="test",
    row_cat="params",
)
pass

# %% [markdown]

# ### Additional code to demonstrate helper functions

# %%
# plots.plot_point_across_experiments(
#     ("Kalman", params_kalman),
#     ...,
#     *exp_hexes.items(),
#     style="test",
#     shape=(1, 7),
#     annotations=False,
# )
# %%


# results = mitosis.load_trial_data(exp_hexes["Cubic HO"], trials_folder=data.TRIAL_DATA)

# plots.plot_experiment_across_gridpoints(
#     ("Cubic HO", exp_hexes["Cubic HO"]),
#     ("Kalman", params_kalman),
#     ("TV", params_tv),
#     ("SavGol", params_savgol),
#     style="test",
#     shape=(3, 1),
# )
# %%
# plots.plot_experiment_across_gridpoints(
#     ("Rossler", exp_hexes["Rossler"]),
#     ("Kalman", params_kalman),
#     ("TV", params_tv),
#     ("SavGol", params_savgol),
#     style="test",
#     shape=(3, 1),
# )

# %%
# from gen_experiments.odes import plot_ode_panel

# single_result = strict_find_grid_match(results, params=params_kalman)
# plot_ode_panel(cast(FullSINDyTrialData, single_result))

# %%

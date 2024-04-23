# %%
import ksindy_figs.plotting as plots
from ksindy_figs.plotting import ExpKey

exp_hexes = {
    "Cubic HO": ExpKey("23526c"),
    "Duffing": ExpKey("23de18"),
    "Hopf": ExpKey("0b9747"),
    "Lotka-Volterra": ExpKey("539481"),
    "Rossler": ExpKey("7b9e27"),
    "SHO": ExpKey("c5d866"),
    "Van der Pol": ExpKey("48a935"),
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
metric = "coeff_mse"
plot_axes = [("sim_params.t_end", (4,))]
noise_params = {"sim_params.t_end": 8, "sim_params.noise_rel": 0.1}
params_kalman = noise_params | {"diff_params.kind": "kalman", "diff_params.alpha": lambda a: isinstance(a, float | int)}
params_kalmanauto = noise_params | {"diff_params.kind": "kalman", "diff_params.alpha": lambda a: a is None}
params_tv = noise_params | {"diff_params.kind": "trend_filtered"}
params_savgol = noise_params | {"diff_params.diffcls": "SmoothedFiniteDifference"}

# %%
fig = plots.plot_summary_test_train(
    [*exp_hexes.items()],
    [
        ("Kalman", params_kalman, "Kalman"),
        ("KalmanAuto", params_kalmanauto, "Auto Kalman"),
        ("TV", params_tv, "Total Variation"),
        ("SavGol", params_savgol, "Savitsky-Golay")
    ],
    style="training",
    metric=metric,
    plot_axes=plot_axes,
    row_cat="params",   
)
pass
# %%
fig = plots.plot_summary_test_train(
    [*exp_hexes.items()],
    [
        ("Kalman", params_kalman, "Kalman"),
        ("KalmanAuto", params_kalmanauto, "Auto Kalman"),
        ("TV", params_tv, "Total Variation"),
        ("SavGol", params_savgol, "Savitsky-Golay")
    ],
    style="test",
    metric=metric,
    plot_axes=plot_axes,
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
from ksindy_figs.data import TRIAL_DATA, load_mitosis_5

results = load_mitosis_5(exp_hexes["Cubic HO"], trials_folder=TRIAL_DATA)

# %%
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
from gen_experiments.odes import plot_ode_panel
from gen_experiments.gridsearch import find_gridpoints, GridLocator

where = GridLocator()
single_result = find_gridpoints(results, params=params_kalman)
plot_ode_panel(cast(FullSINDyTrialData, single_result))

# %%

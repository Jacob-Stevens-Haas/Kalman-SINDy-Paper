# %%
import mitosis
from gen_experiments.utils import strict_find_grid_match

import ksindy_figs.data as data
import ksindy_figs.plotting as plots
from ksindy_figs.plotting import ExpKey

exp_hexes = {
    "Cubic HO": ExpKey("d41b48"),
    "Duffing": ExpKey("9a2339"),
    "Hopf": ExpKey("e6e691"),
    "Lotka-Volterra": ExpKey("d453b9"),
    "Rossler": ExpKey("831dc2"),
    "SHO": ExpKey("a1757b"),
    "Van der Pol": ExpKey("25e6c3"),
}

# %%
plots.plot_summary_metric(
    "coeff_mae", "sim_params.t_end", *exp_hexes.items(), shape=(2, 4)
)
pass
# %%
plots.plot_summary_metric(
    "coeff_f1", "sim_params.t_end", *exp_hexes.items(), shape=(2, 4)
)
pass
# %%

noise_params = {"sim_params.t_end": 16, "sim_params.noise_abs": 1}
params_kalman = noise_params | {"diff_params.kind": "kalman"}
params_tv = noise_params | {"diff_params.kind": "trend_filtered"}
params_savgol = noise_params | {"diff_params.diffcls": "SmoothedFiniteDifference"}

# %%
fig = plots.plot_summary_test_train(
    [*exp_hexes.items()],
    [("Kalman", params_kalman), ("TV", params_tv), ("SavGol", params_savgol)],
    style="training",
    row_cat="params",
)
pass
# %%
fig = plots.plot_summary_test_train(
    [*exp_hexes.items()],
    [("Kalman", params_kalman), ("TV", params_tv), ("SavGol", params_savgol)],
    style="test",
    row_cat="exps",
)
pass

# %% [markdown]

# ### Additional code to demonstrate helper functions

# %%
plots.plot_point_across_experiments(
    ("Kalman", params_kalman),
    ...,
    *exp_hexes.items(),
    style="test",
    shape=(1, 7),
    annotations=False,
)
# %%


results = mitosis.load_trial_data(exp_hexes["Cubic HO"], trials_folder=data.TRIAL_DATA)

plots.plot_experiment_across_gridpoints(
    ("Cubic HO", exp_hexes["Cubic HO"]),
    ("Kalman", params_kalman),
    ("TV", params_tv),
    ("SavGol", params_savgol),
    style="test",
    shape=(3, 1),
)
# %%
plots.plot_experiment_across_gridpoints(
    ("Rossler", exp_hexes["Rossler"]),
    ("Kalman", params_kalman),
    ("TV", params_tv),
    ("SavGol", params_savgol),
    style="test",
    shape=(3, 1),
)

# %%
from gen_experiments.odes import plot_ode_panel

single_result = strict_find_grid_match(results, params=params_kalman)
plot_ode_panel(single_result)

# %%

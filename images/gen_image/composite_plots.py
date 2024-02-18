# %%
import mitosis
from gen_experiments.utils import strict_find_grid_match

import ksindy_figs.data as data
import ksindy_figs.plotting as plots
from ksindy_figs.plotting import ExpKey

exp_hexes = {
    "cubic_ho": ExpKey("d41b48"),
    "duff": ExpKey("9a2339"),
    "hopf": ExpKey("e6e691"),
    "lv": ExpKey("d453b9"),
    "ross": ExpKey("831dc2"),
    "sho": ExpKey("a1757b"),
    "vdp": ExpKey("25e6c3"),
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
plots.plot_summary_test_train(
    [*exp_hexes.items()],
    [("Kalman", params_kalman), ("TV", params_tv), ("SavGol", params_savgol)],
    style="training",
    row_cat="params",
)
pass
# %%
plots.plot_summary_test_train(
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


results = mitosis.load_trial_data(exp_hexes["cubic_ho"], trials_folder=data.TRIAL_DATA)

plots.plot_experiment_across_gridpoints(
    ("Cubic HO", exp_hexes["cubic_ho"]),
    ("Kalman", params_kalman),
    ("TV", params_tv),
    ("SavGol", params_savgol),
    style="test",
    shape=(3, 1),
)
# %%
plots.plot_experiment_across_gridpoints(
    exp_hexes["ross"],
    ("Kalman", params_kalman),
    ("TV", params_tv),
    ("SavGol", params_savgol),
    style="training",
    shape=(3, 1),
)

# %%
plots.plot_summary_test_train(
    [*exp_hexes.items()],
    [("Kalman", params_kalman), ("TV", params_tv), ("SavGol", params_savgol)],
    style="test",
)

# %%
from gen_experiments.odes import plot_ode_panel

single_result = strict_find_grid_match(results, params=params_kalman)
plot_ode_panel(single_result)

# %%

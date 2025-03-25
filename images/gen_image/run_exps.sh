    # --param other_params=lor-ross-kernel \
nohup mitosis gridsearch \
    --debug \
    --config images/gen_image/pyproject.toml \
    -F trials/debug \
    -e gridsearch.seed=19 \
    -e gridsearch.group=\"none\" \
    --param gridsearch.metrics=all \
    --param gridsearch.other_params=4nonzero-kernel \
    --param gridsearch.grid_params=kernel_noise_scale \
    --param gridsearch.grid_vals=small_even2 \
    --param gridsearch.grid_decisions=noplot \
    --param +gridsearch.plot_prefs=absrel-newloc  &> mock.log &

nohup mitosis gridsearch \
    --config images/gen_image/pyproject.toml \
    -F trials \
    -e gridsearch.seed=19 \
    -e gridsearch.group=\"lorenz\" \
    --param gridsearch.metrics=all \
    --param gridsearch.other_params=lor-ross-kernel \
    --param gridsearch.grid_params=kernel_noise_scale \
    --param gridsearch.grid_vals=small_even2 \
    --param gridsearch.grid_decisions=noplot \
    --param +gridsearch.plot_prefs=absrel-newloc  &> lorenz.log &

nohup mitosis gridsearch \
    --config images/gen_image/pyproject.toml \
    -F trials \
    -e gridsearch.seed=19 \
    -e gridsearch.group=\"cubic_ho\" \
    --param gridsearch.metrics=all \
    --param gridsearch.other_params=4nonzero-kernel \
    --param gridsearch.grid_params=kernel_noise_scale \
    --param gridsearch.grid_vals=small_even2 \
    --param gridsearch.grid_decisions=noplot \
    --param +gridsearch.plot_prefs=absrel-newloc  &> cubic_ho.log &

nohup mitosis gridsearch \
    --config images/gen_image/pyproject.toml \
    -F trials \
    -e gridsearch.seed=19 \
    -e gridsearch.group=\"sho\" \
    --param gridsearch.metrics=all \
    --param gridsearch.other_params=4nonzero-kernel \
    --param gridsearch.grid_params=kernel_noise_scale \
    --param gridsearch.grid_vals=small_even2 \
    --param gridsearch.grid_decisions=noplot \
    --param +gridsearch.plot_prefs=absrel-newloc  &> sho.log &

nohup mitosis gridsearch \
    --config images/gen_image/pyproject.toml \
    -F trials \
    -e gridsearch.seed=19 \
    -e gridsearch.group=\"vdp\" \
    --param gridsearch.metrics=all \
    --param gridsearch.other_params=4nonzero-kernel \
    --param gridsearch.grid_params=kernel_noise_scale \
    --param gridsearch.grid_vals=small_even2 \
    --param gridsearch.grid_decisions=noplot \
    --param +gridsearch.plot_prefs=absrel-newloc  &> vdp.log &

nohup mitosis gridsearch \
    --config images/gen_image/pyproject.toml \
    -F trials \
    -e gridsearch.seed=19 \
    -e gridsearch.group=\"lv\" \
    --param gridsearch.metrics=all \
    --param gridsearch.other_params=4nonzero-kernel \
    --param gridsearch.grid_params=kernel_noise_scale \
    --param gridsearch.grid_vals=small_even2 \
    --param gridsearch.grid_decisions=noplot \
    --param +gridsearch.plot_prefs=absrel-newloc  &> lv.log &

nohup mitosis gridsearch \
    --config images/gen_image/pyproject.toml \
    -F trials \
    -e gridsearch.seed=19 \
    -e gridsearch.group=\"duff\" \
    --param gridsearch.metrics=all \
    --param gridsearch.other_params=4nonzero-kernel \
    --param gridsearch.grid_params=kernel_noise_scale \
    --param gridsearch.grid_vals=small_even2 \
    --param gridsearch.grid_decisions=noplot \
    --param +gridsearch.plot_prefs=absrel-newloc  &> duff.log &

nohup mitosis gridsearch \
    --config images/gen_image/pyproject.toml \
    -F trials \
    -e gridsearch.seed=19 \
    -e gridsearch.group=\"hopf\" \
    --param gridsearch.metrics=all \
    --param gridsearch.other_params=hopf-kernel \
    --param gridsearch.grid_params=kernel_noise_scale \
    --param gridsearch.grid_vals=small_even2 \
    --param gridsearch.grid_decisions=noplot \
    --param +gridsearch.plot_prefs=absrel-newloc  &> hopf.log &

nohup mitosis gridsearch \
    --config images/gen_image/pyproject.toml \
    -F trials \
    -e gridsearch.seed=19 \
    -e gridsearch.group=\"ross\" \
    --param gridsearch.metrics=all \
    --param gridsearch.other_params=lor-ross-kernel \
    --param gridsearch.grid_params=kernel_noise_scale \
    --param gridsearch.grid_vals=small_even2 \
    --param gridsearch.grid_decisions=noplot \
    --param +gridsearch.plot_prefs=absrel-newloc  &> ross.log &

# nohup mitosis gridsearch \
#     -e seed=19 \
#     -g ross \
#     -F trials \
#     --param metrics=all \
#     --param other_params=debug \
#     --param grid_params=duration-absnoise \
#     --param grid_vals=debug \
#     --param grid_decisions=noplot \
#     --param +series_params=kalman-auto3 \
#     --param +plot_prefs=test-absrel5 \
#     &> ross.log &

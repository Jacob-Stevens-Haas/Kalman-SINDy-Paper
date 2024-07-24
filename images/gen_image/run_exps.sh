    # --param other_params=lor-ross-cubic \
nohup mitosis gridsearch \
    --debug \
    --config images/gen_image/pyproject.toml \
    -F trials/debug \
    -e gridsearch.seed=19 \
    -e gridsearch.group=\"none\" \
    --param gridsearch.metrics=all \
    --param gridsearch.other_params=4nonzero-cubic \
    --param gridsearch.grid_params=rel_noise \
    --param gridsearch.grid_vals=rel_noise \
    --param gridsearch.grid_decisions=plot2 \
    --param gridsearch.series_params=multikalman2 \
    --param +gridsearch.plot_prefs=absrel-newloc \
    --param +gridsearch.skinny_specs=duration-noise &> mock.log &

nohup mitosis gridsearch \
    --config images/gen_image/pyproject.toml \
    -F trials \
    -e gridsearch.seed=19 \
    -e gridsearch.group=\"lorenz\" \
    --param gridsearch.metrics=all \
    --param gridsearch.other_params=lor-ross-cubic \
    --param gridsearch.grid_params=rel_noise \
    --param gridsearch.grid_vals=rel_noise \
    --param gridsearch.grid_decisions=plot2 \
    --param gridsearch.series_params=multikalman2 \
    --param +gridsearch.plot_prefs=absrel-newloc \
    --param +gridsearch.skinny_specs=duration-noise &> lorenz.log &

nohup mitosis gridsearch \
    --config images/gen_image/pyproject.toml \
    -F trials \
    -e gridsearch.seed=19 \
    -e gridsearch.group=\"cubic_ho\" \
    --param gridsearch.metrics=all \
    --param gridsearch.other_params=4nonzero-cubic \
    --param gridsearch.grid_params=rel_noise \
    --param gridsearch.grid_vals=rel_noise \
    --param gridsearch.grid_decisions=plot2 \
    --param gridsearch.series_params=multikalman2 \
    --param +gridsearch.plot_prefs=absrel-newloc \
    --param +gridsearch.skinny_specs=duration-noise &> cubic_ho.log &

nohup mitosis gridsearch \
    --config images/gen_image/pyproject.toml \
    -F trials \
    -e gridsearch.seed=19 \
    -e gridsearch.group=\"sho\" \
    --param gridsearch.metrics=all \
    --param gridsearch.other_params=4nonzero-cubic \
    --param gridsearch.grid_params=rel_noise \
    --param gridsearch.grid_vals=rel_noise \
    --param gridsearch.grid_decisions=plot2 \
    --param gridsearch.series_params=multikalman2 \
    --param +gridsearch.plot_prefs=absrel-newloc \
    --param +gridsearch.skinny_specs=duration-noise &> sho.log &

nohup mitosis gridsearch \
    --config images/gen_image/pyproject.toml \
    -F trials \
    -e gridsearch.seed=19 \
    -e gridsearch.group=\"vdp\" \
    --param gridsearch.metrics=all \
    --param gridsearch.other_params=4nonzero-cubic \
    --param gridsearch.grid_params=rel_noise \
    --param gridsearch.grid_vals=rel_noise \
    --param gridsearch.grid_decisions=plot2 \
    --param gridsearch.series_params=multikalman2 \
    --param +gridsearch.plot_prefs=absrel-newloc \
    --param +gridsearch.skinny_specs=duration-noise &> vdp.log &

nohup mitosis gridsearch \
    --config images/gen_image/pyproject.toml \
    -F trials \
    -e gridsearch.seed=19 \
    -e gridsearch.group=\"lv\" \
    --param gridsearch.metrics=all \
    --param gridsearch.other_params=4nonzero-cubic \
    --param gridsearch.grid_params=rel_noise \
    --param gridsearch.grid_vals=rel_noise \
    --param gridsearch.grid_decisions=plot2 \
    --param gridsearch.series_params=multikalman2 \
    --param +gridsearch.plot_prefs=absrel-newloc \
    --param +gridsearch.skinny_specs=duration-noise &> lv.log &

nohup mitosis gridsearch \
    --config images/gen_image/pyproject.toml \
    -F trials \
    -e gridsearch.seed=19 \
    -e gridsearch.group=\"duff\" \
    --param gridsearch.metrics=all \
    --param gridsearch.other_params=4nonzero-cubic \
    --param gridsearch.grid_params=rel_noise \
    --param gridsearch.grid_vals=rel_noise \
    --param gridsearch.grid_decisions=plot2 \
    --param gridsearch.series_params=multikalman2 \
    --param +gridsearch.plot_prefs=absrel-newloc \
    --param +gridsearch.skinny_specs=duration-noise &> duff.log &

nohup mitosis gridsearch \
    --config images/gen_image/pyproject.toml \
    -F trials \
    -e gridsearch.seed=19 \
    -e gridsearch.group=\"hopf\" \
    --param gridsearch.metrics=all \
    --param gridsearch.other_params=hopf-cubic \
    --param gridsearch.grid_params=rel_noise \
    --param gridsearch.grid_vals=rel_noise \
    --param gridsearch.grid_decisions=plot2 \
    --param gridsearch.series_params=multikalman2 \
    --param +gridsearch.plot_prefs=absrel-newloc \
    --param +gridsearch.skinny_specs=duration-noise &> hopf.log &

nohup mitosis gridsearch \
    --config images/gen_image/pyproject.toml \
    -F trials \
    -e gridsearch.seed=19 \
    -e gridsearch.group=\"ross\" \
    --param gridsearch.metrics=all \
    --param gridsearch.other_params=lor-ross-cubic \
    --param gridsearch.grid_params=rel_noise \
    --param gridsearch.grid_vals=rel_noise \
    --param gridsearch.grid_decisions=plot2 \
    --param gridsearch.series_params=multikalman2 \
    --param +gridsearch.plot_prefs=absrel-newloc \
    --param +gridsearch.skinny_specs=duration-noise &> ross.log &

# nohup mitosis gridsearch \
#     -e seed=19 \
#     -g ross \
#     -F trials \
#     --param metrics=all \
#     --param other_params=debug \
#     --param grid_params=duration-absnoise \
#     --param grid_vals=debug \
#     --param grid_decisions=plot2 \
#     --param +series_params=kalman-auto3 \
#     --param +plot_prefs=test-absrel5 \
#     &> ross.log &

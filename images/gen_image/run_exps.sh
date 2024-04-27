    # --param other_params=lor-ross-cubic \
nohup mitosis -m gen_experiments.gridsearch \
    -e seed=19 \
    --debug \
    -e group=\"none\" \
    -F trials/debug \
    --param metrics=all \
    --param other_params=4nonzero-cubic \
    --param grid_params=rel_noise \
    --param grid_vals=rel_noise \
    --param grid_decisions=plot2 \
    --param series_params=multikalman \
    --param +plot_prefs=absrel-newloc \
    --param +skinny_specs=duration-noise &> mock.log &

nohup mitosis -m gen_experiments.gridsearch \
    -e seed=19 \
    -e group=\"lorenz\" \
    -F trials \
    --param metrics=all \
    --param other_params=lor-ross-cubic \
    --param grid_params=rel_noise \
    --param grid_vals=rel_noise \
    --param grid_decisions=plot2 \
    --param series_params=multikalman \
    --param +plot_prefs=absrel-newloc \
    --param +skinny_specs=duration-noise &> lorenz.log &

nohup mitosis -m gen_experiments.gridsearch \
    -e seed=19 \
    -e group=\"cubic_ho\" \
    -F trials \
    --param metrics=all \
    --param other_params=4nonzero-cubic \
    --param grid_params=rel_noise \
    --param grid_vals=rel_noise \
    --param grid_decisions=plot2 \
    --param series_params=multikalman \
    --param +plot_prefs=absrel-newloc \
    --param +skinny_specs=duration-noise &> cubic_ho.log &

nohup mitosis -m gen_experiments.gridsearch \
    -e seed=19 \
    -e group=\"sho\" \
    -F trials \
    --param metrics=all \
    --param other_params=4nonzero-cubic \
    --param grid_params=rel_noise \
    --param grid_vals=rel_noise \
    --param grid_decisions=plot2 \
    --param series_params=multikalman \
    --param +plot_prefs=absrel-newloc \
    --param +skinny_specs=duration-noise &> sho.log &

nohup mitosis -m gen_experiments.gridsearch \
    -e seed=19 \
    -e group=\"vdp\" \
    -F trials \
    --param metrics=all \
    --param other_params=4nonzero-cubic \
    --param grid_params=rel_noise \
    --param grid_vals=rel_noise \
    --param grid_decisions=plot2 \
    --param series_params=multikalman \
    --param +plot_prefs=absrel-newloc \
    --param +skinny_specs=duration-noise &> vdp.log &

nohup mitosis -m gen_experiments.gridsearch \
    -e seed=19 \
    -e group=\"lv\" \
    -F trials \
    --param metrics=all \
    --param other_params=4nonzero-cubic \
    --param grid_params=rel_noise \
    --param grid_vals=rel_noise \
    --param grid_decisions=plot2 \
    --param series_params=multikalman \
    --param +plot_prefs=absrel-newloc \
    --param +skinny_specs=duration-noise &> lv.log &

nohup mitosis -m gen_experiments.gridsearch \
    -e seed=19 \
    -e group=\"duff\" \
    -F trials \
    --param metrics=all \
    --param other_params=4nonzero-cubic \
    --param grid_params=rel_noise \
    --param grid_vals=rel_noise \
    --param grid_decisions=plot2 \
    --param series_params=multikalman \
    --param +plot_prefs=absrel-newloc \
    --param +skinny_specs=duration-noise &> duff.log &

nohup mitosis -m gen_experiments.gridsearch \
    -e seed=19 \
    -e group=\"hopf\" \
    -F trials \
    --param metrics=all \
    --param other_params=hopf-cubic \
    --param grid_params=rel_noise \
    --param grid_vals=rel_noise \
    --param grid_decisions=plot2 \
    --param series_params=multikalman \
    --param +plot_prefs=absrel-newloc \
    --param +skinny_specs=duration-noise &> hopf.log &

nohup mitosis -m gen_experiments.gridsearch \
    -e seed=19 \
    -e group=\"ross\" \
    -F trials \
    --param metrics=all \
    --param other_params=lor-ross-cubic \
    --param grid_params=rel_noise \
    --param grid_vals=rel_noise \
    --param grid_decisions=plot2 \
    --param series_params=multikalman \
    --param +plot_prefs=absrel-newloc \
    --param +skinny_specs=duration-noise &> ross.log &

# nohup mitosis gen_experiments.gridsearch \
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

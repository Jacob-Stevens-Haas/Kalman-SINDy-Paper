    # --param other_params=lor-ross-cubic \
nohup mitosis gen_experiments.gridsearch \
    -e seed=19 \
    --debug \
    -g none \
    -F trials/debug \
    --param metrics=all \
    --param other_params=4nonzero-cubic \
    --param grid_params=duration-absnoise \
    --param grid_vals=duration-absnoise \
    --param grid_decisions=plot2 \
    --param series_params=kalman-auto3 \
    --param +plot_prefs=test-absrel5 \
    --param +skinny_specs=duration-noise &> mock.log &

nohup mitosis gen_experiments.gridsearch \
    -e seed=19 \
    -g lorenz \
    -F trials \
    --param metrics=all \
    --param other_params=lor-ross-cubic \
    --param grid_params=duration-absnoise \
    --param grid_vals=duration-absnoise \
    --param grid_decisions=plot2 \
    --param series_params=kalman-auto3 \
    --param +plot_prefs=test-absrel5 \
    --param +skinny_specs=duration-noise &> lorenz.log &

nohup mitosis gen_experiments.gridsearch \
    -e seed=19 \
    -g cubic_ho \
    -F trials \
    --param metrics=all \
    --param other_params=4nonzero-cubic \
    --param grid_params=duration-absnoise \
    --param grid_vals=duration-absnoise \
    --param grid_decisions=plot2 \
    --param series_params=kalman-auto3 \
    --param +plot_prefs=test-absrel5 \
    --param +skinny_specs=duration-noise &> cubic_ho.log &

nohup mitosis gen_experiments.gridsearch \
    -e seed=19 \
    -g sho \
    -F trials \
    --param metrics=all \
    --param other_params=4nonzero-cubic \
    --param grid_params=duration-absnoise \
    --param grid_vals=duration-absnoise \
    --param grid_decisions=plot2 \
    --param series_params=kalman-auto3 \
    --param +plot_prefs=test-absrel5 \
    --param +skinny_specs=duration-noise &> sho.log &

nohup mitosis gen_experiments.gridsearch \
    -e seed=19 \
    -g vdp \
    -F trials \
    --param metrics=all \
    --param other_params=4nonzero-cubic \
    --param grid_params=duration-absnoise \
    --param grid_vals=duration-absnoise \
    --param grid_decisions=plot2 \
    --param series_params=kalman-auto3 \
    --param +plot_prefs=test-absrel5 \
    --param +skinny_specs=duration-noise &> vdp.log &

nohup mitosis gen_experiments.gridsearch \
    -e seed=19 \
    -g lv \
    -F trials \
    --param metrics=all \
    --param other_params=4nonzero-cubic \
    --param grid_params=duration-absnoise \
    --param grid_vals=duration-absnoise \
    --param grid_decisions=plot2 \
    --param series_params=kalman-auto3 \
    --param +plot_prefs=test-absrel5 \
    --param +skinny_specs=duration-noise &> lv.log &

nohup mitosis gen_experiments.gridsearch \
    -e seed=19 \
    -g duff \
    -F trials \
    --param metrics=all \
    --param other_params=4nonzero-cubic \
    --param grid_params=duration-absnoise \
    --param grid_vals=duration-absnoise \
    --param grid_decisions=plot2 \
    --param series_params=kalman-auto3 \
    --param +plot_prefs=test-absrel5 \
    --param +skinny_specs=duration-noise &> duff.log &

nohup mitosis gen_experiments.gridsearch \
    -e seed=19 \
    -g hopf \
    -F trials \
    --param metrics=all \
    --param other_params=hopf-cubic \
    --param grid_params=duration-absnoise \
    --param grid_vals=duration-absnoise \
    --param grid_decisions=plot2 \
    --param series_params=kalman-auto3 \
    --param +plot_prefs=test-absrel5 \
    --param +skinny_specs=duration-noise &> hopf.log &

nohup mitosis gen_experiments.gridsearch \
    -e seed=19 \
    -g ross \
    -F trials \
    --param metrics=all \
    --param other_params=lor-ross-cubic \
    --param grid_params=duration-absnoise \
    --param grid_vals=duration-absnoise \
    --param grid_decisions=plot2 \
    --param series_params=kalman-auto3 \
    --param +plot_prefs=test-absrel5 \
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

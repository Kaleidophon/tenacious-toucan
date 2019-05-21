"""
Plot results of experiment 5.
"""

# PROJECT
from src.utils.plot import *

LOGDIR = "logs/exp6/"
IMGDIR = "img/exp6/"

# ### Plot loss data ###

# Define how to infer model names from paths
def name_function(path):
    shortened = path[:path.rfind("_") - 1].replace(LOGDIR, "")
    if shortened:
        shortened = shortened.split("_")[0]

    return NAME_DICT[shortened]

# Define how to color different model losses
def loss_color_function(curve_type, model_name, y_name):
    return COLOR_DICT[curve_type][model_name]

# Plot training losses
train_selection_func = lambda path: "train" in path
train_log_paths = get_logs_in_dir(LOGDIR, train_selection_func)
train_logs = aggregate_logs(train_log_paths, name_function)
plot_column(
    train_logs, x_name="batch_num", y_names="batch_loss", intervals=False, save_path=f"{IMGDIR}train_losses.png",
    title="Train loss (n=1) | PPL step", color_func=loss_color_function, selection=slice(0, 5700), y_top_lim=15
)

# Plot validation losses
val_selection_func = lambda path: "val" in path
val_log_paths = get_logs_in_dir(LOGDIR, val_selection_func)
val_logs = aggregate_logs(val_log_paths, name_function)
plot_column(
    val_logs, x_name="batch_num", y_names="val_ppl", intervals=False, save_path=f"{IMGDIR}val_ppls.png",
    title="Validation perplexity (n=1) | PPL step", color_func=loss_color_function, selection=slice(0, 27)
)

# Plot step sizes for different models
ppl_step_selection_func = lambda path: "train" in path and "vanilla" not in path
ppl_step_log_paths = get_logs_in_dir(LOGDIR, ppl_step_selection_func)
ppl_step_logs = aggregate_logs(ppl_step_log_paths, name_function)
plot_column(
    ppl_step_logs, x_name="batch_num", y_names="step_sizes_hx_l1", intervals=False, save_path=f"{IMGDIR}ppl_steps.png",
    title="PPL step sizes (n=1)", color_func=loss_color_function, selection=slice(0, 5700)
)

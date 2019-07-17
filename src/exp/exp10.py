# PROJECT
from src.utils.plot import *
from src.utils.log import *

LOGDIR = "logs/exp10/step_sizes/"
IMGDIR = "img/exp10/step_sizes/"

# ### Plot loss data ###

# Define how to infer model names from paths
def name_function(path):
    path = path.replace("mcd_", "")
    shortened = path[:path.rfind("_") - 1].replace(LOGDIR, "")
    if shortened:
        shortened = shortened.split("_")[0]

    return shortened

# Define how to color different model losses
def loss_color_function(curve_type, model_name, y_name):
    return COLOR_DICT[curve_type][model_name]

# Plot training losses
train_selection_func = lambda path: "train" in path
train_log_paths = get_logs_in_dir(LOGDIR, train_selection_func)
train_logs = aggregate_logs(train_log_paths, name_function)
plot_column(
    train_logs, x_name="batch_num", y_names="batch_loss", intervals=True, save_path=f"{IMGDIR}train_losses.png",
    title="Train loss (n=4)", selection=slice(0, 5700), y_top_lim=15
)

# Plot validation losses
val_selection_func = lambda path: "val" in path
val_log_paths = get_logs_in_dir(LOGDIR, val_selection_func)
val_logs = aggregate_logs(val_log_paths, name_function)
plot_column(
    val_logs, x_name="batch_num", y_names="val_ppl", intervals=True, save_path=f"{IMGDIR}val_ppls.png",
    title="Validation perplexity (n=4)", #selection=slice(0, 27)
)

# Plot step sizes for different models
step_selection_func = lambda path: "train" in path and "adapt" in path
step_log_paths = get_logs_in_dir(LOGDIR, step_selection_func)
step_logs = aggregate_logs(step_log_paths, name_function)
plot_column(
    step_logs, x_name="batch_num",
    y_names=["step_sizes_hx_l0", "step_sizes_cx_l0", "step_sizes_hx_l1", "step_sizes_cx_l1"], y_label="Step size",
    intervals=True, save_path=f"{IMGDIR}ppl_steps.png", legend_func=lambda model_name, y_name: y_name,
    title="Recoding step sizes (n=4)", #selection=slice(0, 5700)
)
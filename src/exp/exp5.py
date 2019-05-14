"""
Plot results of experiment 5.
"""

# PROJECT
from src.utils.plot import *

LOGDIR = "logs/exp5/"
IMGDIR = "img/exp5/"

# ### Plot loss data ###

# Define how to infer model names from paths
def name_function(path):
    shortened = path[:path.rfind("_") - 1].replace(LOGDIR, "")
    if shortened:
        shortened = shortened.split("_")[0]

    if shortened == "mcd0":
        shortened = "vanilla"

    return NAME_DICT[shortened]

# Define how to color different model losses
def loss_color_function(curve_type, model_name, y_name):
    return COLOR_DICT[curve_type][model_name]

# Plot training losses
train_selection_func = lambda path: "train" in path and not "mcd0" in path
train_log_paths = get_logs_in_dir(LOGDIR, train_selection_func)
train_logs = aggregate_logs(train_log_paths, name_function)
plot_column(
    train_logs, x_name="batch_num", y_names="batch_loss", intervals=False, save_path=f"{IMGDIR}train_losses.png",
    title="Train loss (n=3)", color_func=loss_color_function, selection=slice(0, 5700), y_top_lim=13
)

# Plot validation losses
val_selection_func = lambda path: "val" in path and not "mcd0" in path
val_log_paths = get_logs_in_dir(LOGDIR, val_selection_func)
val_logs = aggregate_logs(val_log_paths, name_function)
plot_column(
    val_logs, x_name="batch_num", y_names="val_ppl", intervals=False, save_path=f"{IMGDIR}val_ppls.png",
    title="Validation perplexity (n=3)", color_func=loss_color_function, selection=slice(0, 27)
)

# Plot recoding grad norms for BAE
bae_recoding_selection_func = lambda path: "ensemble" in path and "train" in path
bae_recoding_log_paths = get_logs_in_dir(LOGDIR, bae_recoding_selection_func)
bae_recoding_logs = aggregate_logs(bae_recoding_log_paths, name_function)


def recoding_grad_color_func(curve_type, model_name, y_name):
    parts = y_name.split("_")
    grad_type, layer_num = parts[-2], int(parts[-1][1])
    return RECODING_GRAD_COLOR_DICT[grad_type][layer_num]

def recoding_grad_legend_func(model_name, y_name):
    parts = y_name.split("_")
    return parts[-2] + "_" + parts[-1]

gradient_columns = ["recoding_grads_hx_l0",	"recoding_grads_cx_l0",	"recoding_grads_hx_l1",	"recoding_grads_cx_l1"]

plot_column(
    bae_recoding_logs, x_name="batch_num", y_names=gradient_columns, intervals=True,
    save_path=f"{IMGDIR}gradient_norms_bae.png", title="BAE recoding gradient norms (n=3)",
    color_func=recoding_grad_color_func, legend_func=recoding_grad_legend_func, y_label="Recoding grad norm",
)

# Plot recoding grad norms for fixed MCD
mcd_recoding_selection_func = lambda path: "mcd" in path and "train" in path and not "mcd0" in path
mcd_recoding_log_paths = get_logs_in_dir(LOGDIR, mcd_recoding_selection_func)
mcd_recoding_logs = aggregate_logs(mcd_recoding_log_paths, name_function)

plot_column(
    mcd_recoding_logs, x_name="batch_num", y_names=gradient_columns, intervals=True,
    save_path=f"{IMGDIR}gradient_norms_mcd.png", title="MCD recoding gradient norms (n=3)",
    color_func=recoding_grad_color_func, legend_func=recoding_grad_legend_func, y_label="Recoding grad norm",
    y_top_lim=2
)

# Plot deltas for mcd / ensemble / vanilla
delta_selection_func = lambda path: "train" in path and "vanilla" not in path
delta_log_paths = get_logs_in_dir(LOGDIR, delta_selection_func)
delta_logs = aggregate_logs(delta_log_paths, name_function)

plot_column(
    delta_logs, x_name="batch_num", y_names="deltas", intervals=False, save_path=f"{IMGDIR}deltas.png",
    title="Deltas (n=3)", color_func=loss_color_function, selection=slice(2550, 2600)
)

"""
Plot results of experiment 1 - 3.
"""

# PROJECT
from src.utils.plot import *

LOGDIR = "logs/"

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
    train_logs, x_name="batch_num", y_names="batch_loss", intervals=False, save_path="img/train_losses.png",
    title="Train loss (n=3)", color_func=loss_color_function, selection=slice(0, 1000), y_top_lim=13
)

# Plot validation losses
val_selection_func = lambda path: "val" in path
val_log_paths = get_logs_in_dir(LOGDIR, val_selection_func)
val_logs = aggregate_logs(val_log_paths, name_function)
plot_column(
    val_logs, x_name="batch_num", y_names="val_ppl", intervals=False, save_path="img/val_ppls.png",
    title="Validation perplexity (n=3)", color_func=loss_color_function, y_top_lim=6000
)

# ### Plot additional information ###

# Plot uncertainty estimates for perplexity-based recoding model
ppl_recoding_selection_func = lambda path: "ppl" in path and "train" in path and "vanilla" not in path
ppl_recoding_log_paths = get_logs_in_dir(LOGDIR, ppl_recoding_selection_func)
ppl_recoding_logs = aggregate_logs(ppl_recoding_log_paths, name_function)
plot_column(
    ppl_recoding_logs, x_name="batch_num", y_names="deltas", intervals=False, save_path="img/deltas_ppl.png",
    title="Uncertainty estimates (n=3)", color_func=loss_color_function, selection=slice(0, 200), y_top_lim=100
)

# Plot uncertainty estimates for MC Dropout-based recoding models
"""
mcd_recoding_selection_func = lambda path: "mcd" in path and "train" in path and "vanilla" not in path
mcd_recoding_log_paths = get_logs_in_dir(LOGDIR, mcd_recoding_selection_func)
mcd_recoding_logs = aggregate_logs(mcd_recoding_log_paths, name_function)
plot_column(
    mcd_recoding_logs, x_name="batch_num", y_names="deltas", intervals=False, save_path="img/deltas_mcd.png",
    title="Uncertainty estimates (n=3)", color_func=loss_color_function, selection=slice(0, 200)
)
"""

# Plot norms of recoding gradients for perplexity-based recoding model

def recoding_grad_color_func(curve_type, model_name, y_name):
    parts = y_name.split("_")
    grad_type, layer_num = parts[-2], int(parts[-1][1])
    return RECODING_GRAD_COLOR_DICT[grad_type][layer_num]

def recoding_grad_legend_func(model_name, y_name):
    parts = y_name.split("_")
    return parts[-2] + "_" + parts[-1]

gradient_columns = ["recoding_grads_hx_l0",	"recoding_grads_cx_l0",	"recoding_grads_hx_l1",	"recoding_grads_cx_l1"]

plot_column(
    ppl_recoding_logs, x_name="batch_num", y_names=gradient_columns, intervals=False,
    save_path="img/gradient_norms_ppl.png", title="Perplexity recoding gradient norms (n=3)",
    color_func=recoding_grad_color_func, legend_func=recoding_grad_legend_func, y_label="Recoding grad norm",
    selection=slice(0, 200), y_top_lim=200
)

# Plot norms of recoding gradients for MC Dropout-based recoding model
"""
plot_column(
    mcd_recoding_logs, x_name="batch_num", y_names=gradient_columns, intervals=False,
    save_path="img/gradient_norms_mcd.png", title="MC Dropout Recoding gradient norms (n=3)",
    color_func=recoding_grad_color_func, legend_func=recoding_grad_legend_func, y_label="Recoding grad norm",
    selection=slice(0, 200), y_top_lim=0.5
)
"""
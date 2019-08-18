"""
Plot results for experiment 4.
"""

# PROJECT
from src.utils.plot import *

# Load logs, define functions
LOGDIR = "logs/exp4/"
IMGDIR = "img/exp4/"


# Define how to infer model names from paths
def name_function(path):
    shortened = path[:path.rfind("_") - 1].replace(LOGDIR, "")
    if shortened:
        shortened = shortened.split("_")[0]

    return NAME_DICT[shortened]


def recoded_out_name_function(path):
    if "red" in path:
        return "Perplexity (recoded out)"
    else:
        return "Perplexity"


def ppl_step_name_function(path):
    if "vanilla" in path:
        return "Vanilla"
    else:
        step_size = re.search("step(\d+)", path).group(1)
        return f"Perplexity (step={step_size})"


# Define how to color different model losses
def loss_color_function(curve_type, model_name, y_name):

    if model_name in COLOR_DICT[curve_type]:
        return COLOR_DICT[curve_type][model_name]
    else:
        return "blue"

# -----------------------------------------------------------------

# 1. Plot debugged MC Dropout recoding performance compare to baseline
train_selection_func1 = lambda path: "train" in path and not "ppl" in path
train_log_paths1 = get_logs_in_dir(LOGDIR, train_selection_func1)
train_logs1 = aggregate_logs(train_log_paths1, name_function)
plot_column(
    train_logs1, x_name="batch_num", y_names="batch_loss", intervals=False, save_path=f"{IMGDIR}train_losses_mcd.png",
    title="Train loss (n=3)", color_func=loss_color_function, selection=slice(0, 100), y_top_lim=11
)

# -----------------------------------------------------------------

# 2. Plot how computing the loss with or without recoded output activations differs
train_selection_func2 = lambda path: "train" in path and "ppl" in path and "step" not in path
train_log_paths2 = get_logs_in_dir(LOGDIR, train_selection_func2)
train_logs2 = aggregate_logs(train_log_paths2, recoded_out_name_function)
plot_column(
    train_logs2, x_name="batch_num", y_names="batch_loss", intervals=True,
    save_path=f"{IMGDIR}train_losses_recoded_out.png",
    title="Train loss (n=3)", color_func=loss_color_function, y_top_lim=11
)

# -----------------------------------------------------------------

# 3. Plot how using different recoding step sizes differs
train_selection_func3 = lambda path: "train" in path and ("ppl_corr_step" in path or "vanilla" in path)
train_log_paths3 = get_logs_in_dir(LOGDIR, train_selection_func3)
train_logs3 = aggregate_logs(train_log_paths3, ppl_step_name_function)
plot_column(
    train_logs3, x_name="batch_num", y_names="batch_loss", intervals=False,
    save_path=f"{IMGDIR}train_losses_steps.png",
    title="Train loss (n=3)", selection=slice(0, 2000), y_top_lim=11
)
val_selection_func3 = lambda path: "val" in path and ("ppl_step" in path or "vanilla" in path)
val_log_paths3 = get_logs_in_dir(LOGDIR, val_selection_func3)
val_logs3 = aggregate_logs(val_log_paths3, ppl_step_name_function)
plot_column(
    val_logs3, x_name="batch_num", y_names="val_ppl", intervals=False,
    save_path=f"{IMGDIR}val_losses_steps.png",
    title="Validation loss (n=3)", selection=slice(0, 28)
)

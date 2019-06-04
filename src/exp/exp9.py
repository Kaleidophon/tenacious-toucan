"""
Plot results of experiment 9.
"""

# STD
import re

# PROJECT
from src.utils.plot import *
from src.utils.log import *

# CONSTANTS
STEPSIZE_LOGDIR = "logs/exp9/step_sizes/"
SAMPLES_LOGDIR = "logs/exp9/num_samples/"
STEPSIZE_IMGDIR = "img/exp9/step_sizes/"
SAMPLES_IMGDIR = "img/exp9/num_samples/"
STEPSIZES_TO_IDX = {0.1: 1, 0.5: 2, 1: 3, 2: 4, 5: 5}
SAMPLES_TO_IDX = {1: 1, 2: 2, 5: 3, 10: 4, 25: 5, 50: 6}
MODEL_TYPES = {"ensemble": "BAE", "mcd": "MC Dropout", "perplexity": "Perplexity"}
MODEL_TYPE_TO_CMAP = {"ensemble": "Greens", "mcd": "Purples", "perplexity": "Oranges"}

# ##### STEP SIZE EXPERIMENTS #####


# Get the recoding step size from a log path
def step_size_name_func(path: str) -> str:
    matches = re.search("step(\d\.)?\d", path)
    step_size = matches.group(0).replace("step", "")
    return step_size


def step_size_legend_func(model_name, y_name):
    return f"step size = {model_name}"


# Use different shades of the same color for different step sizes
def step_size_color_func_generator(model_type_short):
    # Pick color map based on model type
    cmap = plt.get_cmap(MODEL_TYPE_TO_CMAP[model_type_short])
    step_size_color_dict = {
        str(step_size): cmap(float(idx) / len(STEPSIZES_TO_IDX))
        for step_size, idx in STEPSIZES_TO_IDX.items()
    }

    def step_size_color_func(curve_type, model_name, y_name):
        return step_size_color_dict[model_name]

    return step_size_color_func


for model_type_short, model_type in MODEL_TYPES.items():
    step_size_color_func = step_size_color_func_generator(model_type_short)

    # Plot training losses
    step_size_train_selection_func = lambda path: "train" in path and model_type_short in path
    step_size_train_log_paths = get_logs_in_dir(STEPSIZE_LOGDIR, step_size_train_selection_func)
    step_size_train_logs = aggregate_logs(step_size_train_log_paths, step_size_name_func)
    plot_column(
        step_size_train_logs, x_name="batch_num", y_names="batch_loss", intervals=True,
        save_path=f"{STEPSIZE_IMGDIR}{model_type_short}_train_losses.png",
        title=f"Train loss | {model_type} recoding (n=4)", color_func=step_size_color_func,
        legend_func=step_size_legend_func, selection=slice(0, 400), y_label="Training Loss"
    )

    # Plot validation losses
    step_size_val_selection_func = lambda path: "val" in path and model_type_short in path
    step_size_val_log_paths = get_logs_in_dir(STEPSIZE_LOGDIR, step_size_val_selection_func)
    step_size_val_logs = aggregate_logs(step_size_val_log_paths, step_size_name_func)
    plot_column(
        step_size_val_logs, x_name="batch_num", y_names="val_ppl", intervals=True,
        save_path=f"{STEPSIZE_IMGDIR}{model_type_short}_val_ppls.png",
        title=f"Validation perplexity | {model_type} recoding (n=4)", color_func=step_size_color_func,
        legend_func=step_size_legend_func, selection=slice(0, 50), y_label="Validation Loss"
    )


# ##### NUM SAMPLES EXPERIMENTS #####


# Get the recoding step size from a log path
def samples_name_func(path: str) -> str:
    matches = re.search("samples\d+", path)
    step_size = matches.group(0).replace("samples", "")
    return step_size


def samples_legend_func(model_name, y_name):
    return f"{model_name} samples"


# Use different shades of the same color for different step sizes
def samples_color_func_generator(model_type_short):
    # Pick color map based on model type
    cmap = plt.get_cmap(MODEL_TYPE_TO_CMAP[model_type_short])
    samples_color_dict = {
        str(samples): cmap(float(idx) / len(SAMPLES_TO_IDX))
        for samples, idx in SAMPLES_TO_IDX.items()
    }

    def samples_color_func(curve_type, model_name, y_name):
        return samples_color_dict[model_name]

    return samples_color_func


for model_type_short, model_type in MODEL_TYPES.items():
    if model_type_short == "perplexity": continue
    samples_color_func = samples_color_func_generator(model_type_short)

    # Plot training losses
    samples_train_selection_func = lambda path: "train" in path and model_type_short in path
    samples_train_log_paths = get_logs_in_dir(SAMPLES_LOGDIR, samples_train_selection_func)
    samples_train_logs = aggregate_logs(samples_train_log_paths, samples_name_func)
    plot_column(
        samples_train_logs, x_name="batch_num", y_names="batch_loss", intervals=True,
        save_path=f"{SAMPLES_IMGDIR}{model_type_short}_train_losses.png",
        title=f"Train loss | {model_type} recoding (n=4)", color_func=samples_color_func,
        legend_func=samples_legend_func, selection=slice(0, 400), y_label="Training Loss"
    )

    # Plot validation losses
    samples_val_selection_func = lambda path: "val" in path and model_type_short in path
    samples_val_log_paths = get_logs_in_dir(SAMPLES_LOGDIR, samples_val_selection_func)
    samples_val_logs = aggregate_logs(samples_val_log_paths, samples_name_func)
    plot_column(
        samples_val_logs, x_name="batch_num", y_names="val_ppl", intervals=True,
        save_path=f"{SAMPLES_IMGDIR}{model_type_short}_val_ppls.png",
        title=f"Validation perplexity | {model_type} recoding (n=4)", color_func=samples_color_func,
        legend_func=samples_legend_func, selection=slice(0, 50), y_label="Validation Loss"
    )

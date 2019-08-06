"""
Plot results of experiment 9.
"""

# STD
import re

# PROJECT
from src.utils.plot import *
from src.utils.log import *

# CONSTANTS
STEPSIZE_LOGDIR = "logs/exp10/step_sizes/"
SAMPLES_LOGDIR = "logs/exp10/num_samples/"
DROPOUT_LOGDIR = "logs/exp10/dropout/"
LEARNED_LOGDIR = "logs/exp10/learned_steps/"
STEPSIZE_IMGDIR = "img/exp10/step_sizes/"
SAMPLES_IMGDIR = "img/exp10/num_samples/"
DROPOUT_IMGDIR = "img/exp10/dropout/"
LEARNED_IMGDIR = "img/exp10/learned/"
STEPSIZES_TO_IDX1 = {
    0.1: 1, 0.5: 2, 1: 3, 2: 4, 5: 5, 10: 6, 25: 7, 50: 8, 100: 9, 1000: 10
}
STEPSIZES_TO_IDX2 = {
    0.0001: 1, 0.001: 2, 0.01: 3, 0.1: 4, 0.5: 5, 1: 6, 2: 7, 5: 8
}
SAMPLES_TO_IDX = {1: 1, 2: 2, 5: 3, 10: 4, 25: 5, 50: 6}
DROPOUT_TO_IDX = {0.1: 1, 0.2: 2, 0.3: 3, 0.4: 4, 0.5: 5, 0.6: 6}
HIDDEN_TO_IDX = {"hx_l0": 1, "cx_l0": 2, "hx_l1": 3, "cx_l1": 4}
MODEL_TYPES = {"ensemble": "BAE", "mcd": "MC Dropout", "perplexity": "Surprisal"} #, "variational": "Variational"}
MODEL_TYPE_TO_CMAP = {"ensemble": "Greens", "mcd": "Purples", "perplexity": "Oranges"} #, "variational": "Blues"}

# ##### STEP SIZE EXPERIMENTS #####


# Get the recoding step size from a log path
def step_size_name_func(path: str) -> str:
    matches = re.search("step(\d\.)?\d+", path)
    step_size = matches.group(0).replace("step", "")
    return step_size


def step_size_legend_func(model_name, y_name):
    return f"step size = {model_name}"


# Use different shades of the same color for different step sizes
def step_size_color_func_generator(model_type_short):
    # Pick color map based on model type
    cmap = plt.get_cmap(MODEL_TYPE_TO_CMAP[model_type_short])
    size2idx = STEPSIZES_TO_IDX1 if model_type_short == "perplexity" else STEPSIZES_TO_IDX2

    step_size_color_dict = {
        str(step_size): cmap(float(idx) / len(size2idx))
        for step_size, idx in size2idx.items()
    }

    def step_size_color_func(curve_type, model_name, y_name):
        return step_size_color_dict[model_name]

    return step_size_color_func


for model_type_short, model_type in MODEL_TYPES.items():
    step_size_color_func = step_size_color_func_generator(model_type_short)

    # Plot training losses
    step_size_train_selection_func = \
        lambda path: "train" in path and model_type_short in path and "step" in path
    step_size_train_log_paths = get_logs_in_dir(STEPSIZE_LOGDIR, step_size_train_selection_func)
    step_size_train_logs = aggregate_logs(step_size_train_log_paths, step_size_name_func)
    plot_column(
        step_size_train_logs, x_name="batch_num", y_names="batch_loss", intervals=True,
        save_path=f"{STEPSIZE_IMGDIR}{model_type_short}_train_losses.png",
        #title=f"Train loss | {model_type} recoding (n=4)",
        color_func=step_size_color_func,
        legend_func=step_size_legend_func, selection=slice(0, 400),
        y_label="Training Loss", x_label="# Batches", y_top_lim=(24 if model_type_short != "perplexity" else 12)
    )

    # Plot validation losses
    step_size_val_selection_func = \
        lambda path: "val" in path and model_type_short in path and "step" in path
    step_size_val_log_paths = get_logs_in_dir(STEPSIZE_LOGDIR, step_size_val_selection_func)
    step_size_val_logs = aggregate_logs(step_size_val_log_paths, step_size_name_func)
    plot_column(
        step_size_val_logs, x_name="batch_num", y_names="val_ppl", intervals=True,
        save_path=f"{STEPSIZE_IMGDIR}{model_type_short}_val_ppls.png",
        #title=f"Validation perplexity | {model_type} recoding (n=4)",
        color_func=step_size_color_func,
        legend_func=step_size_legend_func, selection=slice(0, 50),
        y_label="Validation Perplexity", x_label="# Batches"
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
        #title=f"Train loss | {model_type} recoding (n=4)",
        color_func=samples_color_func,
        legend_func=samples_legend_func, selection=slice(0, 400),
        y_label="Training Loss", x_label="# Batches"
    )

    # Plot deltas
    plot_column(
        samples_train_logs, x_name="batch_num", y_names="deltas", intervals=True,
        save_path=f"{SAMPLES_IMGDIR}{model_type_short}_deltas.png",
        # title=f"Train loss | {model_type} recoding (n=4)",
        color_func=samples_color_func,
        legend_func=samples_legend_func, selection=slice(825, 850),
        y_label="Deltas", x_label="# Batches"
    )

    # Plot validation losses
    samples_val_selection_func = lambda path: "val" in path and model_type_short in path
    samples_val_log_paths = get_logs_in_dir(SAMPLES_LOGDIR, samples_val_selection_func)
    samples_val_logs = aggregate_logs(samples_val_log_paths, samples_name_func)
    plot_column(
        samples_val_logs, x_name="batch_num", y_names="val_ppl", intervals=True,
        save_path=f"{SAMPLES_IMGDIR}{model_type_short}_val_ppls.png",
        #title=f"Validation perplexity | {model_type} recoding (n=4)",
        color_func=samples_color_func,
        legend_func=samples_legend_func, selection=slice(0, 50),
        y_label="Validation Perplexity", x_label="# Batches"
    )


# ##### DROPOUT EXPERIMENTS #####

# Get the recoding step size from a log path
def dropout_name_func(path: str) -> str:
    matches = re.search("dropout\d\.\d+", path)
    step_size = matches.group(0).replace("dropout", "")
    return step_size


def dropout_legend_func(model_name, y_name):
    return f"dropout = {model_name}"


# Use different shades of the same color for different dropout razes
def dropout_color_func_generator():
    # Pick color map based on model type
    cmap = plt.get_cmap(MODEL_TYPE_TO_CMAP["mcd"])
    dropout_color_dict = {
        str(samples): cmap(float(idx) / len(SAMPLES_TO_IDX))
        for samples, idx in DROPOUT_TO_IDX.items()
    }

    def dropout_color_func(curve_type, model_name, y_name):
        return dropout_color_dict[model_name]

    return dropout_color_func

# Plot train losses
dropout_train_selection_func = lambda path: "train" in path
dropout_train_log_paths = get_logs_in_dir(DROPOUT_LOGDIR, dropout_train_selection_func)
dropout_train_logs = aggregate_logs(dropout_train_log_paths, dropout_name_func)
plot_column(
    dropout_train_logs, x_name="batch_num", y_names="batch_loss", intervals=True,
    save_path=f"{DROPOUT_IMGDIR}mcd_train_losses.png",
    #title=f"Train loss | {model_type} recoding (n=4)",
    color_func=dropout_color_func_generator(),
    legend_func=dropout_legend_func, selection=slice(0, 400),
    y_label="Training Loss", x_label="# Batches"
)

# Plot deltas
plot_column(
    dropout_train_logs, x_name="batch_num", y_names="deltas", intervals=True,
    save_path=f"{DROPOUT_IMGDIR}mcd_deltas.png",
    # title=f"Train loss | {model_type} recoding (n=4)",
    color_func=dropout_color_func_generator(),
    legend_func=dropout_legend_func, selection=slice(2525, 2550),
    y_label="Deltas", x_label="# Batches"
)


# Plot validation losses
dropout_val_selection_func = lambda path: "val" in path
dropout_val_log_paths = get_logs_in_dir(DROPOUT_LOGDIR, dropout_val_selection_func)
samples_val_logs = aggregate_logs(dropout_val_log_paths, dropout_name_func)
plot_column(
    samples_val_logs, x_name="batch_num", y_names="val_ppl", intervals=True,
    save_path=f"{DROPOUT_IMGDIR}mcd_val_ppls.png",
    #title=f"Validation perplexity | {model_type} recoding (n=4)",
    color_func=dropout_color_func_generator(),
    legend_func=dropout_legend_func, selection=slice(0, 50),
    y_label="Validation Perplexity", x_label="# Batches"
)


# #### LEARNED STEP SIZE EXPERIMENTS ####

# Use latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def step_legend_func(model_name, y_name):
    y_name = y_name.replace("step_sizes_", "")
    y_name = y_name.replace("hx_", "$\mathbf{h}_t$ ")
    y_name = y_name.replace("cx_", "$\mathbf{c}_t$ ")
    y_name = y_name.replace("l0", "$1^{st}$ layer")
    y_name = y_name.replace("l1", "$2^{nd}$ layer")

    return y_name


def step_name_func(path: str) -> str:
    return path[:path.find("_")+1]


for model_type_short, model_type in MODEL_TYPES.items():
    def step_color_func_generator():
        # Pick color map based on model type
        cmap = plt.get_cmap(MODEL_TYPE_TO_CMAP[model_type_short])
        step_color_dict = {
            str(samples): cmap(float(idx) / len(HIDDEN_TO_IDX))
            for samples, idx in HIDDEN_TO_IDX.items()
        }

        def step_color_func(curve_type, model_name, y_name):
            return step_color_dict[y_name.replace("step_sizes_", "")]

        return step_color_func

    learned_step_selection_func = lambda path: "train" in path and "_learned" in path and model_type_short in path
    learned_step_log_paths = get_logs_in_dir(LEARNED_LOGDIR, learned_step_selection_func)
    learned_step_logs = aggregate_logs(learned_step_log_paths, step_name_func)
    plot_column(
        learned_step_logs, x_name="batch_num",
        y_names=["step_sizes_hx_l0", "step_sizes_cx_l0", "step_sizes_hx_l1", "step_sizes_cx_l1"], intervals=False,
        save_path=f"{LEARNED_IMGDIR}{model_type_short}_learned_steps.png",
        color_func=step_color_func_generator(),
        legend_func=step_legend_func,
        y_label="Learned Step size", x_label="\# Batches"
    )

    # #### PREDICTED STEP SIZE EXPERIMENTS ####

    predicted_step_selection_func = lambda path: "train" in path and "_mlp" in path and model_type_short in path
    predicted_step_log_paths = get_logs_in_dir(LEARNED_LOGDIR, predicted_step_selection_func)
    predicted_step_logs = aggregate_logs(predicted_step_log_paths, step_name_func)
    plot_column(
        predicted_step_logs, x_name="batch_num",
        y_names=["step_sizes_hx_l0", "step_sizes_cx_l0", "step_sizes_hx_l1", "step_sizes_cx_l1"], intervals=False,
        save_path=f"{LEARNED_IMGDIR}{model_type_short}_predicted_steps.png",
        color_func=step_color_func_generator(),
        legend_func=step_legend_func,
        y_label="Predicted Step size", x_label="\# Batches"
    )



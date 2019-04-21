"""
Define functions to plot loss curves and other noteworthy model characteristics.
"""

# STD
from collections import OrderedDict
from typing import Optional, Tuple, Union, List, Callable

# EXT
import numpy as np
from matplotlib import pyplot as plt

# PROJECT
from src.utils.log import get_logs_in_dir, aggregate_logs
from src.utils.types import ColorDict, AggregatedLogs

# CONSTANTS
# Define some standard colors for models used in this project
COLOR_DICT = {
    "curve": {
        "Vanilla": "firebrick",
        "Perplexity": "darkorange",
        "MC Dropout": "darkmagenta"
    },
    "interval": {
        "Vanilla": "lightcoral",
        "Perplexity": "orange",
        "MC Dropout": "darkorchid"
    }
}

# Define colors for different kinds of recoding gradients
RECODING_GRAD_COLOR_DICT = {
    "hx": ["midnightblue", "royalblue", "lightskyblue"],
    "cx": ["darkslategrey", "forestgreen", "darkseagreen"],
}

# TODO: Incorporate options to distinguish between fixed / mlp step
NAME_DICT = {
    "mcd": "MC Dropout",
    "ppl": "Perplexity",
    "vanilla": "Vanilla"
}


def plot_column(logs: AggregatedLogs, x_name: str, y_names: Union[str, List[str]], intervals: bool = True,
                title: Optional[str] = None, save_path: Optional[str] = None, color_func: Optional[Callable] = None,
                selection: Optional[slice] = None, y_label: Optional[str] = None,
                legend_func: Callable = lambda model_name, y_name: model_name) -> None:
    """
    Plot data in an aggregated dict either as a series of curves or as curves with uncertainty intervals when given
    multiple data points and corresponding flag turned on.

    Parameters
    ----------
    logs: AggregatedLogs
        Data from logs in a structured form.
    x_name: str
        Name of the data column that should be used for the x-axis.
    y_names: Union[str, List[str]]
        Name or list of names of the data column(s) that should be used for the y-axis.
    intervals: bool
        Indicate whether in the case of data for multiple curves these curves should be plotted separately or if they
        should be used to to display the value range of multiple runs as an interval of plus/minus one standard
        deviation and the mean.
    title: Optional[str]
        Optional title for the plot.
    save_path: Optional[str]
        Path to save the plot to. If not given, plot will be shown on screen.
    colors: Optional[ColorDict]
        Optional dictionary defining the colors of the curves for different models. Should contain sub-dicts with
        corresponding keys "curves" and "interval". Otherwise matplotlib color choices are used, however, color use
        is still consistent for curves belonging to the same model.
    selection: Optional[slice]
        Select a range of the data for plotting via a slice object.
    """
    x = list(logs.values())[0][x_name]
    x = x.astype(np.int)

    # If data for x-axis has been aggregated, only select one data row
    if x.shape[0] != 1 or len(x.shape) > 1:
        x = x[0]

    # If data is not truncated in any way, select everything
    if selection is None:
        selection = slice(x.shape[0])

    x = x[selection]

    if type(y_names) == str:
        y_names = [y_names]

    for model_name, log_dict in logs.items():
        for y_name in y_names:
            y = log_dict[y_name]

            # Single curve data
            if y.shape[0] == 1 or len(y.shape) == 1:
                color = None if color_func is None else color_func("curve", model_name, y_name)
                plt.plot(x, y[selection], label=legend_func(model_name, y_name), color=color)

            # Data from more than one curves
            else:
                # Select color if given or use matplotlib's choice
                color = None if color_func is None else color_func("curve", model_name, y_name)

                # Plot with intervals (± one std dev)
                if intervals:
                    low, mean, high = _get_intervals(y)
                    low, mean, high = low[selection], mean[selection], high[selection]
                    curve = plt.plot(x, mean, label=legend_func(model_name, y_name), color=color)

                    # Make sure the color is the same for the intervals or choose pre-defined
                    color = curve[0].get_color() if color_func is None else color_func("interval", model_name, y_name)
                    plt.fill_between(x, high, mean, alpha=0.6, color=color)
                    plt.fill_between(x, mean, low, alpha=0.6, color=color)

                # Just plot all the curves separately
                else:
                    for i, y_i in enumerate(y):
                        curve = plt.plot(
                            x, y_i[selection], label=legend_func(model_name, y_name), alpha=0.6, color=color
                        )
                        # Make sure color is consistent for all curve belonging to the same model type
                        color = curve[0].get_color()

    # Add additional information to plot
    plt.xlabel(x_name)
    plt.ylabel(y_label if y_label is not None else y_names[0])

    # Avoid having the same labels multiple times in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    if title is not None:
        plt.title(title)

    plt.tight_layout()

    # Output as file or on screen and close
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

    plt.close()


def _get_intervals(data: np.array) -> Tuple[np.array, np.array, np.array]:
    """
    Returns the intervals (± one std dev) and mean for a number of curves.

    Parameters
    ----------
    data: np.array
        Data to get the curves for.

    Returns
    -------
    low, mean, high: Tuple[np.array, np.array, np.array]
        The data for the curves corresponding to mean - std dev, mean and mean + std dev
    """
    assert len(data.shape) == 2, f"Intervals require two-dimensional data, {len(data.shape)} dimensions found."

    mean = data.mean(axis=0)
    std = data.std(axis=0)
    low, high = mean - std, mean + std

    return low, mean, high


if __name__ == "__main__":
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
        title="Train loss (n=5)", color_func=loss_color_function, selection=slice(0, 200)
    )

    # Plot validation losses
    val_selection_func = lambda path: "val" in path
    val_log_paths = get_logs_in_dir(LOGDIR, val_selection_func)
    val_logs = aggregate_logs(val_log_paths, name_function)
    plot_column(
        val_logs, x_name="batch_num", y_names="val_ppl", intervals=False, save_path="img/val_ppls.png",
        title="Validation perplexity (n=5)", color_func=loss_color_function #,selection=slice(0, 20)
    )

    # ### Plot additional information ###

    # Plot uncertainty estimates for perplexity-based recoding model
    ppl_recoding_selection_func = lambda path: "ppl" in path and "train" in path and "vanilla" not in path
    ppl_recoding_log_paths = get_logs_in_dir(LOGDIR, ppl_recoding_selection_func)
    ppl_recoding_logs = aggregate_logs(ppl_recoding_log_paths, name_function)
    plot_column(
        ppl_recoding_logs, x_name="batch_num", y_names="deltas", intervals=False, save_path="img/deltas_ppl.png",
        title="Uncertainty estimates (n=5)", color_func=loss_color_function, selection=slice(0, 200)
    )

    # Plot uncertainty estimates for MC Dropout-based recoding models
    mcd_recoding_selection_func = lambda path: "mcd" in path and "train" in path and "vanilla" not in path
    mcd_recoding_log_paths = get_logs_in_dir(LOGDIR, mcd_recoding_selection_func)
    mcd_recoding_logs = aggregate_logs(mcd_recoding_log_paths, name_function)
    plot_column(
        mcd_recoding_logs, x_name="batch_num", y_names="deltas", intervals=False, save_path="img/deltas_mcd.png",
        title="Uncertainty estimates (n=5)", color_func=loss_color_function, selection=slice(0, 200)
    )

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
        save_path="img/gradient_norms_ppl.png", title="Perplexity recoding gradient norms (n=5)",
        color_func=recoding_grad_color_func, legend_func=recoding_grad_legend_func, y_label="Recoding grad norm",
        selection=slice(0, 200)
    )

    # Plot norms of recoding gradients for MC Dropout-based recoding model
    plot_column(
        mcd_recoding_logs, x_name="batch_num", y_names=gradient_columns, intervals=False,
        save_path="img/gradient_norms_mcd.png", title="MC Dropout Recoding gradient norms (n=5)",
        color_func=recoding_grad_color_func, legend_func=recoding_grad_legend_func, y_label="Recoding grad norm",
        selection=slice(0, 200)
    )

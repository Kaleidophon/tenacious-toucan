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
        "MC Dropout": "darkmagenta",
        "Anc. Ensemble": "forestgreen",
    },
    "interval": {
        "Vanilla": "lightcoral",
        "Perplexity": "orange",
        "MC Dropout": "darkorchid",
        "Anc. Ensemble": "lightgreen"
    }
}

# Define colors for different kinds of recoding gradients
RECODING_GRAD_COLOR_DICT = {
    "hx": ["midnightblue", "royalblue", "lightskyblue"],
    "cx": ["darkslategrey", "forestgreen", "darkseagreen"],
}

# TODO: Incorporate options to distinguish between fixed / mlp step
NAME_DICT = {
    "ens": "Anc. Ensemble",
    "mcd": "MC Dropout",
    "ppl": "Perplexity",
    "vanilla": "Vanilla"
}


def plot_column(logs: AggregatedLogs, x_name: str, y_names: Union[str, List[str]], intervals: bool = True,
                title: Optional[str] = None, save_path: Optional[str] = None, color_func: Optional[Callable] = None,
                selection: Optional[slice] = None, y_label: Optional[str] = None,
                legend_func: Callable = lambda model_name, y_name: model_name, y_top_lim: Optional[float] = None) -> None:
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
    color_func: Optional[Callable]
        Define a function that picks a curve color based on the model name and data column name. Otherwise matplotlib
        color choices are used, however, color use is still consistent for curves belonging to the same model.
    selection: Optional[slice]
        Select a range of the data for plotting via a slice object.
    y_label: Optional[str]
        Explicitly define the y-axis label.
    legend_func: Callable
        Define an optional function that determines the curve label inside the legend based on the model name and the
        data column name.
    y_top_lim: Optional[float]
        Optional upper limit for y-axis.
    """
    x = list(logs.values())[0][x_name]
    x = x.astype(np.int)

    # If data for x-axis has been aggregated, only select one data row
    if (x.shape[0] != 1 and len(x.shape) == 2) or len(x.shape) > 1:
        x = x[0]

    # If data is not truncated in any way, select everything
    if selection is None:
        selection = slice(x.shape[0])

    x = x[selection]

    if type(y_names) == str:
        y_names = [y_names]

    all_y = []
    for model_name, log_dict in logs.items():
        for y_name in y_names:
            y = log_dict[y_name]
            all_y.append(y)

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
                            x, y_i[selection], label=legend_func(model_name, y_name), alpha=0.4, color=color
                        )
                        # Make sure color is consistent for all curve belonging to the same model type
                        color = curve[0].get_color()

    # Add additional information to plot
    plt.xlabel(x_name)
    plt.ylabel(y_label if y_label is not None else y_names[0])

    # Zoom in to majority of points to avoid distortion by outliers
    if y_top_lim is not None:
        bottom, _ = plt.ylim()
        plt.ylim(top=y_top_lim, bottom=max(bottom, 0))

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

"""
Define functions to plot loss curves and other noteworthy model characteristics.
"""

# STD
from collections import OrderedDict
from typing import Optional, Tuple, Union, List, Callable
import re

# EXT
import numpy as np
from matplotlib import pyplot as plt

# PROJECT
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

NAME_DICT = {
    "ensemble": "Anc. Ensemble",
    "mcd": "MC Dropout",
    "ppl": "Perplexity",
    "vanilla": "Vanilla"
}


def plot_column(logs: AggregatedLogs,
                x_name: str,
                y_names: Union[str, List[str]],
                intervals: bool = True,
                title: Optional[str] = None,
                save_path: Optional[str] = None,
                color_func: Optional[Callable] = None,
                selection: Optional[slice] = None,
                x_label: Optional[str] = None,
                y_label: Optional[str] = None,
                legend_func: Callable = lambda model_name, y_name: model_name,
                y_top_lim: Optional[float] = None) -> None:
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
    x_label: Optional[str]
        Explicitly define the x-axis label.
    y_label: Optional[str]
        Explicitly define the y-axis label.
    legend_func: Callable
        Define an optional function that determines the curve label inside the legend based on the model name and the
        data column name.
    y_top_lim: Optional[float]
        Optional upper limit for y-axis.
    """
    # Print soft error message if logs are empty
    if len(logs) == 0:
        description = ""
        if (title, save_path) != (None, None):
            description = " to create {title}".format(title=title if title is not None else save_path)

        print(f"No logs found{description}, skip.")
        return

    # Select values for x-axis
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

    def _missing_mask(y: np.array, x: np.array) -> np.array:
        """ Return a mask that is being applied to missing values. """
        missing = x.shape[0] - y.shape[0]
        mask = np.ones(y.shape).astype(bool)

        if missing > 0:
            mask = np.concatenate((mask, np.zeros((missing, )).astype(bool)), axis=0)

        return mask

    all_y = []
    for model_name, log_dict in logs.items():
        for y_name in y_names:
            y = log_dict[y_name]
            all_y.append(y)

            # Single curve data
            if y.shape[0] == 1 or len(y.shape) == 1:
                color = None if color_func is None else color_func("curve", model_name, y_name)
                y = y[selection]
                mask = _missing_mask(y, x)
                plt.plot(x[mask], y, label=legend_func(model_name, y_name), color=color)

            # Data from more than one curves
            else:
                # Select color if given or use matplotlib's choice
                color = None if color_func is None else color_func("curve", model_name, y_name)

                # Plot with intervals (± one std dev)
                if intervals:
                    low, mean, high = _get_intervals(y)
                    low, mean, high = low[selection], mean[selection], high[selection]
                    mask = _missing_mask(low, x)
                    curve = plt.plot(x[mask], mean, label=legend_func(model_name, y_name), color=color)

                    # Make sure the color is the same for the intervals or choose pre-defined
                    color = curve[0].get_color() if color_func is None else color_func("interval", model_name, y_name)
                    plt.fill_between(x[mask], high, mean, alpha=0.6, color=color)
                    plt.fill_between(x[mask], mean, low, alpha=0.6, color=color)

                # Just plot all the curves separately
                else:
                    for i, y_i in enumerate(y):
                        y_i = y_i[selection]
                        mask = _missing_mask(y_i, x)
                        curve = plt.plot(
                            x[mask], y_i, label=legend_func(model_name, y_name), alpha=0.7, color=color
                        )
                        # Make sure color is consistent for all curve belonging to the same model type
                        color = curve[0].get_color()

    # Add additional information to plot
    plt.xlabel(x_name if x_label is None else x_label)
    plt.ylabel(y_label if y_label is not None else y_names[0])

    # Zoom in to majority of points to avoid distortion by outliers
    if y_top_lim is not None:
        bottom, _ = plt.ylim()
        plt.ylim(top=y_top_lim, bottom=max(bottom, 0))

    # Avoid having the same labels multiple times in the legend and sort them alphabetically
    handles, labels = plt.gca().get_legend_handles_labels()
    # Look for number in label and sort by that
    short_labels = [float(re.search("(\d\.)?\d+", label).group(0)) for label in labels]
    labels, short_labels, handles = zip(*sorted(zip(labels, short_labels, handles), key=lambda t: t[1]))
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right")

    if title is not None:
        plt.title(title)

    plt.tight_layout()

    # Output as file or on screen and close
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300)

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

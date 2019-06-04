"""
Define common functions when plotting experiments.
"""

# PROJECT
from src.utils.plot import NAME_DICT, COLOR_DICT


def name_function(path, log_dir):
    shortened = path[:path.rfind("_") - 1].replace(log_dir, "")

    shortened = shortened.split("_")[0]
    return NAME_DICT[shortened]


# Define how to color different model losses
def loss_color_function(curve_type, model_name, y_name):
    return COLOR_DICT[curve_type][model_name]

"""
Define functions to plot loss curves and other noteworthy model characteristics.
"""

# PROJECT
from utils.log import get_logs_in_dir, aggregate_logs


if __name__ == "__main__":
    LOGDIR = "logs/"
    name_function = lambda path: path[:path.rfind("_") - 1]
    train_selection_func = lambda path: "train" in path

    log_paths = get_logs_in_dir(LOGDIR, train_selection_func)
    logs = aggregate_logs(log_paths, name_function)
    ...

"""
Analyze the correlation between two series of data extracted from logs.
"""

# STD
from argparse import ArgumentParser
import re

# EXT
from diagnnose.config.setup import ConfigSetup
from mlxtend.evaluate import permutation_test
import numpy as np
from scipy.stats import spearmanr

# PROJECT
from src.utils.log import get_logs_in_dir, aggregate_logs


def main():
    config_dict = manage_config()

    # Read in logs and grab corresponding column
    first_log_dir, first_name, first_column = config_dict["targets"]["first"]
    second_log_dir, second_name, second_column = config_dict["targets"]["second"]

    mode = config_dict["general"]["mode"]

    def name_func_generator(log_dir):
        clean = lambda path: path.replace(log_dir, "")
        name_func = lambda path: clean(path)[:re.search("\d", clean(path)).start() - 1]
        return name_func

    # Select the relevant logs
    first_logs = aggregate_logs(
        get_logs_in_dir(first_log_dir, selection_func=lambda path: first_name in path and "train" in path),
        name_func=name_func_generator(first_log_dir)
    )
    second_logs = aggregate_logs(
        get_logs_in_dir(second_log_dir, selection_func=lambda path: second_name in path and "train" in path),
        name_func=name_func_generator(second_log_dir)
    )
    # Select relevant columns
    first_data, second_data = list(first_logs.values())[0][first_column], list(second_logs.values())[0][second_column]

    # Perform analysis
    print(correlation_analysis(first_data, second_data, mode=mode))


def correlation_analysis(first_data: np.array, second_data: np.array, mode: str="mse") -> (float, float):
    """
    Perform some analysis to determine whether to series of measurements are correlated. If the measurements for
    multiple runs are supplied, this analysis is performed for the respective means.

    This can be done by choosing between different approaches ("mode"):

    - mse: Determine the Minimum Squared Error between the curves, i.e. the distance.*
    - spearman: Compute Spearman's rank-order correlation coefficient between measurements.*
    - cross: Compute the cross-correlation between the two series.

    * Order of measurements does not matter.

    Return the score of the analysis including the p-value (for mse and cross this is based on the permutation test).

    :param first_data: Numpy array of measurements of form N x L (N runs and L data points).
    :param second_data: Numpy array of measurements of form N x L (N runs and L data points).
    :param mode: Analysis mode, choice between (mse, spearman, cross).
    :return: Result of analysis and p-value.
    """
    correlation_funcs = {
        "mse": lambda a, b: (np.power((a - b), 2).sum(), permutation_test(a, b, method='approximate', num_rounds=1000)),
        "spearman": spearmanr,
        "cross": lambda a, b: (np.correlate(a, b), permutation_test(a, b, method='approximate', num_rounds=1000))
    }
    assert mode in correlation_funcs.keys(), \
        f"Invalid mode {mode} found, choose one of [{', '.join(correlation_funcs.keys())}]."

    # If multiple runs are available, compare means
    if len(first_data.shape) == 2 and first_data.shape[0] > 1:
        first_data = first_data.mean(axis=0)

    if len(second_data.shape) == 2 and second_data.shape[0] > 1:
        second_data = second_data.mean(axis=0)

    return correlation_funcs[mode](first_data, second_data)


def manage_config() -> dict:
    """
    Parse a config file (if given), overwrite with command line arguments and return everything as dictionary
    of different config groups.
    """
    required_args = {"first", "second"}
    arg_groups = {
        "general": {"device", "mode"},
        "targets": {"first", "second"}
    }
    argparser = init_argparser()
    config_object = ConfigSetup(argparser, required_args, arg_groups)
    config_dict = config_object.config_dict

    return config_dict


def init_argparser() -> ArgumentParser:
    """
    Init the parser that parses additional command line arguments. These overwrite the default options
    that were defined in a config if -c / --config is given.
    """
    parser = ArgumentParser()
    from_config = parser.add_argument_group('From config file', 'Provide full experiment setup via config file')
    from_config.add_argument('-c', '--config', help='Path to json file containing classification config.')
    from_cmd = parser.add_argument_group('From commandline', 'Specify experiment setup via commandline arguments')

    from_cmd.add_argument("--device", type=str, default="cpu", help="Device used for training.")
    from_cmd.add_argument("--first", nargs=3, type=str,
                          help="First series of data in the form of <log_dir> <model_name> <data_column>")
    from_cmd.add_argument("--second", nargs=3, type=str,
                          help="Second series of data in the form of <log_dir> <model_name> <data_column>")
    from_cmd.add_argument("--mode", type=str, choices=["mse", "spearman", "cross"], default="mse",
                          help="Analysis mode")

    return parser


if __name__ == "__main__":
    main()

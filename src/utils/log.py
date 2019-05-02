"""
Define helpful logging functions.
"""

# STD
from collections import defaultdict
from functools import wraps
from genericpath import isfile
from os import listdir
from os.path import join
from typing import Optional, List, Callable
import os

# EXT
import numpy as np
import torch

# PROJECT
from src.utils.types import LogDict, AggregatedLogs, StepSize


class StatsCollector:
    """
    Class that is used to retrieve some interesting information to plot deep from some class, modifying the least
    amount of code from the class as possible. In order to achieve this, we decorate function with desirable arguments
    and return values with class methods of this class. This way the information can stored easily in a static version
    of this class.
    """
    _stats = {}

    @classmethod
    def collect_deltas(cls, func) -> Callable:
        """
        Decorate the compute_recoding_gradient() function of the recoding mechanism and collect information about the
        error signals.
        """
        @wraps(func)
        def wrapper(delta: torch.Tensor, *args) -> None:
            if "delta" not in cls._stats.keys():
                cls._stats["deltas"] = []

            cls._stats["deltas"].append(delta)
            func(delta, *args)

        return wrapper

    @classmethod
    def collect_recoding_gradients(cls, func) -> Callable:
        """
        Decorate the recode() function of the recoding mechanism in order to collect data about the recoding gradients.
        """
        @wraps(func)
        def wrapper(self, hidden: torch.Tensor, step_size: StepSize, name: Optional[str]) -> torch.Tensor:
            grad_norm = torch.norm(hidden.recoding_grad)

            if name is None:
                if "recoding_grads" not in cls._stats.keys():
                    cls._stats["recoding_grads"], cls._stats["step_sizes"] = [], []

                cls._stats["recoding_grads"].append(grad_norm)
                cls._stats["step_sizes"].append(step_size)
            else:
                if "recoding_grads" not in cls._stats.keys():
                    cls._stats["recoding_grads"], cls._stats["step_sizes"] = defaultdict(list), defaultdict(list)

                cls._stats["recoding_grads"][name].append(grad_norm)
                cls._stats["step_sizes"][name].append(step_size)

            return func(self, hidden, step_size, name)

        return wrapper

    @classmethod
    def _reduce_deltas(cls, deltas: List[torch.Tensor]):
        deltas = torch.stack(deltas)
        mean_delta = torch.flatten(deltas).mean()

        return mean_delta.item()

    @classmethod
    def _reduce_recoding_gradients(cls, recoding_gradients: List[torch.Tensor]):
        recoding_gradients = torch.stack(recoding_gradients)
        mean_grad_norm = recoding_gradients.mean()

        return mean_grad_norm.item()

    @classmethod
    def _reduce_step_sizes(cls, step_sizes: List[StepSize]):
        step_sizes = torch.stack(step_sizes)
        mean_step_size = torch.flatten(step_sizes).mean()

        return mean_step_size.item()

    @classmethod
    def reduce(cls) -> dict:
        """
        Reduce collected data in a way that it can be written easily into a log.
        """
        reduction_funcs = {
            "deltas": cls._reduce_deltas,
            "recoding_grads": cls._reduce_recoding_gradients,
            "step_sizes": cls._reduce_step_sizes
        }

        reduced_stats = {}

        for stat, data in cls._stats.items():
            if type(data) in (dict, defaultdict):  # Nested stats
                reduced_stats[stat] = {}
                for name in data.keys():
                    reduced_stats[stat][name] = reduction_funcs[stat](cls._stats[stat][name])
            else:
                reduced_stats[stat] = reduction_funcs[stat](cls._stats[stat])

        return reduced_stats

    @classmethod
    def get_stats(cls) -> dict:
        """
        Return all the collected statistics in a way that is easy to write into a log.
        """
        reduced_stats = cls.reduce()

        return reduced_stats

    @classmethod
    def wipe(cls) -> None:
        """
        Release all collected information.
        """
        cls._stats = {}

    @staticmethod
    def flatten_stats(stats: dict) -> dict:
        """
        Flatten a dictionary of stats by removing any nesting.
        """
        flattened_stats = {}

        for key, value in stats.items():
            if type(value) in (dict, defaultdict):
                for value_name, val in value.items():
                    flattened_stats[f"{key}_{value_name}"] = val
            else:
                flattened_stats[key] = value

        return flattened_stats


def remove_logs(log_dir: str, model_name: str) -> None:
    """
    Remove logs from previous runs if necessary.

    Parameters
    ----------
    log_dir: str
        Logging directory in which to look for out logs.
    model_name: str
        Name of the model the logs belong to.
    """
    train_log_path = f"{log_dir}/{model_name}_train.log"
    val_log_path = f"{log_dir}/{model_name}_val.log"

    # Remove logs from previous runs
    if log_dir is not None and os.path.exists(train_log_path):
        os.remove(train_log_path)

    if log_dir is not None and os.path.exists(val_log_path):
        os.remove(val_log_path)


def log_to_file(data: dict, log_path: Optional[str] = None) -> None:
    """
    Log some data to a normal text file.

    Parameters
    ----------
    data: Any
        Data to be logged.
    log_path: str
        File the data is going to be logged to (if given).
    """
    if log_path is None:
        return

    # If file doesn't exists, create it and write header columns
    if not os.path.exists(log_path):
        with open(log_path, "w") as log_file:
            columns = data.keys()
            log_file.write("\t".join(map(str, columns)) + "\n")
            log_file.write("\t".join(map(str, data.values())) + "\n")

    # If file already exists, write new data
    else:
        with open(log_path, "a") as log_file:
            log_file.write("\t".join(map(str, data.values())) + "\n")


def read_log(path: str) -> LogDict:
    """
    Read a log file into a dictionary.
    """
    data = defaultdict(lambda: np.array([]))

    with open(path, "r") as file:
        lines = file.readlines()
        headers, lines = lines[0].strip(), lines[1:]
        header_parts = headers.split()

        for line in lines:
            line_parts = line.strip().split()
            line_parts = map(float, line_parts)  # Cast to float

            for header, part in zip(header_parts, line_parts):
                data[header] = np.append(data[header], part)

    return data


def merge_logs(log1: LogDict, log2: LogDict) -> LogDict:
    """
    Merge two log dicts by concatenating the data columns
    """
    assert log1.keys() == log2.keys(), "Logs must have the same headers!"

    expand = lambda array: array if len(array.shape) == 2 else array[np.newaxis, ...]

    merged_log = {}

    for header in log1.keys():
        data1, data2 = expand(log1[header]), expand(log2[header])
        merged_data = np.concatenate([data1, data2], axis=0)
        merged_log[header] = merged_data

    return merged_log


def aggregate_logs(paths: List[str], name_func: Optional[Callable] = None) -> AggregatedLogs:
    """
    Aggregate the data from multiple logs into one LogDict. Requires the logs to have the same headers and the same
    number of data points.

    If multiple logs receive the same name via the naming function, merge the corresponding data.
    """
    def _default_name_func(path: str):
        model_name = path[:path.rfind("_")]

        return model_name

    name_func = name_func if name_func is not None else _default_name_func
    logs = {}

    for path in paths:
        name = name_func(path)
        log = read_log(path)

        # Merge data
        if name in logs:
            logs[name] = merge_logs(logs[name], log)
        else:
            logs[name] = log

    return logs


def get_logs_in_dir(log_dir: str, selection_func: Callable = lambda path: True) -> List[str]:
    """
    Select paths to log files in a directory based on some selection function that returns True of the log file matches
    some criterion.
    """
    p = lambda log_dir, path: join(log_dir, path)

    return [
        p(log_dir, path) for path in listdir(log_dir)
        if isfile(p(log_dir, path)) and selection_func(p(log_dir, path))
    ]

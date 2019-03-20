"""
Define helpful logging functions.
"""

# STD
from collections import defaultdict
from genericpath import isfile
from os import listdir
from os.path import join
from typing import Any, Optional, Union, List, Callable, Dict
import os

# EXT
import numpy
import torch
from tensorboardX import SummaryWriter

# TYPES
LogDict = Dict[str, List[float]]
AggregatedLogs = Dict[LogDict]


def log_tb_data(writer: Union[SummaryWriter, None], tags: str, data: Any, step: Optional[int] = None) -> None:
    """
    Log some sort of data to Tensorboard if a SummaryWriter has been initialized.

    Parameters
    ----------
    writer: Union[SummaryWriter, None]
        tensorboardX summary writer if given.
    tags: str
        Tag used for writing information.
    data: Any
        Data to be logged.
    step: Optional[int]
        Global step that is used to write the data.
    """
    if writer is not None:
        if type(data) in (int, float):
            writer.add_scalar(tags, data, global_step=step)

        elif type(data) == str:
            writer.add_text(tags, data, global_step=step)

        elif type(data) == dict:
            writer.add_scalars(tags, data, global_step=step)

        elif type(data) in (numpy.ndarray, numpy.array, torch.Tensor):
            writer.add_embedding(tags, data, global_step=step)

        else:
            raise TypeError(f"Can't log result of data type {type(data)}!")


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
    data = defaultdict(list)

    with open(path, "r") as file:
        lines = file.readlines()
        headers, lines = lines[0].strip(), lines[1:]
        header_parts = headers.split()

        for line in lines:
            line_parts = line.strip().split()
            line_parts = map(float, line_parts)  # Cast to float

            for header, part in zip(header_parts, line_parts):
                data[header].append(part)

    return data


def aggregate_logs(paths: List[str], name_func: Optional[Callable] = None) -> AggregatedLogs:
    """
    Aggregate the data from multiple logs into one LogDict. Requires the logs to have the same headers and the same
    number of data points.
    """
    def _default_name_func(path: str):
        model_name = path[:path.rfind("_")]

        return model_name

    name_func = name_func if name_func is not None else _default_name_func
    logs = {name_func(path): read_log(path) for path in paths}

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

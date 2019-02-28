"""
Define helpful logging functions.
"""

# STD
from functools import wraps
from typing import Any, Callable, Optional, Union
import os

# EXT
import numpy
import torch
from tensorboardX import SummaryWriter


def log_result(writer: Union[SummaryWriter, None], tags: str, step: Optional[int] = None):
    """
    Decorator that uses a tensorboardX SummaryWriter to log the result of a function without having to modify the code
    in any way.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            log_tb_data(writer, tags, result, step)

            return result

        return wrapped

    return decorator


def log_tb_data(writer: Union[SummaryWriter, None], tags: str, data: Any, step: Optional[int] = None) -> None:
    """
    Log some sort of data to Tensorboard if a SummaryWriter has been initialized.
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

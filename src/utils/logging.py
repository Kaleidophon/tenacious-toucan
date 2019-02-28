"""
Define helpful logging functions.
"""

# STD
from typing import Any, Optional, Union
import os

# EXT
import numpy
import torch
from tensorboardX import SummaryWriter


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

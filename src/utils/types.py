"""
Define custom types used in this project.
"""

# STD
from typing import Union, Dict, Iterable

# EXT
import numpy as np
import torch
from torch import Tensor

# Torch device - either actual device object or string like "cpu" or "cuda"
Device = Union[str, torch.device]

# Dictionary defining the colors used for plotting loss curves. Should contain "curves" and "intervals" as keys
ColorDict = Dict[str, Dict]

# Dictionary of data columns from log as keys and data points as values
LogDict = Dict[str, np.array]
# Dictionary of logs with their corresponding model names as keys
AggregatedLogs = Dict[str, LogDict]

# Either tuple of two hidden states or single hidden state
AmbiguousHidden = Union[Tensor, Iterable[Tensor]]

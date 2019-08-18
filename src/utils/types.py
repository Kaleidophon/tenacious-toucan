"""
Define custom types used in this project.
"""

# STD
from typing import Union, Dict, Iterable, Tuple

# EXT
import numpy as np
import torch
from torch import Tensor

# --- General ---

# Torch device - either actual device object or string like "cpu" or "cuda"
Device = Union[str, torch.device]

# --- Plotting ---

# Dictionary defining the colors used for plotting loss curves. Should contain "curves" and "intervals" as keys
ColorDict = Dict[str, Dict]

# --- Logging ---

# Dictionary of data columns from log as keys and data points as values
LogDict = Dict[str, np.array]
# Dictionary of logs with their corresponding model names as keys
AggregatedLogs = Dict[str, LogDict]

# -- Model ---

# Either tuple of two hidden states or single hidden state
AmbiguousHidden = Union[Tensor, Iterable[Tensor]]

# Dictionary of layer -> hidden state
HiddenDict = Dict[int, AmbiguousHidden]

# Output of RNN at a time step - output distribution and hidden state
RecurrentOutput = Tuple[Tensor, HiddenDict]

# Output of a recoding RNN at a time step - either RecurrentOutput or additionally the unrecoded output activations
# in the diagnostics case
RecodingOutput = Union[RecurrentOutput, Tuple[Tensor, Tensor, HiddenDict]]

# Step size used for recoding: Either a single float applied to all batch instances or tensor of one individual step
# size per instance
StepSize = Union[float, Tensor]

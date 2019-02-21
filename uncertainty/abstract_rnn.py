"""
Abstract superclass for a RNN that processes sequence step-wise.
"""

# STD
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

# EXT
from torch import nn, Tensor


class AbstractRNN(ABC, nn.Module):
    """
    Abstract RNN class defining some common attributes and functions.
    """
    def __init__(self, rnn_type):
        super().__init__()
        self.rnn_type = rnn_type

    @abstractmethod
    def forward(self, input_var: Tensor, hidden: Optional[Tensor] = None, **additional: Dict) -> Tuple[Tensor, Tensor]:
        """
        Process a sequence of input variables.

        Parameters
        ----------
        input_var: Tensor
            Current input variable.
        hidden: Tensor
            Current hidden state.
        additional: dict
            Dictionary of additional information delivered via keyword arguments.

        Returns
        -------
        out: Tensor
            Decoded output Tensor of current time step.
        hidden: Tensor
            Hidden state of current time step after recoding.
        """
        ...

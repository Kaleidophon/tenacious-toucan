"""
Abstract superclass for a RNN that processes sequence step-wise.
"""

# STD
from abc import ABC, abstractmethod
from typing import Dict, Optional

# EXT
import torch
from torch import nn, Tensor

# PROJECT
from src.utils.types import AmbiguousHidden, RecurrentOutput, Device, HiddenDict


class AbstractRNN(ABC, nn.Module):
    """
    Abstract RNN class defining some common attributes and functions.
    """
    def __init__(self, rnn_type: str, hidden_size: int, embedding_size: int, num_layers: int, device: Device = "cpu"):
        """
        Parameters
        ----------
        rnn_type: str
            RNN type (e.g. "LSTM", "GRU")
        hidden_size: int
            Dimensionality of hidden activations.
        embedding_size: int
            Dimensionality of word embeddings.
        num_layers: int
            Number of RNN layers.
        device: Device
            Torch device the model is being trained on (e.g. "cpu" or "cuda").
        """
        super().__init__()
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.device = device

    @abstractmethod
    def forward(self, input_var: Tensor, hidden: Optional[HiddenDict] = None, **additional: Dict) -> RecurrentOutput:
        """
        Process a sequence of input variables. Has to be overridden in subclass.

        Parameters
        ----------
        input_var: Tensor
            Current input variable.
        hidden: Optional[HiddenDict]
            Current hidden state.
        additional: dict
            Dictionary of additional information delivered via keyword arguments.

        Returns
        -------
        out: Tensor
            Decoded output Tensor of current time step.
        hidden: AmbiguousHidden
            Tuple of hidden state from the current time step.
        """
        ...

    def init_hidden(self, batch_size: int, device: Device) -> AmbiguousHidden:
        """
        Initialize the hidden states for the current network.

        Parameters:
        -----------
        batch_size: int
            Batch size used for training.
        device: Device
            Torch device the model is being trained on (e.g. "cpu" or "cuda").

        Returns
        -------
        hidden: AmbiguousHidden
            Either one hidden state or tuple of hidden and cell state.
        """
        hidden_zero = torch.zeros(batch_size, self.hidden_size).to(device)

        if self.rnn_type == "LSTM":
            return hidden_zero, hidden_zero.clone()
        else:
            return hidden_zero

"""
Abstract superclass for a RNN that processes sequence step-wise.
"""

# STD
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

# EXT
import torch
from torch import nn, Tensor

# PROJECT
from src.utils.types import AmbiguousHidden


class AbstractRNN(ABC, nn.Module):
    """
    Abstract RNN class defining some common attributes and functions.
    """
    def __init__(self, rnn_type, hidden_size, embedding_size, num_layers, device: torch.device = "cpu"):
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
        device: torch.device
            Torch device the model is being trained on (e.g. "cpu" or "cuda").
        """
        super().__init__()
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, rnn_type)(embedding_size, hidden_size, num_layers, batch_first=True)
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.device = device
        self.inferred_device = device

    @abstractmethod
    def forward(self, input_var: Tensor, hidden: Optional[Tensor] = None, **additional: Dict) -> Tuple[Tensor, Tensor]:
        """
        Process a sequence of input variables. Has to be overridden in subclass.

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
        # Do this in subclasses to ensure right device even when running on multiple GPUs
        device = self.current_device(reference=input_var)
        ...

    def current_device(self, reference: Optional[torch.Tensor] = None) -> torch.device:
        """
        Make sure that tensors are moved to right GPU when training with torch.nn.DataParallel. The problem is that
        models are initialized on the default GPU, but during the forward pass the model is copied to all available
        (or specified) GPUs and tensors that are created during the forward pass on-the-fly will still be moved to the
        GPU specified during model initialization.

        The current device cannot be stored in a model attribute because attributes are shared across all model copies
        living on different GPUs, therefore setting the value on one GPU sets it on all others as well. Therefore
        the right device is determined and returned here (optionally using the device of a reference tensor) and then
        handed down to other functions as an argument within the local function scope.

        Parameters
        ----------
        reference: Optional[Tensor]
           Reference tensor which overrides the default device choice if necessary.

        Returns
        -------
        device: torch.device
            Currently relevant device.
        """
        if reference is not None:
            # The GPU of the current forward pass doesn't correspond to the initially specified one
            # -> Return the relevant GPU
            if self.device != reference.device:
                return reference.device

        # Training is done on single GPU or CPU, no problem here.
        return self.device

    def init_hidden(self, batch_size: int, device: torch.device) -> AmbiguousHidden:
        """
        Initialize the hidden states for the current network.

        Parameters:
        -----------
        batch_size: int
            Batch size used for training.
        device: torch.device
            Torch device the model is being trained on (e.g. "cpu" or "cuda").

        Returns
        -------
        hidden: AmbiguousHidden
            Either one hidden state or tuple of hidden and cell state.
        """
        hidden_zero = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        if self.rnn_type == "LSTM":
            return hidden_zero, hidden_zero.clone()
        else:
            return hidden_zero

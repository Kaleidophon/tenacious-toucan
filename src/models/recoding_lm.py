"""
Module that defines an LSTM language model that enables recoding.
"""

# STD
from typing import Optional, Dict, Any, Type

# EXT
from overrides import overrides
from torch import Tensor

# PROJECT
from src.models.language_model import LSTMLanguageModel
from src.utils.types import RecodingOutput, Device, HiddenDict


class RecodingLanguageModel(LSTMLanguageModel):
    """
    A LSTM Language model with a recoding mechanism applied to it. This class is defined explicitly because
    the usual decorator functionality of the uncertainty mechanism prevents pickling of the model.
    """
    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int, num_layers: int, dropout: float,
                 mechanism_class: Type, mechanism_kwargs: dict, device: Device = "cpu", **unused: Any):
        """
        Parameters
        ----------
        vocab_size: int
            Size of input vocabulary.
        embedding_size: int
            Dimensionality of word embeddings.
        hidden_size: int
            Dimensionality of hidden activations.
        num_layers: int
            Number of RNN layers.
        dropout: float
            Probability of dropout layer that is being applied before decoding.
        mechanism_class: Type
            Class of recoding mechanism used.
        mechanism_kwargs: dict
            Init args for recoding mechanism as dictionary.
        device: Device
            Torch device the model is being trained on (e.g. "cpu" or "cuda").
        """
        super().__init__(vocab_size, embedding_size, hidden_size, num_layers, dropout, device)
        self.mechanism = mechanism_class(model=self, **mechanism_kwargs, device=device)
        self.diagnostics = False

    @overrides
    def forward(self, input_var: Tensor, hidden: Optional[HiddenDict] = None, **additional: Dict) -> RecodingOutput:
        """
        Process a sequence of input variables.

        Parameters
        ----------
        input_var: Tensor
            Current input variable.
        hidden: HiddenDict
            Current hidden state.
        additional: dict
            Dictionary of additional information delivered via keyword arguments.

        Returns
        -------
        out: Tensor
            Decoded output Tensor of current time step.
        hidden: HiddenDict
            Hidden state of current time step after recoding.
        """
        out, hidden = super().forward(input_var, hidden, **additional)

        new_out, new_hidden = self.mechanism.recoding_func(input_var, hidden, out, device=self.device, **additional)

        if self.diagnostics:
            return new_out, out, new_hidden

        elif self.mechanism.redecode_output:
            return new_out, new_hidden

        else:
            return out, new_hidden

    def train(self, mode: bool = True) -> None:
        """
        Set the train mode for the model and all the other model parts. Overriding this function is necessary to pass
        any change in training mode to the recoding mechanism as well.

        Parameters
        ----------
        mode: bool
            Training mode to the set the model and its parts to.
        """
        super().train(mode)
        self.mechanism.train(mode)

    def eval(self) -> None:
        """
        Put the model into eval model. Overriding this function is necessary to pass any change in mode to the recoding
        mechanism as well.
        """
        super().eval()
        self.mechanism.eval()

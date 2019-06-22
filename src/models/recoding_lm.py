"""
Module that defines an LSTM language model that enables recoding.
"""

# STD
from typing import Optional, Dict, Tuple

# EXT
import torch
from overrides import overrides
from torch import Tensor

# PROJECT
from src.models.language_model import LSTMLanguageModel


class RecodingLanguageModel(LSTMLanguageModel):
    """
    A LSTM Language model with a recoding mechanism applied to it. This class is defined explicitly because
    the usual decorator functionality of the uncertainty mechanism prevents pickling of the model.
    """
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout, mechanism_class,
                 mechanism_kwargs, device: torch.device = "cpu"):
        super().__init__(vocab_size, embedding_size, hidden_size, num_layers, dropout, device)
        mechanism_kwargs["predictor_kwargs"]["model"] = self
        self.mechanism = mechanism_class(model=self, **mechanism_kwargs, device=device)

    @overrides
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
        out, hidden = super().forward(input_var, hidden, **additional)

        new_out, new_hidden = self.mechanism.recoding_func(input_var, hidden, out, device=self.device, **additional)

        # Only allow recomputing out when gold token is not given and model has to guess, otherwise task is trivialized
        if "target_idx" not in additional:
            return new_out, new_hidden
        else:
            return out, new_hidden

    def train(self, mode=True):
        super().train(mode)
        self.mechanism.train(mode)

    def eval(self):
        super().eval()
        self.mechanism.eval()

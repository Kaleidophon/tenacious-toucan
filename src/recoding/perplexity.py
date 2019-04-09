"""
Perform the recoding step based on a simple signal of the perplexity of the gold token.
"""

# STD
from typing import Dict, Any, Tuple, Optional

# EXT
from overrides import overrides
import torch
from torch import Tensor

# PROJECT
from src.models.abstract_rnn import AbstractRNN
from src.recoding.mechanism import RecodingMechanism
from src.utils.types import HiddenDict


class PerplexityRecoding(RecodingMechanism):
    """
    Define the error signal of the most simple recoding mechanism by calculating the perplexity - or how "surprised"
    the model is by the next token.
    """
    def __init__(self, model: AbstractRNN, predictor_kwargs: Dict, step_type: str, **unused: Any):
        """
        Initialize the mechanism.

        Parameters
        ----------
        model: AbstractRNN
            Model the mechanism is being applied to.
        """
        super().__init__(model, step_type, predictor_kwargs=predictor_kwargs)

    @overrides
    def recoding_func(self, input_var: Tensor, hidden: HiddenDict, out: Tensor, device: torch.device,
                      **additional: Dict) -> Tuple[Tensor, HiddenDict]:
        """
        Recode activations of current step based on some logic defined in a subclass.

        Parameters
        ----------
        input_var: Tensor
            Current input variable.
        hidden: HiddenDict
            Dictionary of all hidden (and cell states) of all network layers.
        out: Tensor
            Output Tensor of current time step.
        device: torch.device
            Torch device the model is being trained on (e.g. "cpu" or "cuda").
        additional: dict
            Dictionary of additional information delivered via keyword arguments.

        Returns
        -------
        out: Tensor
            Re-decoded output Tensor of current time step.
        hidden: Tensor
            Hidden state of current time step after recoding.
        """
        target_idx = additional.get("target_idx", None)

        # Estimate uncertainty of those same predictions
        delta = self.get_perplexity(out, target_idx)

        # Calculate gradient of uncertainty w.r.t. hidden states and make step
        new_out_dist, new_hidden = self.recode_activations(hidden, delta, device)

        return new_out_dist, new_hidden

    @staticmethod
    def get_perplexity(out: Tensor, target_idx: Optional[Tensor] = None) -> Tensor:
        out = out.squeeze(1)

        # If target indices are not given, just use most likely token
        if target_idx is None:
            target_idx = torch.argmax(out, dim=1, keepdim=True)
        else:
            target_idx = target_idx.unsqueeze(1)

        target_probs = torch.gather(out, 1, target_idx)
        target_probs = torch.sigmoid(target_probs)
        target_ppls = 2 ** (-target_probs.log2())

        return target_ppls

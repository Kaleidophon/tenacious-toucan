"""
Do MC Dropout recoding based on the Variational RNNs of [1].

[1] https://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf
"""

# STD
from typing import Optional, Any, Dict, Tuple

# EXT
import torch
from overrides import overrides
from torch import Tensor

# PROJECT
from src.models.abstract_rnn import AbstractRNN
from src.recoding.mc_dropout import MCDropoutMechanism
from src.utils.types import HiddenDict


class VariationalMechanism(MCDropoutMechanism):
    def __init__(self, model: AbstractRNN, hidden_size: int, num_samples: int, weight_decay: float,
                 prior_scale: float, predictor_kwargs: Dict, step_type: str, data_length: Optional[int] = None,
                 **unused: Any):
        """
        Initialize the mechanism.

        Parameters
        ----------
        model: AbstractRNN
            Model the mechanism is being applied to.
        hidden_size: int
            Dimensionality of hidden activations.
        num_samples: int
            Number of samples used to estimate uncertainty.
        weight_decay: float
            L2-regularization parameter.
        prior_scale: float
            Parameter that express belief about frequencies in the input data.
        data_length: Optional[int]
            Number of data points used.
        """
        super().__init__(
            model, hidden_size, num_samples, 0, weight_decay, prior_scale, predictor_kwargs, step_type, data_length,
            **unused
        )

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
        prediction = self.predict_variational(hidden)

        # Estimate uncertainty of those same predictions
        delta = self._calculate_predictive_entropy(prediction)

        # Calculate gradient of uncertainty w.r.t. hidden states and make step
        new_out_dist, new_hidden = self.recode_activations(hidden, out, delta, device, **additional)

        return new_out_dist, new_hidden

    def predict_variational(self, hidden: HiddenDict):
        """
        Generate the output distributions generated using different dropout masks with an affine transformation.

        Parameters
        ----------
        hidden: HiddenDict
            Dictionary of all hidden (and cell states) of all network layers.

        Returns
        -------
        target_predictions: Tensor
            Predicted probabilities for target token.
        """
        # Get topmost hidden activations
        topmost_hidden = self.select(hidden[self.model.num_layers - 1])  # Select topmost hidden activations

        # Collect sample predictions
        output = self.model.decoder(topmost_hidden)
        predictions = output.view(self.model.current_batch_size, self.num_samples, self.model.vocab_size)

        return predictions


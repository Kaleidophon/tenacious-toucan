"""
Do MC Dropout recoding based on the Variational RNNs of [1].

[1] https://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf
"""

# STD
from typing import Optional, Any, Dict, Tuple

# EXT
import torch
from overrides import overrides
from torch import nn, Tensor

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
        target_idx = additional.get("target_idx", None)
        prediction = self.predict_variational(hidden, device, target_idx)

        # Estimate uncertainty of those same predictions
        delta = self._calculate_predictive_uncertainty(prediction)

        # Calculate gradient of uncertainty w.r.t. hidden states and make step
        new_out_dist, new_hidden = self.recode_activations(hidden, out, delta, device, **additional)

        return new_out_dist, new_hidden

    def predict_variational(self, hidden: HiddenDict, device: torch.device, target_idx: Optional[Tensor] = None):
        """
        Generate the output distributions generated using different dropout masks with an affine transformation.

        Parameters
        ----------
        hidden: HiddenDict
            Dictionary of all hidden (and cell states) of all network layers.
        target_idx: Optional[Tensor]
            Indices of actual next tokens (if given). Otherwise the most likely tokens are used.

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

        # If no target is given, compute uncertainty of most likely token
        #target_idx = target_idx if target_idx is not None else torch.argmax(predictions.sum(dim=1), dim=1)
        #target_idx = target_idx.to(device)


        # Select predicted probabilities of target index
        predictions.exp_()  # Exponentiate for later softmax
        norm = predictions.sum(dim=2).unsqueeze(2)
        target_predictions = predictions / norm

        # Select predicted probabilities of target index
        #predictions.exp_()  # Exponentiate for later softmax
        #target_idx = target_idx.view(target_idx.shape[0], 1, 1)
        #target_idx = target_idx.repeat(1, self.num_samples, 1)
        #target_predictions = torch.gather(predictions, 2, target_idx)
        #target_predictions = target_predictions.squeeze(2)

        # Apply softmax (only apply it to actually relevant probabilities, save some computation)
        #norm_factor = predictions.sum(dim=2)  # Gather normalizing constants for softmax
        #target_predictions = target_predictions / norm_factor

        return target_predictions


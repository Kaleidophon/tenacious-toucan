"""
Perform recoding steps based on Monte-Carlo Dropout.
"""

# STD
from math import sqrt
from typing import Optional, Any, Dict, Tuple

# EXT
import torch
from overrides import overrides
from torch import nn, Tensor

# PROJECT
from src.models.abstract_rnn import AbstractRNN
from src.recoding.mechanism import RecodingMechanism
from src.utils.types import HiddenDict


class MCDropoutMechanism(RecodingMechanism):
    """
    Recoding mechanism that bases its recoding on the predictive uncertainty of the decoder, where the uncertainty
    is estimate using MC Dropout [1].

    [1] http://proceedings.mlr.press/v48/gal16.pdf
    """
    def __init__(self, model: AbstractRNN, hidden_size: int, num_samples: int, mc_dropout: float, weight_decay: float,
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
        mc_dropout: float
            Dropout probability used to estimate uncertainty.
        weight_decay: float
            L2-regularization parameter.
        prior_scale: float
            Parameter that express belief about frequencies in the input data.
        data_length: Optional[int]
            Number of data points used.
        """
        super().__init__(model, step_type, predictor_kwargs=predictor_kwargs)

        self.model = model
        self.hidden_size = hidden_size
        self.num_samples = num_samples
        self.mc_dropout = mc_dropout
        self.weight_decay = weight_decay
        self.prior_scale = prior_scale
        self.data_length = data_length

        # Add dropout layer to estimate predictive uncertainty
        self.mc_dropout_layer = nn.Dropout(p=self.mc_dropout)

        # Initialize weights and bias according to prior scale
        self.model.decoder.weight.data.normal_(0, sqrt(self.prior_scale))
        self.model.decoder.bias.data.normal_(0, sqrt(self.prior_scale))

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
        prediction = self._mc_dropout_predict(hidden, device, target_idx)

        # Estimate uncertainty of those same predictions
        delta = self._calculate_predictive_uncertainty(prediction)

        # Calculate gradient of uncertainty w.r.t. hidden states and make step
        new_out_dist, new_hidden = self.recode_activations(hidden, out, delta, device, **additional)

        return new_out_dist, new_hidden

    def _mc_dropout_predict(self, hidden: HiddenDict, device: torch.device, target_idx: Optional[Tensor] = None):
        """
        Make several predictions about the probability of a token using different dropout masks.

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
        num_layers = len(hidden.keys())
        topmost_hidden = self.select(hidden[num_layers - 1])  # Select topmost hidden activations

        # Collect sample predictions
        topmost_hidden = topmost_hidden.unsqueeze(1)
        topmost_hidden = topmost_hidden.repeat(1, self.num_samples, 1)  # Create identical copies for pseudo-batch
        topmost_hidden = self.mc_dropout_layer(topmost_hidden)
        predictions = self.model.predict_distribution(topmost_hidden)

        # If no target is given, compute uncertainty of most likely token
        target_idx = target_idx if target_idx is not None else torch.argmax(predictions.sum(dim=1), dim=1)
        target_idx = target_idx.to(device)

        # Select predicted probabilities of target index
        predictions.exp_()  # Exponentiate for later softmax
        target_idx = target_idx.view(target_idx.shape[0], 1, 1)
        target_idx = target_idx.repeat(1, self.num_samples, 1)
        target_predictions = torch.gather(predictions, 2, target_idx)
        target_predictions = target_predictions.squeeze(2)

        # Apply softmax (only apply it to actually relevant probabilities, save some computation)
        norm_factor = predictions.sum(dim=2)  # Gather normalizing constants for softmax
        target_predictions = target_predictions / norm_factor

        return target_predictions

    def _calculate_predictive_uncertainty(self, predictions: Tensor):
        """
        Calculate the predictive uncertainty based on the predictions made with different dropout masks.
        This corresponds to the equation of the predicted variance given in §4 of [1].

        [1] http://proceedings.mlr.press/v48/gal16.pdf

        Parameters
        ----------
        predictions: Tensor
            Tensor of num_sample predictions per batch instance.

        Returns
        -------
        uncertainty: Tensor
            Estimated predictive uncertainty per batch instance.
        """
        prior_info = 2 * self.mc_dropout * self.prior_scale ** 2 / (2 * self.data_length * self.weight_decay)
        return predictions.var(dim=1).unsqueeze(1) + prior_info

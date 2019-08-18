"""
Module concerning the recoding based on an anchored ensemble, following the paper
"Uncertainty in Neural Networks: Bayesian Ensembling" (https://arxiv.org/pdf/1810.05546.pdf).
"""

# STD
from math import sqrt
from typing import Optional, Any, Dict, Tuple

# EXT
import torch
from torch import Tensor
import torch.nn as nn
from overrides import overrides
import torch.nn.functional as F

# PROJECT
from src.models.abstract_rnn import AbstractRNN
from src.recoding.mechanism import RecodingMechanism
from src.utils.types import HiddenDict, Device


def has_anchored_ensemble(model: AbstractRNN) -> bool:
    """
    Check whether a model works with an anchored ensemble (and therefore requires an extra loss).

    Parameters
    ----------
    model: AbstractRNN
        Model to be checked.
    """
    if hasattr(model, "mechanism"):
        if isinstance(model.mechanism, AnchoredEnsembleMechanism):
            return True

    return False


class AnchoredEnsembleMechanism(RecodingMechanism):
    """
    Recoding mechanism that bases its recoding on the predictive uncertainty of the decoder, where the uncertainty
    is estimated using Bayesian Anchored Ensembles [1].

    [1] https://arxiv.org/pdf/1810.05546.pdf
    """

    def __init__(self, model: AbstractRNN, hidden_size: int, num_samples: int, data_noise: float, prior_scale: float,
                 predictor_kwargs: dict, step_type: str, device: Device, data_length: Optional[int] = None,
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
            Number of members in the ensemble used to estimate uncertainty.
        data_noise: float
            Estimate of inherent noise in data.
        prior_scale: float
            Parameter that determines the standard deviation initial weights and anchors are sampled from.
        predictor_kwargs: dict
            Dictionary of arguments to initialize the step size predictors.
        step_type: str
            Type of recoding step that is being used.
        device: Device
            Torch device the model is being trained on (e.g. "cpu" or "cuda").
        data_length: Optional[int]
            Number of data points used.
        """
        super().__init__(model, step_type, predictor_kwargs=predictor_kwargs)

        self.model = model
        self.hidden_size = hidden_size
        self.num_samples = num_samples
        self.prior_scale = prior_scale
        self.data_length = data_length
        self.device = device

        # Prepare ensemble
        self._sample_anchors(device)
        self._init_ensemble()
        self.lambda_ = data_noise / prior_scale

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
        # Estimate uncertainty of those same predictions
        delta = self._calculate_predictive_entropy(hidden)

        # Calculate gradient of uncertainty w.r.t. hidden states and make step
        new_out_dist, new_hidden = self.recode_activations(hidden, out, delta, device, **additional)

        return new_out_dist, new_hidden

    def _calculate_predictive_entropy(self, hidden: HiddenDict) -> Tensor:
        """
        Calculate the predictive uncertainty of the decoder ensemble by measuring the variance of the predictions
        w.r.t. to the target token.

        Parameters
        ----------
        hidden: HiddenDict
            Dictionary of all current hidden and cell states.

        Returns
        -------
        delta: Tensor
            Estimated predictive uncertainty per batch instance.
        """
        # Get topmost hidden activations
        num_layers = len(hidden.keys())
        topmost_hidden = self.select(hidden[num_layers - 1])  # Select topmost hidden activations

        predictions = [decoder(topmost_hidden) for decoder in self.model.decoder_ensemble]
        predictions = torch.stack(predictions)

        predictions = F.softmax(predictions, dim=2)  # Log-softmax is already contained in cross_entropy loss above
        mean_predictions = predictions.mean(dim=1)
        pred_entropy = -(mean_predictions * mean_predictions.log()).sum()

        return pred_entropy

    def _init_ensemble(self) -> None:
        """
        Initialize a decoder ensemble.
        """
        # Delete decoder and replace with ensemble
        del self.model.decoder
        self.model.decoder_ensemble = []

        for k in range(self.num_samples):
            decoder = nn.Linear(self.hidden_size, self.model.vocab_size)

            # Initialize weights and bias according to prior scale
            decoder.weight.data.normal_(0, sqrt(self.prior_scale))
            decoder.bias.data.normal_(0, sqrt(self.prior_scale))

            # Integrate into model
            self.model.add_module(f"decoder{k}", decoder)  # Register as model params so all decoders get learned
            self.model.decoder_ensemble.append(decoder)

        # Inject new decoding function into the model
        self.model.decoder = self._decode_ensemble

    def _sample_anchors(self, device: Device) -> None:
        """
        Sample the anchor points for the Bayesian Anchored Ensemble.

        Parameters
        ----------
        device: torch.device
                Torch device the model is being trained on (e.g. "cpu" or "cuda").
        """
        self.weight_anchors, self.bias_anchors = [], []

        # Sample one anchor point per ensemble member
        for _ in range(self.num_samples):
            weight_anchor = torch.zeros(self.model.vocab_size, self.hidden_size, device=device)
            weight_anchor.normal_(0, sqrt(self.prior_scale))
            bias_anchor = torch.zeros(self.model.vocab_size, device=device)
            bias_anchor.normal_(0, sqrt(self.prior_scale))

    @property
    def ensemble_losses(self) -> Tensor:
        """
        Return the current loss of the Bayesian anchored ensemble based on the current parameters of the ensemble's
        members. This basically corresponds to the second term of eq. 9.
        """
        ensemble_losses = torch.ones(self.num_samples).to(self.device)
        members_and_anchors = zip(self.model.decoder_ensemble, self.weight_anchors, self.bias_anchors)

        for k, (decoder, weight_anchor, bias_anchor) in enumerate(members_and_anchors):
            ensemble_losses[k] += torch.norm(decoder.weight - weight_anchor)
            ensemble_losses[k] += torch.norm(decoder.bias - bias_anchor)

        # Just sum everything and do a single backprop for all ensemble members (is faster)
        ensemble_losses = ensemble_losses.mean()

        return self.lambda_ / self.data_length * ensemble_losses

    def _decode_ensemble(self, hidden: Tensor) -> Tensor:
        """
        Decode hidden activations using an ensemble of decoders.

        Parameters
        ----------
        hidden: Tensor
            Hidden activations to be decoded.

        Returns
        -------
        out: Tensor
            Decoded hidden activations batch_size * vocab_size
        """
        decoded_activations = [decoder(hidden) for decoder in self.model.decoder_ensemble]
        out = torch.stack(decoded_activations)
        out = out.mean(dim=0)

        return out

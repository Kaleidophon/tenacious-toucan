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

# PROJECT
from src.models.abstract_rnn import AbstractRNN
from src.recoding.mechanism import RecodingMechanism
from src.utils.types import HiddenDict


class AnchoredEnsembleMechanism(RecodingMechanism):
    """
    Recoding mechanism that bases its recoding on the predictive uncertainty of the decoder, where the uncertainty
    is estimated using Bayesian Anchored Ensembles [1].

    [1] https://arxiv.org/pdf/1810.05546.pdf
    """
    def __init__(self, model: AbstractRNN, hidden_size: int, num_samples: int, data_noise: float, prior_scale: float,
                 predictor_kwargs: Dict, step_type: str, data_length: Optional[int] = None, **unused: Any):
        super().__init__(model, step_type, predictor_kwargs=predictor_kwargs)

        self.model = model
        self.hidden_size = hidden_size
        self.num_samples = num_samples
        self.prior_scale = prior_scale
        self.data_length = data_length

        # Prepare ensemble
        self._sample_anchors()
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
        target_idx = additional.get("target_idx", None)

        # Estimate uncertainty of those same predictions
        delta = self._calculate_predictive_uncertainty(hidden, target_idx)

        # Calculate gradient of uncertainty w.r.t. hidden states and make step
        new_out_dist, new_hidden = self.recode_activations(hidden, delta, device)

        return new_out_dist, new_hidden

    def _calculate_predictive_uncertainty(self, hidden: HiddenDict, target_idx: int) -> Tensor:
        """
        Calculate the predictive uncertainty of the decoder ensemble by measuring the variance of the predictions
        w.r.t. to the target token.
        """
        # Get topmost hidden activations
        num_layers = len(hidden.keys())
        topmost_hidden = self.select(hidden[num_layers - 1])  # Select topmost hidden activations

        decoded_activations = [decoder(topmost_hidden) for decoder in self.model.decoder_ensemble]
        decoded_activations = torch.stack(decoded_activations)
        pred_var = decoded_activations.var(dim=0)

        delta = pred_var[:, target_idx]

        return delta

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

        self.model.decoder = self._decode_ensemble

    def _sample_anchors(self) -> None:
        """
        Sample the anchor points for the Bayesian Anchored Ensemble.
        """
        self.weight_anchor = torch.random.normal(0, sqrt(self.prior_scale), size=[self.hidden_size, 1])
        self.bias_anchor = torch.random.normal(0, sqrt(self.prior_scale), size=[self.hidden_size])

    @property
    def ensemble_loss(self) -> Tensor:
        """
        Return the current loss of the Bayesian anchored ensemble based on the current parameters of the ensemble's
        members. This basically corresponds to the second term of eq. 9.
        """
        loss = 0

        for decoder in self.model.decoder_ensemble:
            loss += torch.flatten(torch.sqrt((decoder.weight - self.weight_anchor).pow(2))).sum()
            loss += torch.flatten(torch.sqrt((decoder.bias - self.bias_anchor).pow(2))).sum()

        return self.lambda_ / self.data_length * loss

    def _decode_ensemble(self, input: Tensor) -> Tensor:
        """
        Decode hidden activations using an ensemble of decoders.

        Parameters
        ----------
        input: Tensor
            Hidden activations to be decoded.

        Returns
        -------
        out: Tensor
            Decoded hidden activations batch_size * vocab_size
        """
        decoded_activations = [decoder(input) for decoder in self.model.decoder_ensemble]
        decoded_activations = torch.stack(decoded_activations)
        out = decoded_activations.mean(dim=0)

        return out

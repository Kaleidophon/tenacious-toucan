"""
Define a model with an intervention mechanism that bases its interventions on the uncertainty of a prediction.
"""

# STD
from typing import Dict, Tuple, List, Optional

# EXT
from overrides import overrides
from rnnalyse.interventions.mechanism import InterventionMechanism, InterventionLSTM
from rnnalyse.typedefs.models import FullActivationDict
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

# PROJECT
from language_model import LanguageModel


class Predictor(nn.Module):

    def __init__(self, predictor_layers, hidden_size):
        super().__init__()
        self.predictor_layers = predictor_layers
        self.hidden_size = hidden_size

        # Init layers
        last_layer_size = predictor_layers[0]
        self.input_layer = nn.Linear(hidden_size, last_layer_size)
        self.hidden_layers = []

        for current_layer_size in predictor_layers[1:]:
            self.hidden_layers.append(nn.Linear(last_layer_size, current_layer_size))
            self.hidden_layers.append(F.relu)
            last_layer_size = current_layer_size

        self.output_layer = nn.Linear(last_layer_size, 1)  # Output scalar alpha_t

        self.model = nn.Sequential(
            self.input_layer,
            F.relu,
            *self.hidden_layers,
            self.output_layer,
            F.sigmoid
        )

    def forward(self, hidden_states: List[Tensor]):
        hidden_window = torch.cat(hidden_states, dim=0)

        return self.model(hidden_window)


class UncertaintyMechanism(InterventionMechanism):
    """
    Intervention mechanism that bases its intervention on the predictive uncertainty of a model.
    """
    # TODO: Set sensible default params here
    def __init__(self,
                 forward_lstm: InterventionLSTM,
                 predictor_layers: Tuple[int],
                 hidden_size: int,
                 window_size: int,
                 num_samples: int,
                 dropout_prob: float,
                 weight_decay: float,
                 prior_scale: float,
                 data_length: Optional[int] = None):

        super().__init__(forward_lstm, trigger_func=self._determine_step_size)

        self.model = forward_lstm
        self.predictor_layers = predictor_layers
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.num_samples = num_samples
        self.dropout_prob = dropout_prob
        self.weight_decay = weight_decay
        self.prior_scale = prior_scale
        self.data_length = data_length

        # Initialize predictor
        self.predictor = Predictor(predictor_layers, hidden_size)

    def _determine_step_size(self, hidden_states: List[Tensor]):
        alpha = self.predictor(hidden_states)

        return alpha

    @overrides
    def intervention_func(self,
                          inp: str,
                          prev_activations: FullActivationDict,
                          out: Tensor,
                          activations: FullActivationDict,
                          **additional: Dict) -> Tuple[Tensor, FullActivationDict]:
        """
        Conduct an intervention based on uncertainty.

        Parameters
        ----------
        inp: str
            Current input token.
        prev_activations: FullActivationDict
            Activations of the previous time step.
        out: Tensor
            Output Tensor of current time step.
        activations: FullActivationDict
            Activations of current time step.
        additional: dict
            Dictionary of additional information delivered via keyword arguments.

        Returns
        -------
        out: Tensor
            Re-decoded output Tensor of current time step.
        activations: FullActivationDict
            Activations of current time step after interventions.
        """
        # Calculate predictive uncertainty using different dropout masks
        # TODO

        # Calculate gradient of uncertainty w.r.t. hidden states
        # TODO

        # Adjust hidden states
        # TODO

        # Redecode
        # TODO

    def _predict_with_dropout(self, output: Tensor, model: LanguageModel, target_idx: int = None):
        # Temporarily add dropout layer
        output_layer = model.out.clone_()
        output_layer = nn.Sequential(
            output_layer,
            nn.Dropout(p=self.dropout_prob)
        )

        # Collect sample predictions
        predictions = torch.zeros(self.num_samples)

        for k in range(self.num_samples):
            out_dist = self.model.predict_distribution(output, output_layer)
            out_dist = F.softmax(out_dist)
            # TODO: Does this make sense?
            prediction = out_dist[target_idx] if target_idx is not None else torch.argmax(out_dist)
            predictions[k] = prediction

        return predictions

    def _calculate_predictive_uncertainty(self, predictions: Tensor):
        # TODO: What to do when data length is not given?
        prior_info = 2 * self.dropout_prob * self.prior_scale ** 2 / (2 * self.data_length * self.weight_decay)
        return predictions.var() * prior_info


"""
Define a model with an intervention mechanism that bases its interventions on the uncertainty of a prediction.
"""

# STD
from typing import Dict, Tuple, List, Optional

# EXT
from overrides import overrides
import torch
from torch import Tensor
from torch import nn
from torch.nn import ReLU, Sigmoid
import torch.nn.functional as F

# PROJECT
from .abstract_rnn import AbstractRNN
from .recoding_mechanism import RecodingMechanism
from .compatability import RNNCompatabilityMixin, AmbiguousHidden

from .language_model import LSTMLanguageModel


class UncertaintyLSTMLanguageModel(LSTMLanguageModel):
    """
    A LSTM Language model with an uncertainty recoding mechanism applied to it. This class is defined explicitly because
    the usual decorator functionality of the uncertainty mechanism prevents pickling of the model.
    """
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, mechanism_kwargs):
        super().__init__(vocab_size, embedding_size, hidden_size, num_layers)
        self.mechanism = UncertaintyMechanism(model=self, **mechanism_kwargs)

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

        return self.mechanism.recoding_func(input_var, hidden, out, **additional)


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
            self.hidden_layers.append(ReLU())
            last_layer_size = current_layer_size

        self.output_layer = nn.Linear(last_layer_size, 1)  # Output scalar alpha_t

        self.model = nn.Sequential(
            self.input_layer,
            ReLU(),
            *self.hidden_layers,
            self.output_layer,
            Sigmoid()
        )

    def forward(self, hidden_states: List[Tensor]):
        hidden_window = torch.cat(hidden_states, dim=0)

        return self.model(hidden_window)


class UncertaintyMechanism(RecodingMechanism, RNNCompatabilityMixin):
    """
    Intervention mechanism that bases its intervention on the predictive uncertainty of a model.
    """
    def __init__(self,
                 model: AbstractRNN,
                 predictor_layers: List[int],
                 hidden_size: int,
                 window_size: int,
                 num_samples: int,
                 dropout_prob: float,
                 weight_decay: float,
                 prior_scale: float,
                 data_length: Optional[int] = None):

        super().__init__(model)

        self.model = model
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
    def recoding_func(self, input_var: Tensor, hidden: Tensor, out: Tensor,
                      **additional: Dict) -> Tuple[Tensor, AmbiguousHidden]:
        """
        Recode activations of current step based on some logic defined in a subclass.

        Parameters
        ----------
        input_var: Tensor
            Current input variable.
        hidden: Tensor
            Current hidden state.
        out: Tensor
            Output Tensor of current time step.
        additional: dict
            Dictionary of additional information delivered via keyword arguments.

        Returns
        -------
        out: Tensor
            Re-decoded output Tensor of current time step.
        hidden: Tensor
            Hidden state of current time step after recoding.
        """
        # Predict step size
        # TODO
        step_size = 0.5

        # Make predictions using different dropout mask
        hidden = self.hidden_compatible(hidden, self._wrap_in_var, requires_grad=True)
        optimizers = [self.optimizer_class(hidden, lr=step_size) for hidden in self.hidden_scatter(hidden)]
        [optimizer.zero_grad() for optimizer in optimizers]
        target_idx = additional.get("target_idx", None)
        predictions = self.hidden_compatible(hidden, self._predict_with_dropout, self.model, target_idx)

        # Estimate uncertainty of those same predictions
        uncertainties = [self._calculate_predictive_uncertainty(prediction) for prediction in predictions]

        # Calculate gradient of uncertainty w.r.t. hidden states and make step
        new_hidden = [self.recode(h, delta, optimizer) for h, delta, optimizer in zip(hidden, uncertainties, optimizers)]

        # Re-decode
        W_ho, b_ho = self._get_output_weights(self.model)
        new_out = torch.tanh(self.hidden_select(hidden) @ W_ho + b_ho)
        num_layers, batch_size, out_dim = new_out.shape
        new_out = new_out.view(batch_size, num_layers, out_dim)

        return new_out, new_hidden

    def _predict_with_dropout(self, hidden: Tensor, model: AbstractRNN, target_idx: Tensor = None):
        # Recompute out, otherwise gradients get lost ;-(
        W_ho, b_ho = self._get_output_weights(model)
        output = torch.tanh(hidden @ W_ho + b_ho)

        # Temporarily add dropout layer
        dropout_output_layer = nn.Sequential(
            model.out_layer,
            nn.Dropout(p=self.dropout_prob)
        )

        # Collect sample predictions
        batch_size = output.shape[1]
        predictions = torch.empty(0, batch_size, self.model.vocab_size)

        for k in range(self.num_samples):
            out_dist = self.model.predict_distribution(output, dropout_output_layer)
            predictions = torch.cat((predictions, out_dist), dim=0)

        # Normalize "in batch"
        # TODO: Does this make sense?
        target_idx = target_idx if target_idx is not None else torch.argmax(predictions.sum(dim=0), dim=1)
        predictions = F.softmax(predictions, dim=2)

        # Select predicted probabilities of target index
        target_idx = target_idx.view(1, target_idx.shape[0], 1)
        target_idx = target_idx.repeat(self.num_samples, 1, 1)
        target_predictions = torch.gather(predictions, 2, target_idx)
        target_predictions = target_predictions.squeeze(2)

        return target_predictions

    def _calculate_predictive_uncertainty(self, predictions: Tensor):
        # TODO: What to do when data length is not given?
        prior_info = 2 * self.dropout_prob * self.prior_scale ** 2 / (2 * self.data_length * self.weight_decay)
        return predictions.var(dim=0).unsqueeze(1) * prior_info

    @staticmethod
    def _get_output_weights(model: AbstractRNN):
        NHID = model.hidden_size

        # TODO: Support multiple layers
        if isinstance(model, LSTMLanguageModel):
            W_ho = model.rnn.weight_hh_l0[3*NHID:4*NHID]
            b_ho = model.rnn.bias_hh_l0[3*NHID:4*NHID]
        else:
            ...
            # TODO

        return W_ho, b_ho


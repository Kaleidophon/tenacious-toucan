"""
Define a model with an intervention mechanism that bases its interventions on the uncertainty of a prediction.
"""

# STD
from typing import Dict, Tuple, Optional, Iterable

# EXT
from overrides import overrides
import torch
from torch import Tensor
from torch import nn
from torch.nn import ReLU, Sigmoid

# PROJECT
from models.abstract_rnn import AbstractRNN
from .recoding_mechanism import RecodingMechanism
from utils.compatability import RNNCompatabilityMixin, AmbiguousHidden

from models.language_model import LSTMLanguageModel


class StepPredictor(nn.Module):

    def __init__(self, predictor_layers, hidden_size, window_size):
        super().__init__()
        self.predictor_layers = predictor_layers
        self.hidden_size = hidden_size
        self.window_size = window_size

        # Init layers
        last_layer_size = predictor_layers[0]
        self.input_layer = nn.Linear(hidden_size * window_size, last_layer_size)
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

    def forward(self, hidden_window: Tensor):
        return self.model(hidden_window)


class UncertaintyMechanism(RecodingMechanism, RNNCompatabilityMixin):
    """
    Intervention mechanism that bases its intervention on the predictive uncertainty of a model.
    In this case the step size is constant during the recoding step.
    """
    def __init__(self,
                 model: AbstractRNN,
                 hidden_size: int,
                 num_samples: int,
                 dropout_prob: float,
                 weight_decay: float,
                 prior_scale: float,
                 average_recoding: bool,
                 step_size: float,
                 data_length: Optional[int] = None,
                 **unused: Dict):

        super().__init__(model, average_recoding=average_recoding)

        self.model = model
        self.hidden_size = hidden_size
        self.num_samples = num_samples
        self.dropout_prob = dropout_prob
        self.weight_decay = weight_decay
        self.prior_scale = prior_scale
        self.data_length = data_length
        self.step_size = step_size

    def _determine_step_size(self, hidden: Tensor) -> float:
        return self.step_size

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
        step_size = self._determine_step_size(hidden)

        # Make predictions using different dropout mask
        hidden = self.hidden_compatible(hidden, self._wrap_in_var, requires_grad=True)
        # Use a step-size (or "learning-rate") of 1 here because optimizers don't support a different step size for
        # for every batch instance, actual step size is applied in recode()
        optimizers = [self.optimizer_class(hidden, lr=1) for hidden in self.hidden_scatter(hidden)]
        [optimizer.zero_grad() for optimizer in optimizers]
        target_idx = additional.get("target_idx", None)
        predictions = self.hidden_compatible(hidden, self._predict_with_dropout, self.model, target_idx)

        # Estimate uncertainty of those same predictions
        uncertainties = [self._calculate_predictive_uncertainty(prediction) for prediction in predictions]

        # Calculate gradient of uncertainty w.r.t. hidden states and make step
        new_hidden = [
            self.recode(h, delta, optimizer, step_size)
            for h, delta, optimizer in zip(hidden, uncertainties, optimizers)
        ]

        # Re-decode
        W_ho, b_ho = self._get_output_weights(self.model)
        new_out = torch.tanh(self.hidden_select(hidden) @ W_ho + b_ho)
        num_layers, batch_size, out_dim = new_out.shape
        new_out = new_out.view(batch_size, num_layers, out_dim)

        return new_out, new_hidden

    def _predict_with_dropout(self, output: Tensor, model: AbstractRNN, target_idx: Tensor = None):
        """
        Make several predictions about the probability of a token using different dropout masks.
        """
        batch_size, seq_len, output_dim = output.size()
        output.view(seq_len, batch_size, output_dim)

        # Temporarily add dropout layer
        # Use DataParallel in order to perform k passes in parallel (with different masks!
        dropout_layer = nn.DataParallel(nn.Dropout(p=self.dropout_prob))

        # Collect sample predictions
        output = model.predict_distribution(output)
        output = output.repeat(self.num_samples, 1, 1)  # Create identical copies for pseudo-batch
        # Because different dropout masks are used in DataParallel, this will yield different results per batch instance
        predictions = dropout_layer(output)

        # Normalize "in batch"
        # TODO: Does this make sense?
        target_idx = target_idx if target_idx is not None else torch.argmax(predictions.sum(dim=0), dim=1)

        # Select predicted probabilities of target index
        predictions = predictions.exp()  # Exponentiate for later softmax
        target_idx = target_idx.view(1, target_idx.shape[0], 1)
        target_idx = target_idx.repeat(self.num_samples, 1, 1)
        target_predictions = torch.gather(predictions, 2, target_idx)
        target_predictions = target_predictions.squeeze(2)

        # Apply softmax (only apply it to actually relevant probabilities, save some computation)
        norm_factor = predictions.sum(dim=2)  # Gather normalizing constants for softmax
        target_predictions = target_predictions / norm_factor

        return target_predictions

    def _calculate_predictive_uncertainty(self, predictions: Tensor):
        prior_info = 2 * self.dropout_prob * self.prior_scale ** 2 / (2 * self.data_length * self.weight_decay)
        return predictions.var(dim=0).unsqueeze(1) * prior_info

    @staticmethod
    def _get_output_weights(model: AbstractRNN) -> Tuple[Tensor, Tensor]:
        NHID = model.hidden_size

        # TODO: Support multiple layers
        if isinstance(model, LSTMLanguageModel):
            W_ho = model.rnn.weight_hh_l0[3*NHID:4*NHID]
            b_ho = model.rnn.bias_hh_l0[3*NHID:4*NHID]
        else:
            W_ho, b_ho = None, None
            # TODO: Support models other than LSTM

        return W_ho, b_ho


class AdaptingUncertaintyMechanism(UncertaintyMechanism):
    """
    Same as UncertaintyMechanism, except that the step size is parameterized by a MLP, which predicts it based
    on a window of previous hidden states.
    """
    def __init__(self,
                 model: AbstractRNN,
                 hidden_size: int,
                 num_samples: int,
                 dropout_prob: float,
                 weight_decay: float,
                 prior_scale: float,
                 average_recoding: bool,
                 window_size: int,
                 predictor_layers: Iterable[int],
                 data_length: Optional[int] = None,
                 **unused: Dict):

        super().__init__(
            model=model, hidden_size=hidden_size, num_samples=num_samples, dropout_prob=dropout_prob,
            weight_decay=weight_decay, prior_scale=prior_scale, average_recoding=average_recoding,
            data_length=data_length, **unused
        )

        # Initialize additional parts of model to make it more adaptive
        self.window_size = window_size
        self.predictor = StepPredictor(predictor_layers, hidden_size, window_size)
        self.hidden_buffer = []  # Save hidden states

    def train(self):
        """ When model mode changes, erase buffer. """
        super().train()
        self.hidden_buffer = []

    def test(self):
        """ When model mode changes, erase buffer. """
        super().test()
        self.hidden_buffer = []

    def _determine_step_size(self, hidden: Tensor) -> float:

        hidden = self.hidden_select(hidden)
        num_layers, batch_size, _ = hidden.size()
        hidden = hidden.view(batch_size, num_layers, self.hidden_size)

        # If buffer is empty, initialize it with zero hidden states
        if len(self.hidden_buffer) == 0:
            self.hidden_buffer = [torch.zeros(hidden.shape)] * self.window_size

        # Add hidden state to buffer
        self.hidden_buffer.append(hidden)

        if len(self.hidden_buffer) > self.window_size:
            self.hidden_buffer.pop()  # If buffer is full, remove oldest element

        # Predict step size
        hidden_window = torch.cat(self.hidden_buffer, dim=2)
        step_size = self.predictor(hidden_window)
        step_size = step_size.view(1, batch_size, 1)

        return step_size

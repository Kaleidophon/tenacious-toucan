"""
Define a model with an intervention mechanism that bases its interventions on the uncertainty of a prediction.
"""

# STD
from typing import Dict, Tuple, Optional, Iterable, Any

# EXT
from overrides import overrides
import torch
from torch import Tensor
from torch import nn
from torch.nn import ReLU, Sigmoid

# PROJECT
from src.models.abstract_rnn import AbstractRNN
from src.recoding.mechanism import RecodingMechanism
from src.utils.compatability import RNNCompatabilityMixin
from src.utils.types import AmbiguousHidden
from src.models.language_model import LSTMLanguageModel


class StepPredictor(nn.Module):
    """
    Function that determines the recoding step size based on a window of previous hidden states.
    """
    def __init__(self, predictor_layers: Iterable[int], hidden_size: int, window_size: int):
        """
        Initialize model.

        Parameters
        ----------
        predictor_layers: Iterable[int]
            Layer sizes for MLP as some sort of iterable.
        hidden_size: int
            Dimensionality of hidden activations.
        window_size: int
            Number of previous hidden states to be considered for prediction.
        """
        super().__init__()
        self.predictor_layers = predictor_layers
        self.hidden_size = hidden_size
        self.window_size = window_size

        # Init layers
        self.model = nn.Sequential()
        last_layer_size = predictor_layers[0]
        self.model.add_module("input", nn.Linear(hidden_size * window_size, last_layer_size, bias=False))
        self.model.add_module("relu0", ReLU())

        for layer_n, current_layer_size in enumerate(predictor_layers[1:]):
            self.model.add_module(f"hidden{layer_n+1}", nn.Linear(last_layer_size, current_layer_size, bias=False))
            self.model.add_module(f"relu{layer_n+1}", ReLU())
            last_layer_size = current_layer_size

        self.model.add_module("out", nn.Linear(last_layer_size, 1, bias=False))  # Output scalar alpha_t
        self.model.add_module("sigmoid", Sigmoid())

    def forward(self, hidden_window: Tensor) -> Tensor:
        """
        Prediction step.

        Parameters
        ----------
        hidden_window: Tensor
            Window of previous hidden states of shape Batch size x Window size x Hidden dim

        Returns
        -------
        step_size: Tensor
            Batch size x 1 tensor of predicted step sizes per batch instance.
        """
        return self.model(hidden_window)


class UncertaintyMechanism(RecodingMechanism, RNNCompatabilityMixin):
    """
    Intervention mechanism that bases its intervention on the predictive uncertainty of a model.
    In this case the step size is constant during the recoding step.
    """
    def __init__(self, model: AbstractRNN, hidden_size: int, num_samples: int, mc_dropout: float, weight_decay: float,
                 prior_scale: float, step_size: float, data_length: Optional[int] = None, **unused: Any):
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
        step_size: float
            Fixed step size for decoding.
        data_length: Optional[int]
            Number of data points used.
        """
        super().__init__(model)

        self.model = model
        self.hidden_size = hidden_size
        self.num_samples = num_samples
        self.mc_dropout = mc_dropout
        self.weight_decay = weight_decay
        self.prior_scale = prior_scale
        self.data_length = data_length
        self.step_size = step_size

        # Add dropout layer to estimate predictive uncertainty
        self.mc_dropout_layer = nn.Dropout(p=self.mc_dropout)

    def _determine_step_size(self, hidden: Tensor, device: torch.device) -> float:
        """
        Determine recoding step size. In this case, only a fixed step size is returned.

        Parameters
        ----------
        hidden: Tensor
            Current hidden state used to determine step size (in this case unused).
        device: torch.device
            Torch device the model is being trained on (e.g. "cpu" or "cuda").

        Returns
        -------
        step_size: float
            Fixed step size.
        """
        return self.step_size

    @overrides
    def recoding_func(self, input_var: Tensor, hidden: Tensor, out: Tensor, device: torch.device,
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
        # Predict step size
        step_size = self._determine_step_size(hidden, device)

        # Make predictions using different dropout mask
        hidden = self.map(hidden, self._wrap_in_var, requires_grad=True)

        # Use a step-size (or "learning-rate") of 1 here because optimizers don't support a different step size for
        # for every batch instance, actual step size is applied in recode()
        optimizers = [self.optimizer_class(hidden, lr=1) for hidden in self.scatter(hidden)]
        [optimizer.zero_grad() for optimizer in optimizers]
        target_idx = additional.get("target_idx", None)
        predictions = self.map(hidden, self._mc_dropout_predict, device, target_idx)

        # Estimate uncertainty of those same predictions
        uncertainties = [self._calculate_predictive_uncertainty(prediction) for prediction in predictions]

        # Calculate gradient of uncertainty w.r.t. hidden states and make step
        new_hidden = [
            self.recode(h, delta, optimizer, step_size, device)
            for h, delta, optimizer in zip(hidden, uncertainties, optimizers)
        ]

        # Re-decode
        W_ho, b_ho = self._get_output_weights(device)
        new_out = torch.tanh(self.select(hidden) @ W_ho + b_ho)
        num_layers, batch_size, out_dim = new_out.shape
        new_out = new_out.view(batch_size, num_layers, out_dim)
        new_out = self.model.dropout_layer(new_out)
        new_out_dist = self.model.predict_distribution(new_out, self.model.out_layer).to(device)

        return new_out_dist, new_hidden

    def _mc_dropout_predict(self, hidden: Tensor, device: torch.device, target_idx: Optional[Tensor] = None):
        """
        Make several predictions about the probability of a token using different dropout masks.

        Parameters
        ----------
        hidden: Tensor
            Current hidden activations.
        model: AbstractRNN
            Model which predictive uncertainty is being estimated.
        target_idx: Optional[Tensor]
            Indices of actual next tokens (if given). Otherwise the most likely tokens are used.

        Returns
        -------
        target_predictions: Tensor
            Predicted probabilities for target token.
        """
        # Re-compute output distribution, otherwise required gradient for hidden gets lost
        W_ho, b_ho = self._get_output_weights(device)
        out = torch.tanh(hidden @ W_ho.detach() + b_ho.detach())
        seq_len, batch_size, out_dim = out.size()
        out = out.view(batch_size, seq_len, out_dim)

        # Collect sample predictions
        output = self.model.predict_distribution(out, self.model.out_layer).to(device)
        output = output.repeat(1, self.num_samples, 1)  # Create identical copies for pseudo-batch
        # Because different dropout masks are used in DataParallel, this will yield different results per batch instance
        predictions = self.mc_dropout_layer(output)

        # TODO: Does this make sense?
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
        return predictions.var(dim=0).unsqueeze(1) * prior_info

    def _get_output_weights(self, device: Optional[torch.device] = None) -> Tuple[Tensor, Tensor]:
        """
        Retrieve output weights of model for later re-decoding.

        Parameters
        ----------
        device: Optional[torch.device]
            Torch device the model is being trained on (e.g. "cpu" or "cuda").

        Returns
        -------
        weights: Tuple[Tensor, Tensor]
            Tuple of weights W_ho and bias b_ho.
        """
        NHID = self.model.hidden_size

        # TODO: Support multiple layers
        if isinstance(self.model, LSTMLanguageModel):
            W_ho = self.model.rnn.weight_hh_l0[3*NHID:4*NHID]
            b_ho = self.model.rnn.bias_hh_l0[3*NHID:4*NHID]
        else:
            # TODO: Support models other than LSTM
            raise Exception("Other models than LSTM are currently not supported!")

        if device is not None:
            W_ho = W_ho.to(device)
            b_ho = b_ho.to(device)

        return W_ho, b_ho


class AdaptingUncertaintyMechanism(UncertaintyMechanism):
    """
    Same as UncertaintyMechanism, except that the step size is parameterized by a MLP, which predicts it based
    on a window of previous hidden states.
    """
    def __init__(self, model: AbstractRNN, hidden_size: int, num_samples: int, mc_dropout: float, weight_decay: float,
                 prior_scale: float, window_size: int, predictor_layers: Iterable[int],
                 data_length: Optional[int] = None, **unused: Any):
        """
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
        predictor_layers: Iterable[int]
            Layer sizes for MLP as some sort of iterable.
        data_length: Optional[int]
            Number of data points used.
        """
        super().__init__(
            model=model, hidden_size=hidden_size, num_samples=num_samples, mc_dropout=mc_dropout,
            weight_decay=weight_decay, prior_scale=prior_scale, data_length=data_length, **unused
        )

        # Initialize additional parts of model to make it more adaptive
        self.device = self.model.device
        self.window_size = window_size
        self.predictor = StepPredictor(predictor_layers, hidden_size, window_size).to(self.device)
        self.hidden_buffer = []  # Save hidden states
        self._buffer_copy = []  # Save training buffer when testing the model

    def train(self):
        """ When model mode changes, erase buffer. """
        super().train()
        # Either use new, empty buffer or continue with buffer used before model was switched to testing mode
        self.hidden_buffer = self._buffer_copy

    def test(self):
        """ When model mode changes, erase buffer. """
        super().test()
        self._buffer_copy = self.hidden_buffer
        self.hidden_buffer = []
        self.mc_dropout_layer.train()  # Don't switch dropout to eval

    def _determine_step_size(self, hidden: Tensor, device: torch.device) -> float:
        """
        Determine recoding step size. In this case, the current hidden activations are added to a window of previous
        hidden states and used with a MLP to predict the appropriate step size.

        Parameters
        ----------
        hidden: Tensor
            Current hidden state used to determine step size.
        device: torch.device
            Torch device the model is being trained on (e.g. "cpu" or "cuda").

        Returns
        -------
        step_size: float
            Predicted step size.
        """
        # TODO: Re-write buffer as actually registered PyTorch buffer

        hidden = self.select(hidden)
        hidden = hidden.detach()
        num_layers, batch_size, _ = hidden.size()
        hidden = hidden.view(batch_size, num_layers, self.hidden_size)

        # If buffer is empty or batch size changes (e.g. when going from training to testing), initialize it with zero
        # hidden states
        buffer_batch_size = -1 if len(self.hidden_buffer) == 0 else self.hidden_buffer[0].shape[0]
        if len(self.hidden_buffer) == 0 or buffer_batch_size != batch_size:
            self.hidden_buffer = [torch.zeros((batch_size, 1, self.hidden_size)).to(device)] * self.window_size

        # Add hidden state to buffer
        self.hidden_buffer.append(hidden)

        if len(self.hidden_buffer) > self.window_size:
            self.hidden_buffer.pop()  # If buffer is full, remove oldest element

        # Predict step size
        hidden_window = torch.cat(self.hidden_buffer, dim=2)
        step_size = self.predictor(hidden_window)
        step_size = step_size.view(1, batch_size, 1)

        return step_size

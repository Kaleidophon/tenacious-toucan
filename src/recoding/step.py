"""
Define a model with an intervention mechanism that bases its interventions on the uncertainty of a prediction.
"""

# STD
from typing import Iterable

# EXT
from torch import nn, Tensor
from torch.nn import ReLU, Sigmoid


class AdaptiveStepPredictor(nn.Module):
    """
    Function that determines the recoding step size based on a window of previous hidden states.
    """
    def __init__(self, predictor_layers: Iterable[int], hidden_size: int, window_size: int, **unused):
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

"""
Define a model with an intervention mechanism that bases its interventions on the uncertainty of a prediction.
"""

# STD
from abc import abstractmethod, ABC
from typing import Any, Iterable

# EXT
import torch
from torch import nn, Tensor
from torch.nn import ReLU, Softplus
import torch.nn.functional as F

# PROJECT
from src.utils.types import StepSize


class AbstractStepPredictor(nn.Module, ABC):
    """
    Abstract class for any kind of model that tries to determine the step size inside the encoding framework.
    """
    @abstractmethod
    def forward(self, hidden: Tensor, out: Tensor, device: torch.device, **additional: Any) -> StepSize:
        """
        Prediction step.

        Parameters
        ----------
        hidden: Tensor
            Current hidden state used to determine step size.
        out: Tensor
            Output Tensor of current time step.
        device: torch.device
            Torch device the model is being trained on (e.g. "cpu" or "cuda").

        Returns
        -------
        step_size: StepSize
            Batch size x 1 tensor of predicted step sizes per batch instance or one single float for the whole batch.
        """
        ...


class FixedStepPredictor(AbstractStepPredictor):
    """
    Simple step predictor that just outputs a constant step size, initialized in the beginning.
    """
    def __init__(self, step_size, **unused):
        super().__init__()
        self.step_size = step_size

    def forward(self, hidden: Tensor, out: Tensor, device: torch.device, **additional: Any) -> StepSize:
        """
        Prediction step.

        Parameters
        ----------
        hidden: Tensor
            Current hidden state used to determine step size.
        out: Tensor
            Output Tensor of current time step.
        device: torch.device
            Torch device the model is being trained on (e.g. "cpu" or "cuda").

        Returns
        -------
        step_size: StepSize
            Batch size x 1 tensor of predicted step sizes per batch instance or one single float for the whole batch.
        """
        return torch.Tensor([self.step_size]).to(device)


class LearnedFixedStepPredictor(AbstractStepPredictor):
    """
    Step predictor that just outputs a constant step size which is being learned during training.
    """
    def __init__(self, **unused):
        super().__init__()
        self.step_size = torch.nn.Parameter(torch.rand([1, 1]))

        self.register_parameter("Step size", self.step_size)

    def forward(self, hidden: Tensor, out: Tensor, device: torch.device, **additional: Any) -> StepSize:
        """
        Prediction step.

        Parameters
        ----------
        hidden: Tensor
            Current hidden state used to determine step size.
        out: Tensor
            Output Tensor of current time step.
        device: torch.device
            Torch device the model is being trained on (e.g. "cpu" or "cuda").

        Returns
        -------
        step_size: StepSize
            Batch size x 1 tensor of predicted step sizes per batch instance or one single float for the whole batch.
        """
        # Only allow positive step sizes - although the step size is initialized as a positive random number, it can
        # become negative through SGD updates
        return F.softplus(self.step_size.to(device))


class AdaptiveStepPredictor(AbstractStepPredictor):
    """
    Function that determines the recoding step size based on a window of previous hidden states.
    """
    def __init__(self, predictor_layers: Iterable[int], hidden_size: int, **unused: Any):
        """
        Initialize model.

        Parameters
        ----------
        predictor_layers: Iterable[int]
            Layer sizes for MLP as some sort of iterable.
        hidden_size: int
            Dimensionality of hidden activations.
        """
        super().__init__()
        self.predictor_layers = predictor_layers
        self.hidden_size = hidden_size
        self.window_size = 1

        # Init layers
        self.model = nn.Sequential()
        last_layer_size = predictor_layers[0]
        self.model.add_module("input", nn.Linear(hidden_size, last_layer_size))
        self.model.add_module("relu0", ReLU())

        for layer_n, current_layer_size in enumerate(predictor_layers[1:]):
            self.model.add_module(f"hidden{layer_n+1}", nn.Linear(last_layer_size, current_layer_size))
            self.model.add_module(f"relu{layer_n+1}", ReLU())
            last_layer_size = current_layer_size

        self.model.add_module("out", nn.Linear(last_layer_size, 1))  # Output scalar alpha_t
        self.model.add_module("relu_out", Softplus())

        # Init buffers
        self.hidden_buffer = []  # Buffer where to store hidden states
        self._buffer_copy = []  # Buffer to copy main buffer to in case the model is switching between modes

    def train(self, mode=True):
        """ When model mode changes, erase buffer. """
        # Either use new, empty buffer or continue with buffer used before model was switched to testing mode
        self.hidden_buffer = self._buffer_copy
        self.model.train(mode)

    def eval(self):
        """ When model mode changes, erase buffer. """
        self._buffer_copy = self.hidden_buffer
        self.hidden_buffer = []
        self.model.eval()

    def forward(self,  hidden: Tensor, out: Tensor, device: torch.device, **additional: Any) -> StepSize:
        """
        Prediction step.

        Parameters
        ----------
        hidden: Tensor
            Current hidden state used to determine step size.
        device: torch.device
            Torch device the model is being trained on (e.g. "cpu" or "cuda").

        Returns
        -------
        step_size: StepSize
            Batch size x 1 tensor of predicted step sizes per batch instance or one single float for the whole batch.
        """
        self._add_to_buffer(hidden, device)
        batch_size, _ = hidden.size()

        # Predict step size
        hidden_window = torch.cat(self.hidden_buffer, dim=1)
        step_size = self.model(hidden_window)

        return step_size

    def _add_to_buffer(self, hidden: Tensor, device: torch.device) -> None:
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
        # Detach from graph so gradients don't flow through them when backpropagating for recoding or main gradients
        hidden = hidden.detach()
        batch_size, _ = hidden.size()

        # If buffer is empty or batch size changes (e.g. when going from training to testing), initialize it with zero
        # hidden states
        buffer_batch_size = -1 if len(self.hidden_buffer) == 0 else self.hidden_buffer[0].shape[0]
        if len(self.hidden_buffer) == 0 or buffer_batch_size != batch_size:
            self.hidden_buffer = [
                torch.zeros((batch_size, self.hidden_size)).to(device)
                for _ in range(self.window_size)
            ]

        # Add hidden state to buffer
        self.hidden_buffer.append(hidden)

        if len(self.hidden_buffer) > self.window_size:
            self.hidden_buffer.pop(0)  # If buffer is full, remove oldest element


class DummyPredictor(AbstractStepPredictor):
    """
    Dummy predictor which always predicts a step size of zero. Useful when you want to avoid recoding to take place.
    """

    def forward(self, hidden: Tensor, out: Tensor, device: torch.device, **additional: Any) -> StepSize:
        """
        Prediction step.

        Parameters
        ----------
        hidden: Tensor
            Current hidden state used to determine step size.
        out: Tensor
            Output Tensor of current time step.
        device: torch.device
            Torch device the model is being trained on (e.g. "cpu" or "cuda").

        Returns
        -------
        step_size: StepSize
            Batch size x 1 tensor of predicted step sizes per batch instance or one single float for the whole batch.
        """
        return torch.Tensor([0]).to(device)


STEP_TYPES = {
    "fixed": FixedStepPredictor,
    "mlp": AdaptiveStepPredictor,
    "learned": LearnedFixedStepPredictor
}

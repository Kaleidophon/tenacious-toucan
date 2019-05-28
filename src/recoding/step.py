"""
Define a model with an intervention mechanism that bases its interventions on the uncertainty of a prediction.
"""

# STD
from abc import abstractmethod, ABC
from typing import Any

# EXT
import torch
from torch import nn, Tensor

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


class PerplexityStepPredictor(AbstractStepPredictor):
    """
    Determine the current step size based on the perplexity of the current target token.
    """
    def __init__(self, **unused):
        super().__init__()

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
        out = out.squeeze(1)
        target_idx = additional.get("target_idx", None)

        # If target indices are not given, just use most likely token
        if target_idx is None:
            target_idx = torch.argmax(out, dim=1, keepdim=True)
        else:
            target_idx = target_idx.unsqueeze(1)

        target_probs = torch.gather(out, 1, target_idx)
        target_probs = torch.sigmoid(target_probs)
        # If model is "unsurprised", ppl is 1, therefore anchor values at 0
        step_size = 2 ** (target_probs * -target_probs.log2()) - 1

        return step_size.data.to(device)  # Detach from graph

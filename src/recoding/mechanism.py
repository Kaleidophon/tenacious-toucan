"""
Define a recoding mechanism. This mechanism is based on the idea of an intervention mechanism (see
rnnalyse.interventions.mechanism.InterventionMechanism), but works entirely supervised. Furthermore, it can already
be employed during training, not only testing time.
"""

# STD
from abc import abstractmethod, ABC
from functools import wraps
from typing import Tuple, Dict, Type, Callable, Union

# EXT
import torch
from torch import Tensor
from torch.autograd import Variable, backward
from torch.optim import SGD, Optimizer

# PROJECT
from ..models.abstract_rnn import AbstractRNN


class RecodingMechanism(ABC):
    """
    Abstract superclass for a recoding mechanism.
    """
    @abstractmethod
    def __init__(self, model: AbstractRNN, optimizer_class: Type[Optimizer] = SGD):
        self.model = model
        self.optimizer_class = optimizer_class
        self.device = model.device

    @abstractmethod
    def recoding_func(self, input_var: Tensor, hidden: Tensor, out: Tensor, device: torch.device,
                      **additional: Dict) -> Tuple[Tensor, Tensor]:
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
        ...

    def recode(self, hidden: Tensor, delta: Tensor, optimizer: Optimizer, step_size: Union[float, Tensor],
               device: torch.device) -> Tensor:
        """
        Perform a single recoding step on the current time step's hidden activations.

        Parameters
        ----------
        hidden: Tensor
            Current hidden state.
        delta: Tensor
            Current error signal that is used to calculate the gradient w.r.t. the current hidden state.
        optimizer: Optimizer
            Optimizer used to make recoding step.
        step_size: Tensor
            Either 1 x 1 tensor / float for constant batch size or Batch_size x 1 tensor with individual_batch_size for
            all batch instances.
        device: torch.device
            Torch device the model is being trained on (e.g. "cpu" or "cuda").

        Returns
        -------
        hidden: Tensor
            Recoded activations.predictor_layers: Iterable[int]
            Layer sizes for MLP as some sort of iterable.
        """
        # Compute recoding gradients - in contrast to the usual backward() call, we calculate the derivatives
        # of a batch of values w.r.t some parameters instead of a single (loss) term
        # Idk why this works but it does
        backward(delta, grad_tensors=torch.ones(delta.shape).to(device))

        # Correct any corruptions
        hidden.grad = self.replace_nans(hidden.grad)

        # Apply step sizes
        hidden.grad = hidden.grad * step_size

        # Perform recoding
        optimizer.step()
        hidden = Variable(hidden.data)  # Detach from computational graph

        return hidden

    @staticmethod
    def _wrap_in_var(tensor: Tensor, requires_grad: bool) -> Variable:
        """
        Wrap a numpy array into a PyTorch Variable.

        Parameters
        ----------
        tensor: Tensor
            Tensor to be converted to a PyTorch Variable.
        requires_grad: bool
            Whether the variable requires the calculation of its gradients.

        Returns
        -------
        variable: Variable
            Tensor wrapped in variable.
        """
        return Variable(tensor, requires_grad=requires_grad)

    @staticmethod
    def replace_nans(tensor: Tensor) -> Tensor:
        """
        Replace nans in a PyTorch tensor with zeros.

        Parameters
        ----------
        tensor: Tensor
            Input tensor.

        Returns
        -------
        tensor: Tensor
            Tensor with nan values replaced.
        """
        tensor[tensor != tensor] = 0  # Exploit the fact that nan != nan

        return tensor

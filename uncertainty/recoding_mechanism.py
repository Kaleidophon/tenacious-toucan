"""
Define a recoding mechanism. This mechanism is based on the idea of an intervention mechanism (see
rnnalyse.interventions.mechanism.InterventionMechanism), but works entirely supervised. Furthermore, it can already
be employed during training, not only testing time.
"""

# STD
from abc import abstractmethod, ABC
from typing import Tuple, Dict, Type

# EXT
from torch import Tensor
from torch.autograd import Variable
from torch.optim import SGD, Optimizer

# PROJECT
from uncertainty.language_model import AbstractRNN


class RecodingMechanism(ABC):
    """
    Abstract superclass for a recoding mechanism.
    """
    @abstractmethod
    def __init__(self, model: AbstractRNN, optimizer_class: Type[Optimizer] = SGD):
        self.model = model
        self.optimizer_class = optimizer_class

    @abstractmethod
    def recoding_func(self, hidden: Tensor, out: Tensor, **additional: Dict) -> Tuple[Tensor, Tensor]:
        """
        Recode activations of current step based on some logic defined in a subclass.

        Parameters
        ----------
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
        ...

    @abstractmethod
    def recode(self, hidden: Tensor, delta: Tensor, step_size: float) -> Tensor:
        """
        Perform a single recoding step on the current time step's hidden activations.

        Parameters
        ----------
        hidden: Tensor
            Current hidden state.
        delta: Tensor
            Current error signal that is used to calculate the gradient w.r.t. the current hidden state.
        step_size: float
            Degree of influence of gradient on hidden state.
        """
        hidden = self._wrap_in_var(hidden, requires_grad=True)
        optimizer = self.optimizer_class(params=[hidden], lr=step_size)
        optimizer.zero_grad()

        # Compute gradients and correct any corruptions
        delta.backward()
        hidden.grad = self.replace_nans(hidden.grad)

        # Perform recoding
        optimizer.step()

        return hidden

    @staticmethod
    def _wrap_in_var(tensor: Tensor,
                     requires_grad: bool) -> Variable:
        """
        Wrap a numpy array into a PyTorch Variable.

        Parameters
        ----------
        tensor: Tensor
            Tensor to be converted to a PyTorch Variable.
        requires_grad: bool
            Whether the variable requires the calculation of its gradients.
        """
        return Variable(tensor, requires_grad=requires_grad)

    @staticmethod
    def replace_nans(tensor: Tensor) -> Tensor:
        """ Replace nans in a PyTorch tensor with zeros. """
        tensor[tensor != tensor] = 0  # Exploit the fact that nan != nan

        return tensor


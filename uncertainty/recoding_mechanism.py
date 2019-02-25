"""
Define a recoding mechanism. This mechanism is based on the idea of an intervention mechanism (see
rnnalyse.interventions.mechanism.InterventionMechanism), but works entirely supervised. Furthermore, it can already
be employed during training, not only testing time.
"""

# STD
from abc import abstractmethod, ABC
from functools import wraps
from typing import Tuple, Dict, Type, Callable, Union, Any, Iterable

# EXT
from torch import Tensor
from torch.autograd import Variable
from torch.optim import SGD, Optimizer

# PROJECT
from .abstract_rnn import AbstractRNN


class RecodingMechanism(ABC):
    """
    Abstract superclass for a recoding mechanism.
    """
    @abstractmethod
    def __init__(self, model: AbstractRNN, optimizer_class: Type[Optimizer] = SGD):
        self.model = model
        self.optimizer_class = optimizer_class

    def __call__(self,
                 forward_func: Callable) -> Callable:
        """
        Wrap the intervention function about the models forward function and return the decorated function.

        Parameters
        ----------
        forward_func: Callable
            Forward function of the model the mechanism is applied to.

        Returns
        -------
        wrapped: Callable:
            Decorated forward function.
        """
        @wraps(forward_func)
        def wrapped(input_var: Tensor, hidden: Tensor, **additional: Dict) -> Tuple[Tensor, Tensor]:

            # Start recording grads for hidden here
            out, hidden = forward_func(input_var, hidden, **additional)

            return self.recoding_func(input_var, hidden, out, **additional)

        return wrapped

    def apply(self) -> AbstractRNN:
        """
        Return an instance of the model where the recoding function decorates the model's forward function.

        Returns
        -------
        model : AbstractRNN
            Model with recoding mechanism applied to it.
        """
        self.model.forward = self(self.model.forward)  # Decorate forward function
        return self.model

    @abstractmethod
    def recoding_func(self, input_var: Tensor, hidden: Tensor, out: Tensor, **additional: Dict) -> Tuple[Tensor, Tensor]:
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
        ...

    def recode(self, hidden: Tensor, delta: Tensor, optimizer) -> Tensor:
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
        # Compute gradients and correct any corruptions
        # TODO: Questionable design decision?!
        delta = delta.mean(dim=0)
        delta.backward()
        #for d in delta:
        #    d.backward(retain_graph=True)

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


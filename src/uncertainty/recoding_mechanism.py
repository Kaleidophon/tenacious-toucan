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
    def __init__(self, model: AbstractRNN, optimizer_class: Type[Optimizer] = SGD, average_recoding: bool = True):
        self.model = model
        self.optimizer_class = optimizer_class
        self.average_recoding = average_recoding

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

    def recode(self, hidden: Tensor, delta: Tensor, optimizer: Optimizer, step_size: Union[float, Tensor]) -> Tensor:
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

        Returns
        -------
        hidden: Tensor
            Recoded activations.predictor_layers: Iterable[int]
            Layer sizes for MLP as some sort of iterable.
        """
        device = self.model.device

        # Compute recoding gradients
        # Average recoding gradients for a batch -> Higher speed, less accuracy
        # The speedup of course depends on the batch size
        if self.average_recoding:
            delta = delta.mean(dim=0)
            delta.backward()
        # Calculate recoding gradients per instance -> More computationally expensive but higher accuracy
        else:
            backward(delta, grad_tensors=torch.ones(delta.shape).to(device))  # Idk why this works but it does

        # Correct any corruptions
        hidden.grad = self.replace_nans(hidden.grad)

        # Perform recoding
        optimizer.step()
        hidden = Variable(hidden.data)  # Detach from computational graph

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

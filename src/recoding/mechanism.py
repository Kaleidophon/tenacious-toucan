"""
Define a recoding mechanism. This mechanism is based on the idea of an intervention mechanism (see
diagnnose.interventions.mechanism.InterventionMechanism), but works entirely supervised. Furthermore, it can already
be employed during training, not only testing time.
"""

# STD
from abc import abstractmethod, ABC
from typing import Tuple, Dict, Optional

# EXT
import torch
from torch import Tensor
from torch.autograd import backward

# PROJECT
from src.models.abstract_rnn import AbstractRNN
from src.utils.compatability import RNNCompatabilityMixin
from src.utils.log import StatsCollector
from src.recoding.step import FixedStepPredictor, AdaptiveStepPredictor
from src.utils.types import Device, HiddenDict, StepSize

# CONSTANTS
STEP_TYPES = {
    "fixed": FixedStepPredictor,
    "mlp": AdaptiveStepPredictor
}


class RecodingMechanism(ABC, RNNCompatabilityMixin):
    """
    Abstract superclass for a recoding mechanism.
    """
    @abstractmethod
    def __init__(self, model: AbstractRNN, step_type: str, predictor_kwargs: Dict):
        assert step_type in STEP_TYPES, \
            f"Invalid step type {step_type} found! Choose one of {', '.join(STEP_TYPES.keys())}"

        self.model = model
        self.device = model.current_device()
        self.device = model.device
        self.step_type = step_type

        # Initialize one predictor per state per layer
        #  TODO: Make GRU compatible
        self.predictors = {
            l: [
                STEP_TYPES[step_type](**predictor_kwargs).to(self.device),
                STEP_TYPES[step_type](**predictor_kwargs).to(self.device)
            ]
            for l in range(self.model.num_layers)
        }

        # Collect predictor modules and add them to the model so that parameters are learned
        for l, predictors in self.predictors.items():
            for n, p in enumerate(predictors):
                self.model.add_module(f"predictor{n}_l{l}", p)

    @abstractmethod
    def recoding_func(self, input_var: Tensor, hidden: HiddenDict, out: Tensor, device: torch.device,
                      **additional: Dict) -> Tuple[Tensor, Tensor]:
        """
        Recode activations of current step based on some logic defined in a subclass.

        Parameters
        ----------
        input_var: Tensor
            Current input variable.
        hidden: HiddenDict
            Dictionary of all hidden (and cell states) of all network layers.
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

    def recode_activations(self, hidden: HiddenDict, delta: Tensor, device: Device) -> Tuple[Tensor, HiddenDict]:
        """
        Recode all activations stored in a HiddenDict based on an error signal delta.

        Parameters
        ----------
        hidden: HiddenDict
            Dictionary of all hidden (and cell states) of all network layers.
        delta: Tensor
            Current error signal that is used to calculate the gradients w.r.t. the hidden states.
        device: torch.device
            Torch device the model is being trained on (e.g. "cpu" or "cuda").

        Returns
        -------
        new_out_dist, new_hidden: Tuple[Tensor, HiddenDict]
            New re-decoded output distribution alongside all recoded hidden activations.
        """
        # Register gradient hooks
        for l, hid in hidden.items():
            for h in hid:
                self.register_grad_hook(h)

        # Calculate gradient of uncertainty w.r.t. hidden states and make step
        self.compute_recoding_gradient(delta, device)

        # Do actual recoding step
        new_hidden = {
            l: tuple([
                # Use the step predictor for the corresponding state and layer
                self.recode(h, step_size=predictor(h, device), name=f"{name}_l{l}")
                for h, predictor, name in zip(hid, self.predictors[l], ["hx", "cx"])])  # TODO: Be LSTM / GRU agnostic
            for l, hid in hidden.items()
        }

        # Re-decode current output
        new_out_dist = self.redecode_output_dist(new_hidden)

        return new_out_dist, new_hidden

    @StatsCollector.collect_recoding_gradients
    def recode(self, hidden: Tensor, step_size: StepSize, name: Optional[str] = None) -> Tensor:
        """
        Perform a single recoding step on the current time step's hidden activations.

        Parameters
        ----------
        hidden: Tensor
            Current hidden state.
        step_size: StepSize
            Batch size x 1 tensor of predicted step sizes per batch instance or one single float for the whole batch.
        name: Optional[int]
            Optional name for the kind of activations that might be accessed by decorators.

        Returns
        -------
        hidden: Tensor
            Recoded activations.predictor_layers: Iterable[int]
            Layer sizes for MLP as some sort of iterable.
        """
        # Correct any corruptions
        hidden.grad = self.replace_nans(hidden.grad)

        # Perform recoding by doing a gradient decent step
        # Important: Detach hidden in-place here to avoid memory spill
        recoding_grad = hidden.grad
        hidden.detach_()
        new_hidden = hidden - step_size * recoding_grad

        # Another way could be also to detach new_hidden, but in this case gradients can't flow through step size,
        # which is necessary to train the parameters of models like AdaptiveStepPredictor

        return new_hidden

    @staticmethod
    @StatsCollector.collect_deltas
    def compute_recoding_gradient(delta: Tensor, device: Device) -> None:
        """
        Compute the recoding gradient of the error signal delta w.r.t to all hidden activations of the network.

        Parameters
        ----------
        delta: Tensor
            Current error signal that is used to calculate the gradient w.r.t. the current hidden state.
        device: torch.device
            Torch device the model is being trained on (e.g. "cpu" or "cuda").
        """
        # Compute recoding gradients - in contrast to the usual backward() call, we calculate the derivatives
        # of a batch of values w.r.t some parameters instead of a single (loss) term
        #
        # As far as I understood, when using backward with some vector instead of a scalar, we also supply
        # the gradients of the loss term w.r.t. the parameters. We don't have a classic scalar loss here, so just use
        # a tensor of ones instead. As this grad_tensor is multiplied with the remaining gradient following the chain
        # rule, we actually only obtain the derivative of delta w.r.t. the activations.
        # https://medium.com/@saihimalallu/how-exactly-does-torch-autograd-backward-work-f0a671556dc4 was pretty
        # helpful in realizing this.
        # Important: Do NOT use create_graph=True here, it will cause a memory spill.
        backward(delta, grad_tensors=torch.ones(delta.shape).to(device))

    def redecode_output_dist(self, new_hidden: HiddenDict) -> Tensor:
        """
        Based on the recoded activations, also re-decode the output distribution.

        Parameters
        ----------
        new_hidden: HiddenDict
            Recoded hidden activations for all layers of the network.

        Returns
        -------
        new_out_dist: Tensor
            Re-decoded output distributions.
        """
        num_layers = len(new_hidden.keys())
        new_out = self.select(new_hidden[num_layers - 1])  # Select topmost hidden activations
        new_out = self.model.dropout_layer(new_out)
        new_out_dist = self.model.predict_distribution(new_out)

        return new_out_dist

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

    @staticmethod
    def register_grad_hook(var: Tensor) -> None:
        """
        Register a hook that assigns the (recoding) gradient to a special attribute of the variable.

        Parameters
        ----------
        var: Tensor
            Variable we register the hook for.
        """
        def hook(grad: Tensor):
            # Important: Assign to .grad and not other custom attribute here or else PyTorch will allocate
            # additional space for this tensor which won't properly be freed up after using backward(), eventually
            # causing a memory spill
            var.grad = grad
        
        var.register_hook(hook)

        return var

    def train(self, mode=True):
        for predictors in self.predictors.values():
            for pred in predictors:
                pred.train(mode)

    def eval(self):
        for predictors in self.predictors.values():
            for pred in predictors:
                pred.eval()

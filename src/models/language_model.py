"""
Implementation of a simple RNN language model.
"""

# STD
from typing import Optional, Dict, Tuple

# EXT
import torch
from torch import nn, Tensor
from torch.distributions.bernoulli import Bernoulli
from overrides import overrides

# PROJECT
from src.models.abstract_rnn import AbstractRNN
from src.utils.types import HiddenDict, AmbiguousHidden, DropoutDict


class LSTMLanguageModel(AbstractRNN):
    """
    Implementation of a LSTM language model that can process inputs token-wise or in sequences.
    """
    def __init__(self, vocab_size, hidden_size, embedding_size, num_layers, dropout, device: torch.device = "cpu"):
        """
        Parameters
        ----------
        vocab_size: int
            Size of input vocabulary.
        hidden_size: int
            Dimensionality of hidden activations.
        embedding_size: int
            Dimensionality of word embeddings.
        num_layers: int
            Number of RNN layers.
        device: torch.device
            Torch device the model is being trained on (e.g. "cpu" or "cuda").
        """
        super().__init__("LSTM", hidden_size, embedding_size, num_layers, device)
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.vocab_size = vocab_size
        self.dropout_layer = nn.Dropout(dropout)

        # Define parameters
        self.gates = {}
        self.decoder = nn.Linear(hidden_size, vocab_size)

        for l in range(num_layers):
            # Input to first layer is embedding, for others it's the hidden state of the previous layer
            input_size = embedding_size if l == 0 else hidden_size

            self.gates[l] = {
                'ii': nn.Linear(input_size, hidden_size),
                'if': nn.Linear(input_size, hidden_size),
                'ig': nn.Linear(input_size, hidden_size),
                'io': nn.Linear(input_size, hidden_size),
                'hi': nn.Linear(hidden_size, hidden_size),
                'hf': nn.Linear(hidden_size, hidden_size),
                'hg': nn.Linear(hidden_size, hidden_size),
                'ho': nn.Linear(hidden_size, hidden_size),
            }

            # Add gates to modules so that their parameters are registered by PyTorch
            for gate_name, gate in self.gates[l].items():
                super().add_module(f"{gate_name}_l{l}", gate)

    @overrides
    def forward(self, input_var: Tensor, hidden: Optional[HiddenDict] = None,
                **additional: Dict) -> Tuple[Tensor, HiddenDict]:
        """
        Process a sequence of input variables.

        Parameters
        ----------
        input_var: Tensor
            Current input variable.
        hidden: Tensor
            Current hidden state.
        additional: dict
            Dictionary of additional information delivered via keyword arguments.

        Returns
        -------
        out: Tensor
            Decoded output Tensor of current time step.
        hidden: Tensor
            Hidden state of current time step after recoding.
        """
        device = self.current_device(reference=input_var)

        if hidden is None:
            batch_size = input_var.shape[0]
            hidden = {l: self.init_hidden(batch_size, device) for l in range(self.num_layers)}

        embed = self.embeddings(input_var)  # batch_size x seq_len x embedding_dim+
        embed = self.dropout_layer(embed)

        input_ = embed.squeeze(1)
        for l in range(self.num_layers):
            new_hidden = self.forward_step(l, hidden[l], input_)
            input_ = new_hidden[0]  # New hidden state becomes input for next layer
            hidden[l] = new_hidden  # Store for next step

        out = self.dropout_layer(input_)
        output = self.predict_distribution(out)

        return output, hidden

    def forward_step(self, layer: int, hidden: AmbiguousHidden, input_: Tensor) -> AmbiguousHidden:
        """
        Do a single step for a single layer inside an LSTM. Intuitively, this can be seen as an upward-step inside the
        network, going from a lower layer to the one above.

        Parameters
        ----------
        layer: int
            Current layer number.
        hidden: AmbiguousHidden
            Tuple of hidden and cell state from the previous time step.
        input_: Tensor
            Input to the current layer: Either embedding if layer = 0 or hidden state from previous layer.

        Returns
        -------
        hx, cx: AmbiguousHidden
            New hidden and cell state for this layer.
        """
        hx, cx = hidden

        # Forget gate
        f_g = torch.sigmoid(self.gates[layer]['if'](input_) + self.gates[layer]['hf'](hx))

        # Input gate
        i_g = torch.sigmoid(self.gates[layer]['ii'](input_) + self.gates[layer]['hi'](hx))

        # Output gate
        o_g = torch.sigmoid(self.gates[layer]['io'](input_) + self.gates[layer]['ho'](hx))

        # Intermediate cell state
        c_tilde_g = torch.tanh(self.gates[layer]['ig'](input_) + self.gates[layer]['hg'](hx))

        # New cell state
        cx = f_g * cx + i_g * c_tilde_g

        # New hidden state
        hx = o_g * torch.tanh(cx)

        return hx, cx

    def predict_distribution(self, output: Tensor, out_layer: Optional[nn.Module] = None):
        """
        Generate the output distribution using an affine transformation.

        Parameters
        ----------
        output: Tensor
            Decoded output Tensor of current time step.
        out_layer: nn.Module
            Layer used to transform the current output to the distribution.

        Returns
        -------
        output_dist: Tensor
            Unnormalized output distribution for current time step.
        """
        # Default to models own output layer
        out_layer = out_layer if out_layer is not None else self.decoder

        output_dist = out_layer(output)

        return output_dist


class UncertaintyLSTMLanguageModel(LSTMLanguageModel):
    """
    A LSTM Language model with an uncertainty recoding mechanism applied to it. This class is defined explicitly because
    the usual decorator functionality of the uncertainty mechanism prevents pickling of the model.
    """
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout, mechanism_class,
                 mechanism_kwargs, device: torch.device = "cpu"):
        super().__init__(vocab_size, embedding_size, hidden_size, num_layers, dropout, device)
        self.mechanism = mechanism_class(model=self, **mechanism_kwargs, device=device)

    @overrides
    def forward(self, input_var: Tensor, hidden: Optional[Tensor] = None, **additional: Dict) -> Tuple[Tensor, Tensor]:
        """
        Process a sequence of input variables.

        Parameters
        ----------
        input_var: Tensor
            Current input variable.
        hidden: Tensor
            Current hidden state.
        additional: dict
            Dictionary of additional information delivered via keyword arguments.

        Returns
        -------
        out: Tensor
            Decoded output Tensor of current time step.
        hidden: Tensor
            Hidden state of current time step after recoding.
        """
        device = self.current_device(reference=input_var)

        out, hidden = super().forward(input_var, hidden, **additional)

        new_out, new_hidden = self.mechanism.recoding_func(input_var, hidden, out, device=device, **additional)

        # Only allow recomputing out when gold token is not given and model has to guess, otherwise task is trivialized
        if "target_idx" not in additional:
            return new_out, new_hidden
        else:
            return out, new_hidden

    def train(self, mode=True):
        super().train(mode)
        self.mechanism.train(mode)

    def eval(self):
        super().eval()
        self.mechanism.eval()


def is_variational(model: AbstractRNN) -> bool:
    """
    Check whether a model is a Variational RNN.
    """
    return isinstance(model, VariationalLSTM)


class VariationalLSTM(UncertaintyLSTMLanguageModel):
    """
    Implemented the Variational LSTM from [1], where the same set of dropout masks is used throughout a batch
    for the same connections.

    [1] https://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf
    """
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout, mechanism_class,
                 mechanism_kwargs, device: torch.device = "cpu"):
        super().__init__(
            vocab_size, embedding_size, hidden_size, num_layers, dropout, mechanism_class, mechanism_kwargs, device
        )
        self.dropout_masks: DropoutDict = {}
        self.num_samples = mechanism_kwargs["num_samples"]
        self.dropout = dropout
        self.current_batch_size = None
        self.device = device

    def forward(self, input_var: Tensor, hidden: Optional[Tensor] = None, **additional: Dict):
        """
        Process a sequence of input variables.

        Parameters
        ----------
        input_var: Tensor
            Current input variable.
        hidden: Tensor
            Current hidden state.
        additional: dict
            Dictionary of additional information delivered via keyword arguments.

        Returns
        -------
        out: Tensor
            Decoded output Tensor of current time step.
        hidden: Tensor
            Hidden state of current time step after recoding.
        """
        device = self.current_device(reference=input_var)
        self.current_batch_size = input_var.shape[0]
        embed = self.embeddings(input_var)  # batch_size x seq_len x embedding_dim

        if hidden is None:
            batch_size = input_var.shape[0]
            hidden = {l: self.init_hidden(batch_size, device) for l in range(self.num_layers)}
            self.sample_masks()

        # If hiddens were already initialized before, apply inter-time step masks
        else:
            for l, hiddens in hidden.items():
                hidden[l] = [h * mask for h, mask in zip(hiddens, self.dropout_masks[f"{l}->{l}"])]

        # Repeat output for every dropout masks and reshape into pseudo-batch
        input_ = embed.squeeze(1)
        input_ = input_.repeat(self.num_samples, 1)
        input_ = input_ * self.dropout_masks["input"]

        for l in range(self.num_layers):
            new_hidden = self.forward_step(l, hidden[l], input_)
            input_ = new_hidden[0]  # New hidden state becomes input for next layer
            hidden[l] = new_hidden  # Store for next step

            # Apply inter-layer dropout mask
            input_ = input_ * self.dropout_masks[f"{l}->{l+1}"]

        out = self.predict_distribution(input_)

        new_out, new_hidden = self.mechanism.recoding_func(input_var, hidden, out, device=device, **additional)

        # Only allow recomputing out when gold token is not given and model has to guess, otherwise task is trivialized
        if "target_idx" not in additional:
            return new_out, new_hidden
        else:
            return out, new_hidden

    def sample_masks(self) -> None:
        """
        (Re-)sample a set of dropout masks for every type of connection throughout the network.
        """
        def sample_mask(shape) -> torch.Tensor:
            dist = Bernoulli(probs=torch.Tensor([1 - self.dropout]))
            return dist.sample(shape).squeeze(-1).to(self.device)

        # Sample masks for input embeddings
        self.dropout_masks["input"] = sample_mask([self.current_batch_size * self.num_samples, self.embedding_size])

        # Sample masks for all recurrent connections
        for l in range(self.num_layers):
            self.dropout_masks[f"{l}->{l}"] = [
                sample_mask([self.current_batch_size * self.num_samples, self.hidden_size]),  # hx
                sample_mask([self.current_batch_size * self.num_samples, self.hidden_size]),  # cx
            ]

        # Sample masks between layers
        for l in range(self.num_layers):
            self.dropout_masks[f"{l}->{l+1}"] = sample_mask(
                [self.current_batch_size * self.num_samples, self.hidden_size]
            )  # hx

    def init_hidden(self, batch_size: int, device: torch.device) -> AmbiguousHidden:
        """
        Initialize the hidden states for the current network.

        Parameters:
        -----------
        batch_size: int
            Batch size used for training.
        device: torch.device
            Torch device the model is being trained on (e.g. "cpu" or "cuda").

        Returns
        -------
        hidden: AmbiguousHidden
            Either one hidden state or tuple of hidden and cell state.
        """
        # In contrast to the superclasses, create a pseudo-batch of batch_size x num_samples here, where we will
        # apply a different dropout mask to every sample. These masks are sampled at once for efficiency
        hidden_zero = torch.zeros(batch_size * self.num_samples, self.hidden_size).to(device)

        if self.rnn_type == "LSTM":
            return hidden_zero, hidden_zero.clone()
        else:
            return hidden_zero

    def predict_distribution(self, output: Tensor, out_layer: Optional[nn.Module] = None):
        """
        Generate the output distribution using an affine transformation.

        Parameters
        ----------
        output: Tensor
            Decoded output Tensor of current time step.
        out_layer: nn.Module
            Layer used to transform the current output to the distribution.

        Returns
        -------
        output_dist: Tensor
            Unnormalized output distribution for current time step.
        """
        # Default to models own output layer
        out_layer = out_layer if out_layer is not None else self.decoder

        output_dist = out_layer(output)
        output_dist = output_dist.view(self.current_batch_size, self.num_samples, self.vocab_size)

        # Average over all predictions with different dropout masks
        output_dist = output_dist.mean(dim=1)

        return output_dist



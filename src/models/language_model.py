"""
Implementation of a simple RNN language model.
"""

# STD
from math import sqrt
from typing import Optional, Dict, Tuple

# EXT
import torch
from torch import nn, Tensor
from overrides import overrides
from torch.nn.functional import dropout, relu

# PROJECT
from src.models.abstract_rnn import AbstractRNN
from src.utils.types import HiddenDict, AmbiguousHidden, Device


class Decoder(nn.Module):
    """
    Implementation of the language model decoder that allows for an easy extension to multiple layers.
    """
    def __init__(self, vocab_size: int, hidden_size: int, dropout_prob: float, device: Device, layer_sizes: Tuple[int],
                 prior_scale: Optional[float] = None):
        """
        Initialize the decoder.

        Parameters
        ----------
        vocab_size: int
            Size of input vocabulary.
        hidden_size: int
            Dimensionality of hidden activations.
        dropout_prob: float
            Dropout probability.
        device: torch.device
            Torch device the model is being trained on (e.g. "cpu" or "cuda").
        layer_sizes: Tuple[int]
            Tuple of integers specifying the sizes of intermediate decoder layers.
        prior_scale: Optional[float]
            Prior scale that is used to initializes layer weights and biases if given.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.device = device
        self.prior_scale = prior_scale
        self.layer_sizes = layer_sizes
        self.layers = []

        # Initialize layers
        last_layer_size = hidden_size
        for n, layer_size in enumerate(layer_sizes):
            current_layer = nn.Linear(last_layer_size, layer_size).to(device)
            self.layers.append(current_layer)
            last_layer_size = layer_size

        out = nn.Linear(last_layer_size, vocab_size).to(device)
        self.layers.append(out)

        # Re-initialize weights with prior scale if given
        if self.prior_scale is not None:
            for layer in self.layers:
                layer.weight.data.normal_(0, sqrt(self.prior_scale))
                layer.bias.data.normal_(0, sqrt(self.prior_scale))

    def forward(self, hidden: Tensor, dropout_prob: Optional[float] = None) -> Tensor:
        """
        Run the topmost hidden activations of the network through (potentially) multiple layers and output
        the unnormalized output activations.

        Parameters
        ----------
        hidden: Tensor
            Current topmost hidden state.
        dropout_prob: Optional[float]
            Optional dropout probability. If none is specified, the one specified during init will be used.

        Returns
        -------
        out: Tensor
            Unnormalized output activations.
        """
        dropout_prob = dropout_prob if dropout_prob is not None else self.dropout_prob

        out = hidden
        for layer in self.layers:
            out = relu(layer(dropout(out, p=dropout_prob)))

        return out


class LSTMLanguageModel(AbstractRNN):
    """
    Implementation of a LSTM language model that can process inputs token-wise or in sequences.
    """
    def __init__(self, vocab_size: int, hidden_size: int, embedding_size: int, num_layers: int, dropout: float,
                 decoder_layer_sizes: Tuple[int], prior_scale: Optional[float] = None, device: torch.device = "cpu"):
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
        dropout: float
            Dropout probability.
        decoder_layer_sizes: Tuple[int]
            Tuple indicating the sizes of intermediate decoder layers.
        prior_scale: Optional[float]
            Prior scale that is used to initializes decoder layer weights and biases if given.
        device: torch.device
            Torch device the model is being trained on (e.g. "cpu" or "cuda").
        """
        super().__init__("LSTM", hidden_size, embedding_size, num_layers, device)
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.vocab_size = vocab_size
        self.dropout_layer = nn.Dropout(dropout)
        self.num_layers = num_layers

        # Define parameters
        self.gates = {}
        self.decoder = Decoder(vocab_size, hidden_size, dropout, device, decoder_layer_sizes, prior_scale=prior_scale)

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

        # This is necessary when training on multiple GPUs - the batch of hidden states is moved back to main GPU
        # after every step
        else:
            hidden = {l: (h[0].to(device), h[1].to(device)) for l, h in hidden.items()}

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
        Do a single step for a ingle layer inside an LSTM.

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
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout, mechanism_class, mechanism_kwargs,
                 decoder_layer_sizes: Tuple[int], device: torch.device = "cpu"):
        super().__init__(
            vocab_size, embedding_size, hidden_size, num_layers, dropout, decoder_layer_sizes,
            prior_scale=mechanism_kwargs.get("prior_scale", None), device=device
        )
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

"""
Implementation of a simple RNN language model.
"""

# STD
from typing import Optional, Dict, Tuple

# EXT
import torch
from torch import nn, Tensor
from overrides import overrides

# PROJECT
from src.models.abstract_rnn import AbstractRNN


class LSTMLanguageModel(AbstractRNN):
    """
    Implementation of a LSTM language model that can process inputs token-wise or in sequences.
    """
    def __init__(self, vocab_size, hidden_size, embedding_size, num_layers, device: torch.device = "cpu"):
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
        self.out_layer = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.recoding_backward = 0

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
        embed = self.embeddings(input_var)  # batch_size x seq_len x embedding_dim
        output, hidden = self.rnn(embed, hidden)  # Output: batch:size x seq_len x hidden_dim

        return output, hidden

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
        out_layer = out_layer if out_layer is not None else self.out_layer

        batch_size, seq_len, hidden_size = output.size()
        output_dist = out_layer(output.view(batch_size * seq_len, hidden_size))
        output_dist = output_dist.view(batch_size, seq_len, self.vocab_size)

        return output_dist

    def detached_predict_distribution(self, output: Tensor, out_layer: Optional[nn.Module] = None):
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
        out_layer = out_layer if out_layer is not None else self.out_layer

        batch_size, seq_len, hidden_size = output.size()
        W_out = out_layer.weight
        b_out = out_layer.bias
        output = output.view(batch_size * seq_len, hidden_size)
        output_dist = output @ W_out.detach().t() + b_out.detach()
        output_dist = output_dist.view(batch_size, seq_len, self.vocab_size)

        return output_dist

    def get_params(self):
        # TODO: Documentation
        params = []
        NHID = self.hidden_size

        for param in self.rnn._parameters.values():
            ii = param[0: NHID]
            if_ = param[NHID:2 * NHID]
            ig = param[2 * NHID:3 * NHID]
            io = param[3 * NHID:4 * NHID]
            params.extend([ii, if_, ig, io])

        return params


class UncertaintyLSTMLanguageModel(LSTMLanguageModel):
    """
    A LSTM Language model with an uncertainty recoding mechanism applied to it. This class is defined explicitly because
    the usual decorator functionality of the uncertainty mechanism prevents pickling of the model.
    """
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, mechanism_class, mechanism_kwargs,
                 device: torch.device = "cpu"):
        super().__init__(vocab_size, embedding_size, hidden_size, num_layers, device)
        self.mechanism = mechanism_class(model=self, **mechanism_kwargs)

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
        out, hidden = super().forward(input_var, hidden, **additional)

        return self.mechanism.recoding_func(input_var, hidden, out, **additional)

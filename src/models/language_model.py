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

        if hidden is None:
            batch_size = input_var.shape[0]
            hidden = self.init_hidden(batch_size, device)

        # This is necessary when training on multiple GPUs - the batch of hidden states is moved back to main GPU
        # after every step
        else:
            hidden = hidden[0].to(device), hidden[1].to(device)

        embed = self.embeddings(input_var)  # batch_size x seq_len x embedding_dim
        out, hidden = self.rnn(embed, hidden)  # Output: batch:size x seq_len x hidden_dim

        output = self.predict_distribution(out)

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
        device = self.current_device(reference=input_var)

        out, hidden = super().forward(input_var, hidden, **additional)

        return self.mechanism.recoding_func(input_var, hidden, out, device=device, **additional)

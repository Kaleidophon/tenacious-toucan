"""
Implementation of a simple RNN language model.
"""

# STD
from typing import Optional, Dict, Tuple

# EXT
from overrides import overrides
from torch import nn, Tensor

# PROJECT
from uncertainty.abstract_rnn import AbstractRNN


class LSTMLanguageModel(AbstractRNN):
    """
    Implementation of a LSTM language model that can process inputs token-wise or in sequences.
    """
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers):
        super().__init__("LSTM", hidden_size, embedding_size, num_layers)
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

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
    # TODO: Add docstring
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers):
        super().__init__("lstm")
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.num_layers = num_layers
        self.rnn = getattr(nn, "LSTM")(embedding_size, hidden_size, num_layers)
        self.out_layer = nn.Linear(hidden_size, vocab_size)

    @overrides
    def forward(self, input_var: Tensor, hidden: Optional[Tensor] = None, **additional: Dict) -> Tuple[Tensor, Tensor]:
        embed = self.embeddings(input)
        output, hidden = self.rnn(embed, hidden)

        output_dist = self.predict_distribution(output, self.out_layer)

        return output_dist, hidden

    @staticmethod
    def predict_distribution(output, out_layer):
        batch_size, seq_len, hidden_size = output.size()
        out = out_layer(output.view(batch_size * seq_len, hidden_size))
        out = out.view(batch_size, seq_len, hidden_size)

        return out

"""
Implementation of a simple RNN language model.
"""

# STD
from abc import ABC

# EXT
from overrides import overrides
from torch import nn


class AbstractRNN(ABC, nn.Module):
    """
    Abstract RNN class defining some common attributes and functions.
    """
    def __init__(self, rnn_type):
        super().__init__()
        self.rnn_type = rnn_type


class SimpleLanguageModel(AbstractRNN):
    # TODO: Add docstring
    def __init__(self, rnn_type, vocab_size, input_size, hidden_size, num_layers):
        super().__init__(rnn_type)
        self.embeddings = nn.Embedding(vocab_size, input_size)

        assert rnn_type in ("lstm", "gru")
        self.num_layers = num_layers
        self.rnn = getattr(nn, rnn_type.upper())(input_size, hidden_size, num_layers)
        self.out_layer = nn.Linear(hidden_size, vocab_size)

    @overrides
    def forward(self, input, hidden):
        embed = self.embeddings(input)
        output, hidden = self.rnn(embed, hidden)

        output_dist = self.predict_distribution(output, self.out_layer)

        # TODO: Make GRU / LSTM agnostic

        activation_dict = dict()
        activation_dict[0]["embd"] = embed
        activation_dict[self.num_layers - 1] = {
            "hx": hidden[0],
            "cx": hidden[1]
        }

        return output_dist, activation_dict

    @staticmethod
    def predict_distribution(output, out_layer):
        batch_size, seq_len, hidden_size = output.size()
        out = out_layer(output.view(batch_size * seq_len, hidden_size))
        out = out.view(batch_size, seq_len, hidden_size)

        return out

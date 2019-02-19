"""
Implementation of a simple RNN language model.
"""

# EXT
from overrides import overrides
from torch import nn
from rnnalyse.models.intervention_lstm import InterventionLSTM


class LanguageModel(nn.Module, InterventionLSTM):
    # TODO: Add docstring
    def __init__(self, rnn_type, vocab_size, input_size, hidden_size, num_layers):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, input_size)

        assert rnn_type in ("lstm", "gru")
        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, num_layers)
        self.out_layer = nn.Linear(hidden_size, vocab_size)

    @overrides
    def forward(self, input, hidden):
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

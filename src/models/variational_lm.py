"""
Module that defines a Variational Dropout LSTM Language modelling which also uses recoding.
"""

# STD
from typing import Optional, Dict

# EXT
import torch
from torch import Tensor, nn
from torch.distributions import Bernoulli

# PROJECT
from src.models.abstract_rnn import AbstractRNN
from src.models.recoding_lm import RecodingLanguageModel
from src.utils.types import DropoutDict, AmbiguousHidden


def is_variational(model: AbstractRNN) -> bool:
    """
    Check whether a model is a Variational RNN.
    """
    return isinstance(model, VariationalLSTM)


class VariationalLSTM(RecodingLanguageModel):
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
        self.current_batch_size = input_var.shape[0]
        embed = self.embeddings(input_var)  # batch_size x seq_len x embedding_dim

        if hidden is None:
            batch_size = input_var.shape[0]
            hidden = {l: self.init_hidden(batch_size, self.device) for l in range(self.num_layers)}
            self.sample_masks()

        # If hiddens were already initialized before, apply inter-time step masks
        else:
            for l, hiddens in hidden.items():
                hidden[l] = [h * mask for h, mask in zip(hiddens, self.dropout_masks[f"{l}->{l}"])]

        # Repeat output for every dropout masks and reshape into pseudo-batch
        input_ = embed.squeeze(1)
        input_ = input_.repeat(self.num_samples + 1, 1)
        input_ = input_ * self.dropout_masks["input"]

        for l in range(self.num_layers):
            new_hidden = self.forward_step(l, hidden[l], input_)
            input_ = new_hidden[0]  # New hidden state becomes input for next layer
            hidden[l] = new_hidden  # Store for next step

            # Apply inter-layer dropout mask
            input_ = input_ * self.dropout_masks[f"{l}->{l+1}"]

        out = self.predict_distribution(input_[:self.current_batch_size, :])

        new_out, new_hidden = self.mechanism.recoding_func(
            input_var, hidden, out, device=self.device, **additional
        )

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
            dist = Bernoulli(probs=torch.Tensor([self.dropout]))
            return dist.sample(shape).squeeze(-1).to(self.device)

        # Sample masks for input embeddings
        self.dropout_masks["input"] = torch.cat(
            (torch.ones(self.current_batch_size, self.embedding_size),
            sample_mask([self.current_batch_size * self.num_samples, self.embedding_size]))
        )

        # Sample masks for all recurrent connections
        for l in range(self.num_layers):
            self.dropout_masks[f"{l}->{l}"] = [
                torch.cat(
                    (torch.ones(self.current_batch_size, self.hidden_size),
                    sample_mask([self.current_batch_size * self.num_samples, self.hidden_size]))  # hx
                ),
                torch.cat(
                    (torch.ones(self.current_batch_size, self.hidden_size),
                    sample_mask([self.current_batch_size * self.num_samples, self.hidden_size]))  # cx,
                )
            ]

        # Sample masks between layers
        for l in range(self.num_layers):
            self.dropout_masks[f"{l}->{l+1}"] = torch.cat(
                (torch.ones(self.current_batch_size, self.hidden_size),
                sample_mask([self.current_batch_size * self.num_samples, self.hidden_size]))  # hx
            )

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
        hidden_zero = torch.zeros(batch_size * (self.num_samples + 1), self.hidden_size).to(device)

        if self.rnn_type == "LSTM":
            return hidden_zero, hidden_zero.clone()
        else:
            return hidden_zero

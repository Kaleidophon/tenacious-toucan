"""
This modules defines some function to test models.
"""

# STD
import math
from typing import Tuple

# EXT
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

# PROJECT
from src.corpus.corpora import WikiCorpus
from src.models.abstract_rnn import AbstractRNN
from src.utils.types import Device
from src.utils.compatability import RNNCompatabilityMixin


def evaluate_model(model: AbstractRNN, test_set: WikiCorpus, batch_size: int, device: Device,
                   perplexity: bool = False) -> Tuple[float, float]:
    """
    Evaluate a model on a given test set.

    Parameters
    ----------
    model: AbstractRNN
        Model to be trained.
    test_set: Optional[WikiCorpus]
        Validation set the model is being evaluated on.
    batch_size: int
        Batch size used for training.
    device: Device
        Torch device the model is being trained on (e.g. "cpu" or "cuda").
    perplexity: bool
        Indicate whether perplexity should be returned instead of the loss.

    Returns
    -------
    test_loss: float
        Loss on test set.
    """
    unk_idx = test_set.vocab["<unk>"]
    dataloader = DataLoader(test_set, batch_size=batch_size, drop_last=True)
    loss = CrossEntropyLoss(reduction=("sum" if perplexity else None), ignore_index=unk_idx).to(device)
    test_metric = 0
    total_length = 0
    hidden = None

    model.eval()
    for batch in dataloader:
        batch_size, seq_len = batch.shape

        for t in range(seq_len - 1):
            input_vars = batch[:, t].unsqueeze(1).to(device)  # Make input vars batch_size x 1
            output_dist, hidden = model(input_vars, hidden)
            output_dist = output_dist.squeeze(1)
            current_loss = loss(output_dist, batch[:, t + 1].to(device)).item()

            test_metric += current_loss

        total_length += (seq_len - 1) * batch_size

        hidden = RNNCompatabilityMixin.hidden_compatible(hidden, func=lambda h: Variable(h.data))

    model.train()

    if perplexity:
        test_metric /= total_length
        test_metric = math.exp(test_metric)

    return test_metric

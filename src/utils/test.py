"""
This modules defines some function to test models.
"""

# STD
import math
from typing import Tuple

# EXT
import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss

# PROJECT
from src.utils.corpora import WikiCorpus
from src.models.abstract_rnn import AbstractRNN
from src.utils.types import Device
from src.utils.compatability import RNNCompatabilityMixin as CompatibleRNN


def evaluate_model(model: AbstractRNN, test_set: WikiCorpus, batch_size: int, device: Device,
                   perplexity: bool = False, ignore_unk: bool = False) -> Tuple[float, float]:
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
    ignore_unk: bool
        Determine whether target <unk> tokens should be ignored when computing the test metric.

    Returns
    -------
    test_loss: float
        Loss on test set.
    """
    unk_idx = test_set.vocab["<unk>"]
    loss = CrossEntropyLoss(reduction="sum").to(device)
    test_metric = 0
    global_norm = 0
    hidden = None
    test_set.create_batches(batch_size, repeat=False, drop_last=False, device=device)

    model.eval()
    for batch in test_set:
        batch_size, seq_len = batch.shape

        for t in range(seq_len - 1):
            input_vars = batch[:, t].unsqueeze(1).to(device)  # Make input vars batch_size x 1
            output_dist, hidden = model(input_vars, hidden, target_idx=batch[:, t+1].to(device))
            output_dist = output_dist.squeeze(1)

            # Calculate loss where the target is not <unk>
            target_indices = batch[:, t + 1] != unk_idx if ignore_unk else torch.ones(batch_size, dtype=torch.long)
            targets = batch[target_indices, t + 1].to(device)
            target_output_dist = output_dist[target_indices, :]

            current_loss = loss(target_output_dist, targets).item()

            global_norm += targets.shape[0]
            test_metric += current_loss

        hidden = {l: CompatibleRNN.map(h, func=lambda h: Variable(h.data)) for l, h in hidden.items()}

    model.train()

    if perplexity:
        test_metric /= global_norm
        test_metric = math.exp(test_metric)

    return test_metric

"""
This modules defines some function to test models.
"""

# STD
import math
import time
from typing import Tuple, Optional

# EXT
from diagnnose.models.w2i import W2I
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable

# PROJECT
from src.utils.corpora import Corpus, read_wiki_corpus
from src.models.abstract_rnn import AbstractRNN
from src.utils.types import Device
from src.utils.compatability import RNNCompatabilityMixin as CompatibleRNN


def evaluate_model(model: AbstractRNN, test_set: Corpus, batch_size: int, device: Device, perplexity: bool = False,
                   give_gold: bool = True, return_speed: bool = False) -> Tuple[float, float]:
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
    give_gold: bool
        Determine whether recoding models are given the next gold target to compute the recoding signal or have to take
        their best guess.
    return_speed: bool
        Flag that indicates whether the processing speed should be returned as well.

    Returns
    -------
    test_loss: float
        Loss on test set.
    """
    loss = CrossEntropyLoss(reduction="sum").to(device)
    test_metric = 0
    global_norm = 0
    hidden = None
    test_set.create_batches(batch_size, repeat=False, drop_last=False, device=device)
    start = time.time()

    model.eval()
    for batch, targets in test_set:
        # Batch and targets come out here with seq_len x batch_size
        # So invert batch here so batch dimension is first and flatten targets later
        batch.t_()
        batch_size, seq_len = batch.shape

        for t in range(seq_len):
            input_vars = batch[:, t].to(device)
            output_dist, hidden = model(input_vars, hidden, target_idx=targets[t, :].to(device) if give_gold else None)

            current_targets = targets[t, :].to(device)
            current_loss = loss(output_dist, current_targets).item()

            global_norm += current_targets.shape[0]
            test_metric += current_loss

        hidden = {l: CompatibleRNN.map(h, func=lambda h: Variable(h.data)) for l, h in hidden.items()}

    model.train()
    end = time.time()
    duration = end - start
    speed = len(test_set) / duration

    if perplexity:
        test_metric /= global_norm
        test_metric = math.exp(test_metric)

    if return_speed:
        return test_metric, speed

    else:
        return test_metric


def load_test_set(corpus_dir: str, max_seq_len: int, vocab: Optional[W2I] = None,
                  stop_after: Optional[int] = None) -> Corpus:
    """
    Load the test set.
    """
    test_set = read_wiki_corpus(corpus_dir, "test", max_seq_len=max_seq_len, vocab=vocab,
                                stop_after=stop_after)

    return test_set

"""
Train the model with the uncertainty-based intervention mechanism.
"""

# STD
from argparse import ArgumentParser
import time
from typing import Optional

# EXT
import numpy as np
from rnnalyse.config.setup import ConfigSetup
import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.nn.utils.clip_grad import clip_grad_norm

# PROJECT
from corpus.corpora import read_wiki_corpus, WikiCorpus
from uncertainty.abstract_rnn import AbstractRNN
from uncertainty.language_model import LSTMLanguageModel
from uncertainty.uncertainty_recoding import UncertaintyMechanism

# TODO:
# Debug new model
# Test new model training


def main():
    # Manage config
    required_args = {"embedding_size", "hidden_size", "num_layers", "corpus_dir"}
    arg_groups = {
        "model": {"embedding_size", "hidden_size", "num_layers"},
        "train": {"weight_decay", "learning_rate", "batch_size", "num_epochs", "clip", "print_every", "eval_every",
                  "model_save_path"},
        "corpus": {"corpus_dir"},
        "recoding": {"predictor_layers", "window_size", "num_samples", "dropout_prob", "prior_scale",
                     "hidden_size", "weight_decay"},
    }
    argparser = init_argparser()
    config_object = ConfigSetup(argparser, required_args, arg_groups)
    config_dict = config_object.config_dict

    # Retrieve config options
    corpus_dir = config_dict["corpus"]["corpus_dir"]

    # Load data
    start = time.time()
    # TODO: Change this back, this is just for debugging
    train_set = read_wiki_corpus(corpus_dir, "valid", max_sentence_len=35, stop_after=100, load_torch=False)
    valid_set = read_wiki_corpus(corpus_dir, "test", max_sentence_len=35, stop_after=10, load_torch=False)
    #test_set = read_wiki_corpus(corpus_dir, "test", vocab=train_set.vocab)
    end = time.time()
    duration = end - start
    minutes, seconds = divmod(duration, 60)
    print(f"Data loading took {int(minutes)} minute(s), {seconds:.2f} second(s).")
    #torch.save(train_set, f"{corpus_dir}/train.pt")
    #torch.save(valid_set, f"{corpus_dir}/valid.pt")
    #torch.save(test_set, f"{corpus_dir}/test.pt")

    # Initialize model
    vocab_size = len(train_set.vocab)
    N = len(train_set)
    model = LSTMLanguageModel(vocab_size, **config_dict["model"])
    mechanism = UncertaintyMechanism(model, **config_dict["recoding"], data_length=N)
    #model = mechanism.apply()
    train_model(model, train_set, **config_dict["train"], valid_set=valid_set)


def train_model(model: AbstractRNN, train_set: WikiCorpus, learning_rate: float, num_epochs: int, batch_size: int,
                weight_decay: float, clip: float, print_every: int, eval_every: int,
                valid_set: Optional[WikiCorpus] = None, model_save_path: Optional[str] = None) -> None:
    """
    Training loop for model.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay,  amsgrad=True)
    dataloader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    num_batches = len(dataloader)
    loss = CrossEntropyLoss(reduction="sum")  # Don't average
    hidden = None
    total_batch_i = 0
    best_validation_loss = np.inf

    for epoch in range(num_epochs):
        epoch_loss = 0

        for batch_i, batch in enumerate(dataloader):

            seq_len = batch.shape[1]
            optimizer.zero_grad()
            batch_loss = 0

            for t in range(seq_len - 1):
                input_vars = batch[:, t].unsqueeze(1)  # Make input vars batch_size x 1
                out, hidden = model(input_vars, hidden)
                out = out.squeeze(1)
                batch_loss += loss(out, batch[:, t+1])

            if (total_batch_i + 1) % print_every == 0:
                print(f"Epoch {epoch+1:>3} | Batch {batch_i+1:>4}/{num_batches} | Training Loss: {batch_loss:.4f}")

            # Backward pass
            batch_loss /= batch_size
            epoch_loss += batch_loss
            batch_loss.backward(retain_graph=True)
            clip_grad_norm(batch_loss, clip)
            optimizer.step()

            # TODO: Make this model-agnostic and use detach()
            hidden = Variable(hidden[0]), Variable(hidden[1])  # Detach from history
            total_batch_i += 1

        # Calculate validation loss
        if (epoch + 1) % eval_every == 0 and valid_set is not None:
            validation_loss = evaluate_model(model, valid_set, batch_size)
            print(f"Epoch {epoch+1:>3} | Batch {batch_i+1:>4}/{num_batches} | Validation Loss: {validation_loss:.4f}")

            if validation_loss < best_validation_loss and model_save_path is not None:
                model = add_model_info(model, epoch, epoch_loss.detach().numpy(), validation_loss)
                torch.save(model, model_save_path)
                best_validation_loss = validation_loss


def evaluate_model(model: AbstractRNN, test_set: WikiCorpus, batch_size: int) -> float:
    """ Evaluate a model on a given test set. """
    dataloader = DataLoader(test_set, batch_size=batch_size)
    loss = CrossEntropyLoss()
    test_loss = 0
    hidden = None

    model.eval()
    for batch in dataloader:
        seq_len = batch.shape[1]

        for t in range(seq_len - 1):
            input_vars = batch[:, t].unsqueeze(1)  # Make input vars batch_size x 1
            out, hidden = model(input_vars, hidden)
            out = out.squeeze(1)
            test_loss += loss(out, batch[:, t + 1])

    model.train()

    return test_loss.detach().numpy()


def add_model_info(model: AbstractRNN, epoch, train_loss, validation_loss):
    model.info = {
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": validation_loss
    }

    return model


def init_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    from_config = parser.add_argument_group('From config file', 'Provide full experiment setup via config file')
    from_config.add_argument('-c', '--config', help='Path to json file containing classification config.')
    from_cmd = parser.add_argument_group('From commandline', 'Specify experiment setup via commandline arguments')

    from_cmd.add_argument("--embedding_size", type=int, help="Dimensionality of word embeddings.")
    from_cmd.add_argument("--hidden_size", type=int, help="Dimensionality of hidden states.")
    from_cmd.add_argument("--num_layers", type=int, help="Number of network layers.")
    from_cmd.add_argument("--dropout_prob", type=float, help="Dropout probability when estimating uncertainty.")
    from_cmd.add_argument("--weight_decay", type=float, help="Weight decay parameter when estimating uncertainty.")
    from_cmd.add_argument("--prior_scale", type=float,
                          help="Prior length scale. A lower scale signifies a prior belief that the input data is "
                               "distributed infrequently, a higher scale does the opposite.")
    from_cmd.add_argument("--learning_rate", type=float, help="Learning rate during training.")
    from_cmd.add_argument("--batch_size", type=int, help="Batch size during training.")
    from_cmd.add_argument("--num_epochs", type=int, help="Number of training epochs.")
    from_cmd.add_argument("--corpus_dir", type=str, help="Directory to corpus files.")
    from_cmd.add_argument("--clip", type=float, help="Threshold for gradient clipping.")
    from_cmd.add_argument("--print_every", type=int, help="Batch interval at which training info should be printed.")
    from_cmd.add_argument("--eval_every", type=int,
                          help="Epoch interval at which the model should be evaluated on validation set.")
    from_cmd.add_argument("--model_save_dir", type=str, help="Directory to which current best model should be saved to.")

    return parser


if __name__ == "__main__":
    main()

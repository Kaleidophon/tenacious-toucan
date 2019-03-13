"""
Train the model with the uncertainty-based intervention mechanism.
"""

# STD
from argparse import ArgumentParser
import sys
import time
from typing import Optional, Dict, Any, Tuple

# EXT
import numpy as np
from rnnalyse.config.setup import ConfigSetup
import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, DataParallel
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter
from tqdm import tqdm

# PROJECT
from src.corpus.corpora import read_wiki_corpus, WikiCorpus
from src.models.abstract_rnn import AbstractRNN
from src.recoding.uncertainty import AdaptingUncertaintyMechanism, UncertaintyMechanism
from src.models.language_model import LSTMLanguageModel, UncertaintyLSTMLanguageModel
from src.utils.compatability import RNNCompatabilityMixin
from src.utils.logging import log_tb_data, log_to_file

# GLOBALS
WRITER = None
MODEL_NAME = None


def main():
    config_dict = manage_config()

    # Init logging
    init_writer(config_dict)

    # Load data
    train_set, valid_set = load_data(config_dict)

    # Initialize model
    model = init_model(config_dict, vocab_size=len(train_set.vocab), corpus_size=len(train_set))

    # Train
    log_dir = config_dict["logging"]["log_dir"]
    train_model(model, train_set, **config_dict['train'], valid_set=valid_set, log_dir=log_dir)


def train_model(model: AbstractRNN, train_set: WikiCorpus, learning_rate: float, num_epochs: int, batch_size: int,
                weight_decay: float, clip: float, print_every: int, eval_every: int, device: torch.device,
                valid_set: Optional[WikiCorpus] = None, model_save_path: Optional[str] = None,
                log_dir: Optional[str] = None, **unused: Any) -> None:
    """
    Training loop for model.

    Parameters
    ----------
    model: AbstractRNN
        Model to be trained.
    train_set: WikiCorpus
        Training set.
    learning_rate: float
        Learning rate used for optimizer.
    num_epochs: int
        Number of training epochs.
    batch_size: int
        Batch size used for training.
    weight_decay: float
        L2-regularization parameter.
    clip: float
        Threshold used for gradient clipping.
    print_every: int
        Interval at which training loss is being printed.
    eval_every: int
        Interval at which model is evaluated on the validation set.
    device: torch.device
        Torch device the model is being trained on (e.g. "cpu" or "cuda").
    valid_set: Optional[WikiCorpus]
        Validation set the model is being evaluated on.
    model_save_path: Optional[str]
        Path the best model is being saved to if given.
    log_dir: Optional[str]
        Path log data is being written to if given.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=True)
    dataloader = DataLoader(train_set, shuffle=True, batch_size=batch_size, drop_last=True)
    num_batches = len(dataloader)
    loss = CrossEntropyLoss(reduction="sum").to(device)  # Don't average
    total_batch_i = 0
    hidden = None
    best_validation_loss = np.inf

    # Changing format to avoid redundant information
    bar_format = "{desc}{percentage:3.0f}% {bar} | {elapsed} < {remaining} | {rate_fmt}\n"
    with tqdm(total=num_epochs * num_batches, file=sys.stdout, bar_format=bar_format) as progress_bar:

        for epoch in range(num_epochs):
            epoch_loss = 0

            for batch_i, batch in enumerate(dataloader):

                batch_size, seq_len = batch.shape
                optimizer.zero_grad()
                batch_loss = 0

                for t in range(seq_len - 1):
                    input_vars = batch[:, t].unsqueeze(1).to(device)  # Make input vars batch_size x 1
                    output_dist, hidden = model(input_vars, hidden, target_idx=batch[:, t+1].to(device))
                    output_dist = output_dist.squeeze(1)
                    batch_loss += loss(output_dist, batch[:, t+1].to(device))

                # Backward pass
                batch_loss /= batch_size
                epoch_loss += batch_loss.item()
                batch_loss.backward(retain_graph=True)
                clip_grad_norm_(batch_loss, clip)
                optimizer.step()

                # Detach from history so the computational graph from the previous sentence doesn't get carried over
                hidden = RNNCompatabilityMixin.hidden_compatible(hidden, func=lambda h: Variable(h.data))
                total_batch_i += 1

                if (total_batch_i + 1) % print_every == 0:
                    progress_bar.set_description(
                        f"Epoch {epoch+1:>3} | Batch {batch_i+1:>4}/{num_batches} | Train Loss: {batch_loss:.4f}",
                        refresh=False
                    )
                    progress_bar.update(print_every)

                # Log
                batch_loss = float(batch_loss.cpu().detach())
                log_to_file({"batch_num": total_batch_i, "batch_loss": batch_loss}, f"{log_dir}/{MODEL_NAME}_train.log")
                log_tb_data(WRITER, f"data/batch_loss/{MODEL_NAME}", batch_loss, total_batch_i)

            # Calculate validation loss
            if (epoch + 1) % eval_every == 0 and valid_set is not None:
                validation_loss = evaluate_model(model, valid_set, batch_size, device)
                progress_bar.set_description(f"Epoch {epoch+1:>3} | Val Loss: {validation_loss:.4f}")

                if validation_loss < best_validation_loss and model_save_path is not None:
                    model_info = add_model_info(model, epoch, epoch_loss, validation_loss)
                    torch.save(model, model_save_path)
                    best_validation_loss = validation_loss

                    log_tb_data(WRITER, f"data/best_model/{MODEL_NAME}/", model_info, total_batch_i)

                log_to_file(
                    {"batch_num": total_batch_i, "val_loss": validation_loss}, f"{log_dir}/{MODEL_NAME}_val.log"
                )
                log_tb_data(WRITER, f"data/val_loss/{MODEL_NAME}/", validation_loss, total_batch_i)


def evaluate_model(model: AbstractRNN, test_set: WikiCorpus, batch_size: int, device: torch.device) -> float:
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
    device: torch.device
        Torch device the model is being trained on (e.g. "cpu" or "cuda").

    Returns
    -------
    test_loss: float
        Loss on test set.
    """
    dataloader = DataLoader(test_set, batch_size=batch_size, drop_last=True)
    loss = CrossEntropyLoss().to(device)
    test_loss = 0
    hidden = None

    model.eval()
    for batch in dataloader:
        batch_size, seq_len = batch.shape

        for t in range(seq_len - 1):
            input_vars = batch[:, t].unsqueeze(1).to(device)  # Make input vars batch_size x 1
            output_dist, hidden = model(input_vars, hidden)
            output_dist = output_dist.squeeze(1)
            test_loss += loss(output_dist, batch[:, t + 1].to(device)).item()

        hidden = RNNCompatabilityMixin.hidden_compatible(hidden, func=lambda h: Variable(h.data))

    model.train()

    return test_loss


def add_model_info(model: AbstractRNN, epoch: int, train_loss: float, validation_loss: float, **misc: Dict) -> dict:
    """
    Add information about the training conditions to a model.
    """
    model.info = {
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": validation_loss
    }
    model.info.update(**misc)

    return model.info


def init_model(config_dict: dict, vocab_size: int, corpus_size: int) -> LSTMLanguageModel:
    """
    Initialize the model for training.
    """
    # Set device for training
    device = config_dict['train']['device']
    if not (device == "cuda" and torch.cuda.is_available()):
        config_dict["train"]["device"] = "cpu"

    print(f"Using {device} for training...")

    # Init model
    model_type = config_dict["general"]["model_type"]
    mechanism_kwargs = config_dict["recoding"]
    mechanism_kwargs["data_length"] = corpus_size

    if model_type == "vanilla":
        model = LSTMLanguageModel(vocab_size, **config_dict["model"], device=device)

    elif model_type == "fixed_step":
        model = UncertaintyLSTMLanguageModel(
            vocab_size, **config_dict["model"],
            mechanism_class=UncertaintyMechanism, mechanism_kwargs=mechanism_kwargs, device=device
        )

    elif model_type == "mlp_step":
        model = UncertaintyLSTMLanguageModel(
            vocab_size, **config_dict["model"],
            mechanism_class=AdaptingUncertaintyMechanism, mechanism_kwargs=mechanism_kwargs, device=device
        )

    else:
        raise Exception("Invalid model type chosen!")

    # Distribute model over GPUs
    multi_gpu = config_dict["train"]["multi_gpu"]
    num_gpus = torch.cuda.device_count()

    if device != "cpu" and multi_gpu and num_gpus > 1:
        print(f"Using {num_gpus} GPUs for training...")
        model = DataParallel(model)

        batch_size = config_dict["train"]["batch_size"]

        if batch_size % num_gpus > 0:
            raise ValueError(
                f"Batch size {batch_size} should be divisible by number of GPUs ({num_gpus}) to avoid problems!"
            )

    model.to(device)

    return model


def load_data(config_dict) -> Tuple[WikiCorpus, WikiCorpus]:
    """
    Load training and validation set.
    """
    corpus_dir = config_dict["corpus"]["corpus_dir"]
    max_sentence_len = config_dict["corpus"]["max_sentence_len"]

    start = time.time()
    train_set = read_wiki_corpus(corpus_dir, "train", max_sentence_len=max_sentence_len)
    valid_set = read_wiki_corpus(corpus_dir, "valid", max_sentence_len=max_sentence_len, vocab=train_set.vocab)
    end = time.time()
    duration = end - start
    minutes, seconds = divmod(duration, 60)

    print(f"Data loading took {int(minutes)} minute(s), {seconds:.2f} second(s).")

    return train_set, valid_set


def manage_config() -> dict:
    """
    Parse a config file (if given), overwrite with command line arguments and return everything as dictionary
    of different config groups.
    """
    required_args = {"embedding_size", "hidden_size", "num_layers", "corpus_dir", "model_type"}
    arg_groups = {
        "general": {"model_type"},
        "model": {"embedding_size", "hidden_size", "num_layers"},
        "train": {"weight_decay", "learning_rate", "batch_size", "num_epochs", "clip", "print_every", "eval_every",
                  "model_save_path", "device", "model_name", "multi_gpu"},
        "logging": {"log_dir", "layout"},
        "corpus": {"corpus_dir", "max_sentence_len"},
        "recoding": {"predictor_layers", "window_size", "num_samples", "dropout_prob", "prior_scale", "hidden_size",
                     "weight_decay", "step_size"},
    }
    argparser = init_argparser()
    config_object = ConfigSetup(argparser, required_args, arg_groups)
    config_dict = config_object.config_dict

    return config_dict


def init_writer(config_dict: dict) -> None:
    """
    Initialize the tensorboardX writer.
    """
    log_dir = config_dict["logging"]["log_dir"]
    layout = config_dict["logging"]["layout"]

    # Initialize writer
    if log_dir is not None:
        global WRITER, MODEL_NAME
        WRITER = SummaryWriter(log_dir)
        MODEL_NAME = config_dict["train"]["model_name"]

        if layout is not None:
            custom_layout = {
                "Losses": {
                    "train_loss": ["Multiline", layout], "val_loss": ["Multiline", layout]
                }
            }
            WRITER.add_custom_scalars(custom_layout)


def init_argparser() -> ArgumentParser:
    """
    Init the parser that parses additional command line arguments. These overwrite the default options
    that were defined in a config if -c / --config is given.
    """
    parser = ArgumentParser()
    from_config = parser.add_argument_group('From config file', 'Provide full experiment setup via config file')
    from_config.add_argument('-c', '--config', help='Path to json file containing classification config.')
    from_cmd = parser.add_argument_group('From commandline', 'Specify experiment setup via commandline arguments')

    # Model options
    from_cmd.add_argument("--model_type", type=str, choices=["vanilla", "fixed_step", "mlp_step"],
                          help="Model type used for training. Choices include a vanilla Language Model, a "
                               "uncertainty-based model with fixed step size and an uncertainty based-model with a "
                               "step-sized parameterized by a MLP.")
    from_cmd.add_argument("--embedding_size", type=int, help="Dimensionality of word embeddings.")
    from_cmd.add_argument("--hidden_size", type=int, help="Dimensionality of hidden states.")
    from_cmd.add_argument("--num_layers", type=int, help="Number of network layers.")
    from_cmd.add_argument("--dropout_prob", type=float, help="Dropout probability when estimating uncertainty.")
    from_cmd.add_argument("--step_size", type=str, help="Step-size for fixed-step uncertainty-based recoding.")
    from_cmd.add_argument("--num_samples", type=int, help="Number of samples used when estimating uncertainty.")

    # Training options
    from_cmd.add_argument("--weight_decay", type=float, help="Weight decay parameter when estimating uncertainty.")
    from_cmd.add_argument("--prior_scale", type=float,
                          help="Prior length scale. A lower scale signifies a prior belief that the input data is "
                               "distributed infrequently, a higher scale does the opposite.")
    from_cmd.add_argument("--learning_rate", type=float, help="Learning rate during training.")
    from_cmd.add_argument("--batch_size", type=int, help="Batch size during training.")
    from_cmd.add_argument("--num_epochs", type=int, help="Number of training epochs.")
    from_cmd.add_argument("--clip", type=float, help="Threshold for gradient clipping.")
    from_cmd.add_argument("--multi_gpu", action="store_true", default=None,
                          help="Flag to indicate whether multiple GPUs should be used for training (if available).")

    # Corpus options
    from_cmd.add_argument("--corpus_dir", type=str, help="Directory to corpus files.")
    from_cmd.add_argument("--max_sentence_len", type=int, help="Maximum sentence length when reading in the corpora.")

    # Screen output optins
    from_cmd.add_argument("--print_every", type=int, help="Batch interval at which training info should be printed.")
    from_cmd.add_argument("--eval_every", type=int,
                          help="Epoch interval at which the model should be evaluated on validation set.")

    # Model saving and logging options
    from_cmd.add_argument("--model_name", type=str, help="Model identifier.")
    from_cmd.add_argument("--model_save_dir", type=str, help="Directory to which current best model should be saved to.")
    from_cmd.add_argument("--device", type=str, default="cpu", help="Device used for training.")
    from_cmd.add_argument("--log_dir", type=str, help="Directory to write (tensorboard) logs to.")
    from_cmd.add_argument("--layout", type=list, default=None,
                            help="Define which models should be grouped together on tensorboard. Layout here is a list "
                                 "of tags corresponding to the models.")

    return parser


if __name__ == "__main__":
    main()

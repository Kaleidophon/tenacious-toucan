"""
Train the model with the uncertainty-based intervention mechanism.
"""

# STD
from argparse import ArgumentParser
import math
import sys
from typing import Optional, Any

# EXT
import numpy as np
from diagnnose.config.setup import ConfigSetup
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss, DataParallel
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

# PROJECT
from src.utils.corpora import load_data
from src.utils.test import evaluate_model
from src.utils.corpora import WikiCorpus
from src.models.abstract_rnn import AbstractRNN
from src.recoding.anchored_ensemble import AnchoredEnsembleMechanism, has_anchored_ensemble
from src.recoding.mc_dropout import MCDropoutMechanism
from src.recoding.perplexity import PerplexityRecoding
from src.models.language_model import LSTMLanguageModel, UncertaintyLSTMLanguageModel
from src.utils.compatability import RNNCompatabilityMixin as CompatibleRNN
from src.utils.log import remove_logs, log_to_file, StatsCollector
from src.utils.types import Device

# GLOBALS
RECODING_TYPES = {
    "ensemble": AnchoredEnsembleMechanism,
    "perplexity": PerplexityRecoding,
    "mc_dropout": MCDropoutMechanism
}


def main():
    config_dict = manage_config()

    # Load data
    corpus_dir = config_dict["corpus"]["corpus_dir"]
    max_sentence_len = config_dict["corpus"]["max_sentence_len"]
    train_set, valid_set = load_data(corpus_dir, max_sentence_len)

    # Initialize model
    model = init_model(config_dict, vocab_size=len(train_set.vocab), corpus_size=len(train_set))

    # Train
    log_dir = config_dict["logging"]["log_dir"]
    ignore_unk = config_dict["eval"]["ignore_unk"]
    train_model(model, train_set, **config_dict['train'], valid_set=valid_set, log_dir=log_dir, ignore_unk=ignore_unk)


def train_model(model: AbstractRNN, train_set: WikiCorpus, learning_rate: float, num_epochs: int, batch_size: int,
                weight_decay: float, clip: float, print_every: int, eval_every: int, device: Device,
                valid_set: Optional[WikiCorpus] = None, model_save_path: Optional[str] = None,
                log_dir: Optional[str] = None, ignore_unk: bool = True, model_name: str = "model",
                **unused: Any) -> None:
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
    device: Device
        Torch device the model is being trained on (e.g. "cpu" or "cuda").
    valid_set: Optional[WikiCorpus]
        Validation set the model is being evaluated on.
    model_save_path: Optional[str]
        Path the best model is being saved to if given.
    log_dir: Optional[str]
        Path log data is being written to if given.
    ignore_unk: bool
        Determine whether <unk> tokens should be ignored as targets when evaluation.
    model_name: str
        Optional model name used for logging.
    """
    remove_logs(log_dir, model_name)

    train_set.create_batches(batch_size, repeat=False, drop_last=True, device=device)
    num_batches = len(train_set)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if valid_set is not None:
        # Anneal learning rate if no improvement is seen after a while
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.25, threshold=100, patience=(num_batches // eval_every)
        )

    loss = CrossEntropyLoss().to(device)
    total_batch_i = 0
    hidden = None
    best_validation_ppl = np.inf
    stats_collector = StatsCollector()

    # Changing format to avoid redundant information
    bar_format = "{desc}{percentage:3.0f}% {bar} | {elapsed} < {remaining} | {rate_fmt}\n"
    with tqdm(total=num_epochs * num_batches, file=sys.stdout, bar_format=bar_format) as progress_bar:

        for epoch in range(num_epochs):

            for batch_i, (batch, targets) in enumerate(train_set):

                # Batch and targets come out here with seq_len x batch_size
                # So invert batch here so batch dimension is first and flatten targets later
                batch.t_()
                batch_size, seq_len = batch.shape
                optimizer.zero_grad()
                outputs = []

                for t in range(seq_len):
                    input_vars = batch[:, t]  # Make input vars batch_size x 1
                    output_dist, hidden = model(input_vars, hidden, target_idx=targets[t, :])
                    outputs.append(output_dist)

                # Backward pass
                outputs = torch.cat(outputs)
                targets = torch.flatten(targets)
                batch_loss = loss(outputs, target=targets)

                # Extra loss component for recoding with Anchored Bayesian Ensembles
                if has_anchored_ensemble(model):
                    ensemble_loss = model.mechanism.ensemble_loss
                    batch_loss += ensemble_loss

                batch_loss.backward()

                clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

                # Detach from history so the computational graph from the previous sentence doesn't get carried over
                hidden = {l: CompatibleRNN.map(h, func=lambda h: h.detach()) for l, h in hidden.items()}
                total_batch_i += 1

                if total_batch_i % print_every == 0:
                    ppl = math.exp(batch_loss)
                    progress_bar.set_description(
                        f"Epoch {epoch+1:>3} | Batch {batch_i+1:>4}/{num_batches} | LR: {learning_rate:<2} | "
                        f"Train Loss: {batch_loss:>7.3f} | Train ppl: {ppl:>7.3f}",
                        refresh=False
                    )
                    progress_bar.update(print_every)
                    learning_rate = optimizer.param_groups[0]["lr"]  # This is only for printing

                # Log
                batch_loss = float(batch_loss.cpu().detach())
                batch_stats = stats_collector.get_stats()
                batch_stats = stats_collector.flatten_stats(batch_stats)
                log_stats = {"batch_num": total_batch_i, "batch_loss": batch_loss, **batch_stats}
                log_to_file(log_stats, f"{log_dir}/{model_name}_train.log")
                stats_collector.wipe()

                # Calculate validation loss
                if (total_batch_i + 1) % eval_every == 0 and valid_set is not None:
                    validation_ppl = evaluate_model(
                        model, valid_set, batch_size, device, perplexity=True, ignore_unk=ignore_unk
                    )
                    progress_bar.set_description(f"Epoch {epoch+1:>3} | Val Perplexity: {validation_ppl:.4f}")

                    if validation_ppl < best_validation_ppl and model_save_path is not None:
                        torch.save(model, model_save_path)
                        best_validation_ppl = validation_ppl
                    else:
                        scheduler.step(validation_ppl)  # Anneal learning rate

                    log_to_file(
                        {"batch_num": total_batch_i, "val_ppl": validation_ppl}, f"{log_dir}/{model_name}_val.log"
                    )


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
    recoding_type = config_dict["general"]["recoding_type"]
    mechanism_kwargs = config_dict["recoding"]
    mechanism_kwargs["data_length"] = corpus_size
    mechanism_kwargs["predictor_kwargs"] = config_dict["step"]

    if recoding_type is None:
        model = LSTMLanguageModel(vocab_size, **config_dict["model"], device=device)

    elif recoding_type in RECODING_TYPES.keys():
        model = UncertaintyLSTMLanguageModel(
            vocab_size, **config_dict["model"], recode_output=config_dict["general"]["recode_output"],
            mechanism_class=RECODING_TYPES[recoding_type], mechanism_kwargs=mechanism_kwargs, device=device
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


def manage_config() -> dict:
    """
    Parse a config file (if given), overwrite with command line arguments and return everything as dictionary
    of different config groups.
    """
    required_args = {"embedding_size", "hidden_size", "num_layers", "corpus_dir"}
    arg_groups = {
        "general": {"recoding_type", "recode_output"},
        "model": {"embedding_size", "hidden_size", "num_layers", "dropout"},
        "train": {"weight_decay", "learning_rate", "batch_size", "num_epochs", "clip", "print_every", "eval_every",
                  "model_save_path", "device", "model_name", "multi_gpu"},
        "logging": {"log_dir"},
        "corpus": {"corpus_dir", "max_sentence_len"},
        "recoding": {"step_type", "num_samples", "mc_dropout", "prior_scale", "hidden_size", "weight_decay",
                     "data_noise"},
        "step": {"predictor_layers", "window_size", "step_size", "hidden_size"},
        "eval": {"ignore_unk"}
    }
    argparser = init_argparser()
    config_object = ConfigSetup(argparser, required_args, arg_groups)
    config_dict = config_object.config_dict

    return config_dict


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
    from_cmd.add_argument("--recoding_type", type=str, default=None, choices=["mc_dropout", "perplexity", "ensemble"],
                          help="Recoding model type used for trainign. Choices include recoding based on MC Dropout,"
                               "perplexity and anchored ensembles. If not specified, a vanilla model without recoding"
                               "is used.")
    from_cmd.add_argument("--step_type", type=str, default=None, choices=["fixed", "mlp"],
                          help="Specifies the way the step size is determined when using a recoding model.")
    from_cmd.add_argument("--step_size", type=float,
                          help="Step size for recoding in case the fixed step predictor is used.")
    from_cmd.add_argument("--embedding_size", type=int, help="Dimensionality of word embeddings.")
    from_cmd.add_argument("--hidden_size", type=int, help="Dimensionality of hidden states.")
    from_cmd.add_argument("--num_layers", type=int, help="Number of network layers.")
    from_cmd.add_argument("--mc_dropout", type=float, help="Dropout probability when estimating uncertainty.")
    from_cmd.add_argument("--dropout", type=float, help="Dropout probability for model in general.")
    from_cmd.add_argument("--num_samples", type=int, help="Number of samples used when estimating uncertainty.")
    from_cmd.add_argument("--window_size", type=int, default=None, help="Window size for adaptive step predictor.")

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
    from_cmd.add_argument("--recode_output", action="store_true", default=None,
                          help="Determine whether recoded output activations should be used to calculate the loss "
                               "during training.")

    # Corpus options
    from_cmd.add_argument("--corpus_dir", type=str, help="Directory to corpus files.")
    from_cmd.add_argument("--max_sentence_len", type=int, help="Maximum sentence length when reading in the corpora.")

    # Screen output optins
    from_cmd.add_argument("--print_every", type=int, help="Batch interval at which training info should be printed.")
    from_cmd.add_argument("--eval_every", type=int,
                          help="Epoch interval at which the model should be evaluated on validation set.")

    # Model saving and logging options
    from_cmd.add_argument("--model_name", type=str, help="Model identifier.")
    from_cmd.add_argument("--model_save_path", type=str,
                          help="Directory to which current best model should be saved to.")
    from_cmd.add_argument("--device", type=str, default="cpu", help="Device used for training.")
    from_cmd.add_argument("--log_dir", type=str, help="Directory to write (tensorboard) logs to.")

    # Eval options
    from_cmd.add_argument("--ignore_unk", action="store_true", default=None,
                          help="Whether to ignore the <unk> token during evaluation.")

    return parser


if __name__ == "__main__":
    main()

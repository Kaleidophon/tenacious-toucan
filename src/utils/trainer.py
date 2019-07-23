"""
Module defining some functions for model training.
"""

# STD
import math
import sys
from argparse import ArgumentParser
from typing import Optional, Any, Callable

# EXT
import numpy as np
import torch
from diagnnose.config.setup import ConfigSetup
from torch import optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

# PROJECT
from src.models.abstract_rnn import AbstractRNN
from src.models.language_model import LSTMLanguageModel
from src.models.recoding_lm import RecodingLanguageModel
from src.models.variational_lm import is_variational, VariationalLSTM
from src.recoding.anchored_ensemble import has_anchored_ensemble, shares_anchors, AnchoredEnsembleMechanism
from src.recoding.mc_dropout import MCDropoutMechanism
from src.recoding.perplexity import PerplexityRecoding
from src.recoding.variational import VariationalMechanism
from src.utils.compatability import RNNCompatabilityMixin as CompatibleRNN
from src.utils.corpora import Corpus
from src.utils.log import remove_logs, StatsCollector, log_to_file
from src.utils.test import evaluate_model
from src.utils.types import Device

# GLOBALS
RECODING_TYPES = {
    "ensemble": AnchoredEnsembleMechanism,
    "perplexity": PerplexityRecoding,
    "mc_dropout": MCDropoutMechanism,
    "variational": VariationalMechanism
}


def train_model(model: AbstractRNN,
                train_set: Corpus,
                learning_rate: float,
                num_epochs: int,
                batch_size: int,
                weight_decay: float,
                clip: float,
                print_every: int,
                eval_every: int,
                device: Device,
                valid_set: Optional[Corpus] = None,
                model_save_path: Optional[str] = None,
                log_dir: Optional[str] = None,
                model_name: str = "model",
                **unused: Any) -> AbstractRNN:
    """
    Training loop for several models, including:

    * Vanilla LSTM Language Model
    * LSTM LM with perplexity recoding
    * LSTM LM with MC Dropout recoding
    * LSTM LM with variational recoding
    * LSTM LM with Bayesian Anchored Ensemble recoding
      (with optional sharing of anchors and losses for all members of the ensemble)

    Disclaimer: Because all these models have slightly different requirements for the way they are trained, the
    training loop became a bit spaghetti in some places. Sorry!

    Parameters
    ----------
    model: AbstractRNN
        Model to be trained.
    train_set: Corpus
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
    model_name: str
        Optional model name used for logging.
    """
    remove_logs(log_dir, model_name)

    train_set.create_batches(batch_size, repeat=False, drop_last=True, device=device)
    num_batches = len(train_set)

    def _filter_parameters(filter_func: Callable):
        # Filter out parameters by their name using this ugly expression:
        # Split parameters into tuple of name - param, apply filter, zip the remaining instances back together and
        # discard the names
        try:
            return list(zip(*filter(filter_func, model.named_parameters())))[1]
        except IndexError:
            return []

    model_optim = {"params": _filter_parameters(lambda tpl: "predictor" not in tpl[0]),
                    "lr": learning_rate, "weight_decay": weight_decay}
    all_params = [model_optim]

    # If a recoding LM is used, train mechanism parameters separately without weight decay
    if isinstance(model, RecodingLanguageModel):
        mechanism_optim = {"params": _filter_parameters(lambda tpl: "predictor" in tpl[0]), "lr": learning_rate}
        all_params.append(mechanism_optim)

    optimizer = optim.SGD(all_params)

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
                    output_dist, hidden = model(input_vars, hidden) #, target_idx=targets[t, :])
                    outputs.append(output_dist)

                # Backward pass
                targets = torch.flatten(targets)

                # When using bayesian anchored ensembles that do not share losses
                if has_anchored_ensemble(model) and not shares_anchors(model):
                    outputs = torch.cat(outputs, dim=1)  # K x (batch_size * seq_len) x vocab_size
                    # Extra loss component for recoding with Anchored Bayesian Ensembles
                    ensemble_losses = model.mechanism.ensemble_losses

                    # Compute loss for every member of the ensemble separately
                    for k in range(model.mechanism.num_samples):
                        member_outputs = outputs[k, :, :]
                        batch_loss = loss(member_outputs) #, target=targets)
                        member_loss = batch_loss + ensemble_losses[k]
                        member_loss.backward(retain_graph=True)

                # All other models
                else:
                    outputs = torch.cat(outputs)
                    batch_loss = loss(outputs, target=targets)

                    # When losses in an anchored ensemble are shared, just add a global ensemble weight decay loss
                    if shares_anchors(model):
                        batch_loss += model.mechanism.ensemble_losses

                    batch_loss.backward()

                # For the variational RNN, sample a new set of dropout masks after every batch
                if is_variational(model):
                    model.sample_masks()

                clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

                # Detach from history so the computational graph from the previous sentence doesn't get carried over
                hidden = {l: CompatibleRNN.map(h, func=lambda h: h.detach()) for l, h in hidden.items()}
                total_batch_i += 1

                if total_batch_i % print_every == 0:
                    try:
                        ppl = math.exp(batch_loss)
                    except OverflowError:
                        ppl = sys.maxsize  # In case of integer overflow, return highest possible integer

                    progress_bar.set_description(
                        f"Epoch {epoch+1:>3} | Batch {batch_i+1:>4}/{num_batches} | LR: {learning_rate:<2} | "
                        f"Train Loss: {batch_loss:>7.3f} | Train ppl: {ppl:>7.3f}",
                        refresh=False
                    )
                    progress_bar.update(print_every)
                    learning_rate = optimizer.param_groups[0]["lr"]  # This is only for printing

                # Log
                batch_loss = batch_loss.item()
                batch_stats = stats_collector.get_stats()
                batch_stats = stats_collector.flatten_stats(batch_stats)
                log_stats = {"batch_num": total_batch_i, "batch_loss": batch_loss, **batch_stats}
                log_to_file(log_stats, f"{log_dir}/{model_name}_train.log")
                stats_collector.wipe()

                # Calculate validation loss
                if (total_batch_i + 1) % eval_every == 0 and valid_set is not None:
                    validation_ppl = evaluate_model(model, valid_set, batch_size, device, perplexity=True)
                    progress_bar.set_description(f"Epoch {epoch+1:>3} | Val Perplexity: {validation_ppl:.4f}")

                    if validation_ppl < best_validation_ppl and model_save_path is not None:
                        torch.save(model, model_save_path)
                        best_validation_ppl = validation_ppl
                    else:
                        scheduler.step(validation_ppl)  # Anneal learning rate

                    log_to_file(
                        {"batch_num": total_batch_i, "val_ppl": validation_ppl}, f"{log_dir}/{model_name}_val.log"
                    )

    return model


def init_model(config_dict: dict, vocab_size: int, corpus_size: int) -> LSTMLanguageModel:
    """
    Initialize the model for training.
    """
    # Set device for training
    device = config_dict['train']['device']
    if not ("cuda" in device and torch.cuda.is_available()):
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
        if recoding_type == "variational":
            model = VariationalLSTM(
                vocab_size, **config_dict["model"],
                mechanism_class=RECODING_TYPES[recoding_type], mechanism_kwargs=mechanism_kwargs, device=device
            )
        else:
            model = RecodingLanguageModel(
                vocab_size, **config_dict["model"],
                mechanism_class=RECODING_TYPES[recoding_type], mechanism_kwargs=mechanism_kwargs, device=device
            )

    else:
        raise Exception("Invalid model type chosen!")

    model.to(device)

    return model


def manage_config() -> dict:
    """
    Parse a config file (if given), overwrite with command line arguments and return everything as dictionary
    of different config groups.
    """
    required_args = {"embedding_size", "hidden_size", "num_layers", "corpus_dir"}
    arg_groups = {
        "general": {"recoding_type"},
        "model": {"embedding_size", "hidden_size", "num_layers", "dropout"},
        "train": {"weight_decay", "learning_rate", "batch_size", "num_epochs", "clip", "print_every", "eval_every",
                  "model_save_path", "device", "model_name"},
        "logging": {"log_dir"},
        "corpus": {"corpus_dir", "max_seq_len"},
        "recoding": {"step_type", "num_samples", "mc_dropout", "prior_scale", "hidden_size", "weight_decay",
                     "data_noise", "share_anchor", "use_cross_entropy"},
        "step": {"predictor_layers", "window_size", "step_size", "hidden_size"}
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
    from_cmd.add_argument("--recoding_type", type=str, default=None,
                          choices=["mc_dropout", "perplexity", "ensemble", "variational"],
                          help="Recoding model type used for trainign. Choices include recoding based on MC Dropout,"
                               "perplexity and anchored ensembles. If not specified, a vanilla model without recoding"
                               "is used.")
    from_cmd.add_argument("--step_type", type=str, default=None, choices=["fixed", "ppl", "mlp", "learned"],
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
    from_cmd.add_argument("--share_anchor", action="store_true", default=None,
                          help="Determine whether anchor and losses should be shared for all member of the anchored"
                               "bayesian ensemble (increases speed but looses some theoretical guarantees. Only "
                               "applicable if recoding_type=ensemble")

    # Training options
    from_cmd.add_argument("--weight_decay", type=float, help="Weight decay parameter when estimating uncertainty.")
    from_cmd.add_argument("--prior_scale", type=float,
                          help="Prior length scale. A lower scale signifies a prior belief that the input data is "
                               "distributed infrequently, a higher scale does the opposite.")
    from_cmd.add_argument("--learning_rate", type=float, help="Learning rate during training.")
    from_cmd.add_argument("--batch_size", type=int, help="Batch size during training.")
    from_cmd.add_argument("--num_epochs", type=int, help="Number of training epochs.")
    from_cmd.add_argument("--clip", type=float, help="Threshold for gradient clipping.")

    # Corpus options
    from_cmd.add_argument("--corpus_dir", type=str, help="Directory to corpus files.")
    from_cmd.add_argument("--max_seq_len", type=int, help="Maximum sentence length when reading in the corpora.")

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

    return parser

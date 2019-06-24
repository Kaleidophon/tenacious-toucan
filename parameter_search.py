"""
Perform parameter search for a certain model type.
"""

# STD
from argparse import ArgumentParser
import asyncio
import torch
import json
from typing import List, Dict, Union

# EXT
import numpy as np
from diagnnose.config.setup import ConfigSetup

# PROJECT
from src.utils.corpora import load_data, Corpus
from src.utils.trainer import train_model
from src.utils.types import Device
from src.models.recoding_lm import RecodingLanguageModel
from src.models.variational_lm import VariationalLSTM
from src.recoding.mc_dropout import MCDropoutMechanism
from src.recoding.perplexity import PerplexityRecoding
from src.recoding.variational import VariationalMechanism
from src.recoding.anchored_ensemble import AnchoredEnsembleMechanism
from src.utils.test import evaluate_model

# GLOBALS
RECODING_TYPES = {
    "ensemble": AnchoredEnsembleMechanism,
    "perplexity": PerplexityRecoding,
    "mc_dropout": MCDropoutMechanism,
    "variational": VariationalMechanism
}


def main():
    config_dict = manage_parameter_search_config()

    # Load data
    corpus_dir = config_dict["general"]["corpus_dir"]
    max_seq_len = config_dict["general"]["max_seq_len"]
    train_set, valid_set = load_data(corpus_dir, max_seq_len)

    # Generate sets of hyperparameters
    trials = generate_trials(config_dict, num_trials=config_dict["general"]["num_trials"])

    # Run parameter sweep
    distributed_run(
        train_set, valid_set, trials,
        num_gpus=config_dict["general"]["num_gpus"], search_log_path=config_dict["general"]["search_log_path"]
    )


def generate_trials(config_dict: dict, num_trials: int) -> List[Dict]:
    """
    Generate all the configurations of hyperparameters that are going to be tested out.
    """
    default_config = config_dict["default_parameters"]
    trials = [dict(default_config) for _ in range(num_trials)]  # Create new copies of the same default hyperparams
    sample_funcs = {
        "uniform": sample_uniform,
        "log": sample_log
    }

    # Replace hyperparameters by sampled ones
    for trial in trials:
        for hyperparameter, search_info in config_dict["general"]["parameter_search"].items():
            sample_range, type_ = search_info["range"], search_info["type"]
            sampled_value = sample_funcs[search_info["dist"]](*sample_range, type_=type_)
            trial[hyperparameter] = sampled_value

    return trials


def distributed_run(train_set: Corpus, valid_set: Corpus, trials: List[Dict], num_gpus: int, search_log_path: str) -> None:
    with open(search_log_path, "w") as log_file:
        # Outer loop: Launch process on all available GPUs until the number of samples is reached
        while len(trials) > 0:
            # Get the next batch of trials
            next_trials = [trials.pop(0) for _ in range(min(num_gpus, len(trials)))]
            # Always await until a batch of processes is done so all GPUs are used optimally
            loop = asyncio.get_event_loop()
            # Blocking call which returns when the hello_world() coroutine is done
            results = loop.run_until_complete(run_batch(train_set, valid_set, next_trials))
            loop.close()

            for trial, result in zip(next_trials, results):
                print(f"{str(trial)}: {result}")
                log_file.write(f"{str(trial)}: {result}\n")


async def run_batch(train_set: Corpus, valid_set: Corpus, selected_trials: List[Dict]) -> List:
    """
    Run a batch of trials where every trial is run asychronously on a different gpu.
    """
    results = await asyncio.gather(
        *[
            train_and_eval_model(train_set, valid_set, trial, device=f"cuda:{i}")
            for i, trial in enumerate(selected_trials)
        ]
    )

    return results


def train_and_eval_model(train_set: Corpus, valid_set: Corpus, config_dict: dict, device: Device) -> float:
    """
    Train and evaluate a model and return the validation score.
    """
    recoding_type = config_dict["recoding_type"]
    mechanism_kwargs = dict(config_dict)
    del mechanism_kwargs["device"]
    mechanism_kwargs["predictor_kwargs"] = dict(mechanism_kwargs)

    # Initialize model
    if recoding_type == "variational":
        model = VariationalLSTM(
            len(train_set.vocab), **config_dict,
            mechanism_class=RECODING_TYPES[recoding_type], mechanism_kwargs=mechanism_kwargs
        )
    else:
        model = RecodingLanguageModel(
            len(train_set.vocab), **config_dict,
            mechanism_class=RECODING_TYPES[recoding_type], mechanism_kwargs=mechanism_kwargs
        )
    model.to(device)

    # Train
    trained_model = train_model(model, train_set, **config_dict)

    # Evaluate
    batch_size = config_dict["batch_size"]
    val_score = evaluate_model(
        trained_model, valid_set, batch_size, device=device, perplexity=True, return_speed=False
    )

    return val_score


def sample_uniform(lower_limit: Union[float, int], upper_limit: Union[float, int], type_="float") -> Union[float, int]:
    """
    Sample hyperparameter value from a uniform distribution.
    """
    value = np.random.uniform(lower_limit, upper_limit)
    value = int(value) if type_ == "int" else value

    return value


def sample_log(lower_limit: Union[float, int], upper_limit: Union[float, int], type_="float") -> Union[float, int]:
    """
    Sample hyperparameter value from a uniform distribution and transform with a log scale.
    """
    value = np.random.exponential((upper_limit - lower_limit)/4)
    value = int(value) if type_ == "int" else value

    return value


def manage_parameter_search_config() -> dict:
    """
    Parse a config file (if given), overwrite with command line arguments and return everything as dictionary
    of different config groups.
    """
    required_args = {"corpus_dir", "num_epochs", "num_trials", "num_gpus", "model_config"}
    arg_groups = {
        "general": {"parameter_search", "corpus_dir", "num_epochs", "num_trials", "num_gpus", "model_config",
                    "max_seq_len", "search_log_path"},
    }
    argparser = init_argparser()
    config_object = ConfigSetup(argparser, required_args, arg_groups)
    config_dict = config_object.config_dict
    with open(config_dict["general"]["model_config"], "r") as model_config:
        config_dict["default_parameters"] = json.load(model_config)

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
    from_cmd.add_argument("--model_config", type=str, default=None, help="Path to model config defining parameters "
                                                                         "that are not being searched")
    from_cmd.add_argument("--recoding_type", type=str, default=None,
                          choices=["mc_dropout", "perplexity", "ensemble", "variational"],
                          help="Recoding model type used for training. Choices include recoding based on MC Dropout,"
                               "perplexity and anchored ensembles. If not specified, a vanilla model without recoding"
                               "is used.")
    from_cmd.add_argument("--step_type", type=str, default=None, choices=["fixed", "ppl", "mlp"],
                          help="Specifies the way the step size is determined when using a recoding model.")

    # Training options
    from_cmd.add_argument("--num_epochs", type=int, help="Number of training epochs.")
    from_cmd.add_argument("--num_trials", type=int, help="Number of hyperparameter configurations that should be tested.")

    # Model saving and logging options
    from_cmd.add_argument("--device", type=str, default="cpu", help="Device used for training.")
    from_cmd.add_argument("--search_log_path", type=str, help="File to write log to.")

    return parser


if __name__ == "__main__":
    main()

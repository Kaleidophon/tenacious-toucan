"""
Perform parameter search for a certain model type by generating a set of config file, where hyperparameters of interest
have been replaced by potential candidates. These candidates and the ranges they are sampled from are specified in
an extra config file. Due to some conflicts with the cluster queueing system, these configs have to be run separately
and the best set of hyperparameters have to be selected manually from the results.
"""

# STD
from argparse import ArgumentParser
import json
from typing import List, Dict, Union

# EXT
import numpy as np
from diagnnose.config.setup import ConfigSetup

# PROJECT
from src.recoding.mc_dropout import MCDropoutMechanism
from src.recoding.perplexity import SurprisalRecoding
from src.recoding.variational import VariationalMechanism
from src.recoding.anchored_ensemble import AnchoredEnsembleMechanism

# GLOBALS
RECODING_TYPES = {
    "ensemble": AnchoredEnsembleMechanism,
    "perplexity": SurprisalRecoding,
    "mc_dropout": MCDropoutMechanism,
    "variational": VariationalMechanism
}


def main():
    config_dict = manage_parameter_search_config()
    out_dir = config_dict["general"]["out_dir"]

    # Generate sets of hyperparameters
    trials = generate_trials(config_dict, num_trials=config_dict["general"]["num_trials"])

    # Dump in new config files
    for n, trial in enumerate(trials):
        dump_config(trial, f"{out_dir}/trial{n+1}.json")


def dump_config(config_dict: dict, config_path: str) -> None:
    """
    Dump a config to a .json file.
    """
    with open(config_path, "w") as config_file:
        json.dump(config_dict, config_file, sort_keys=True, indent=4)


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

    # Generate config for run
    for trial in trials:
        # Replace the number of epochs
        trial["num_epochs"] = config_dict["general"]["num_epochs"]

        # Models will not be saved
        trial["model_save_path"] = None

        # Replace other hyperparameters with candidates
        for hyperparameter, search_info in config_dict["general"]["parameter_search"].items():
            sample_range, type_ = search_info["range"], search_info["type"]
            sampled_value = sample_funcs[search_info["dist"]](*sample_range, type_=type_)
            trial[hyperparameter] = sampled_value

    return trials


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
    required_args = {"num_epochs", "num_trials", "model_config", "out_dir"}
    arg_groups = {
        "general": {"parameter_search", "corpus_dir", "num_epochs", "num_trials", "model_config", "out_dir"},
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
    from_cmd.add_argument("--out_dir", type=str, help="Directory to put the generated config files.")
    from_cmd.add_argument("--num_epochs", type=int, help="Number of training epochs.")
    from_cmd.add_argument("--num_trials", type=int, help="Number of hyperparameter configurations that should be tested.")

    # Model saving and logging options
    from_cmd.add_argument("--device", type=str, default="cpu", help="Device used for training.")

    return parser


if __name__ == "__main__":
    main()

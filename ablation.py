"""
Perform ablation experiments for baseline and recoding model.
"""

# STD
from argparse import ArgumentParser

# EXT
from diagnnose.config.setup import ConfigSetup
import numpy as np
import torch

# PROJECT
from src.models.recoding_lm import RecodingLanguageModel
from src.recoding.step import DummyPredictor
from src.utils.corpora import load_data
from src.utils.test import load_test_set, evaluate_model
from src.utils.types import Device


def main():
    config_dict = manage_config()
    weight_model_paths = config_dict["models"]["weights"]
    mechanism_model_paths = config_dict["models"].get("mechanism", None)
    device = config_dict["general"]["device"]
    corpus_dir = config_dict["general"]["corpus_dir"]
    max_seq_len = config_dict["general"]["max_seq_len"]
    batch_size = config_dict["general"]["batch_size"]

    train_set, _ = load_data(corpus_dir, max_seq_len)
    test_set = load_test_set(corpus_dir, max_seq_len, train_set.vocab)
    del train_set

    # Load the model whose weights will be used - if these models have a recoder, it will be removed
    weight_models = (torch.load(path, map_location=device) for path in weight_model_paths)

    # If paths are given, load the models whose recoder will be used for the weight models
    if mechanism_model_paths is not None:
        mechanism_models = (torch.load(path, map_location=device) for path in mechanism_model_paths)

        assert len(weight_model_paths) == len(mechanism_model_paths), \
            "Number of models with weights and mechanisms has to be equal"

        models = (
            mechanism_model.load_parameters_from(weight_model).to(device)
            for mechanism_model, weight_model in zip(mechanism_models, weight_models)
        )

    # Otherwise just use the normal model parameters and remove the recoder if given
    else:
        models = (replace_predictors(model, device) for model in weight_models)

    # Evaluate
    perplexities = []
    for i, model in enumerate(models):
        print(f"\rEvaluating model {i+1} / {len(weight_model_paths)}...", end="", flush=True)

        perplexity = evaluate_model(
            model, test_set, batch_size, device=device, perplexity=True
        )
        perplexities.append(perplexity)
        del model

    perplexities = np.array(perplexities)
    print(f"Test perplexity: {perplexities.mean():.4f} | Std. dev {perplexities.std():.4f}")


def replace_predictors(model: RecodingLanguageModel, device: Device) -> RecodingLanguageModel:
    """
    Replace predictors in model with dummy predictors so that recoding is effectively disabled.
    """
    model.mechanism.predictors = {
        l: [DummyPredictor().to(device), DummyPredictor().to(device)]
        for l in range(model.num_layers)
    }

    return model


def manage_config() -> dict:
    """
    Parse a config file (if given), overwrite with command line arguments and return everything as dictionary
    of different config groups.
    """
    required_args = {"corpus_dir", "max_seq_len", "batch_size", "weights", "device"}
    arg_groups = {"models": {"weights", "mechanism"}, "general": {"corpus_dir", "max_seq_len", "batch_size", "device"}}
    argparser = init_argparser()
    config_object = ConfigSetup(argparser, required_args, arg_groups)
    config_dict = config_object.config_dict

    return config_dict


def init_argparser() -> ArgumentParser:
    parser = ArgumentParser()

    from_config = parser.add_argument_group('From config file', 'Provide full experiment setup via config file')
    from_config.add_argument('-c', '--config', help='Path to json file containing classification config.')

    from_cmd = parser.add_argument_group('From commandline', 'Specify experiment setup via commandline arguments')

    # Model options
    from_cmd.add_argument("--weights", nargs="+", help="Path to model whose weights should be used.")
    from_cmd.add_argument("--mechanism", nargs="+", help="Path to models whose recoder should be used.")

    # Corpus options
    from_cmd.add_argument("--corpus_dir", type=str, help="Directory to corpus files.")
    from_cmd.add_argument("--max_seq_len", type=int, help="Maximum sentence length when reading in the corpora.")

    # Evaluation options
    from_cmd.add_argument("--batch_size", type=int, help="Batch size while evaluating on the test set.")
    from_cmd.add_argument("--device", type=str, default="cpu", help="Device used for evaluation.")

    return parser


if __name__ == "__main__":
    main()

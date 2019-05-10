"""
Script to evaluate a model.
"""

# STD
from argparse import ArgumentParser
from collections import defaultdict
from typing import Optional

# EXT
import numpy as np
from diagnnose.config.setup import ConfigSetup
import torch
from diagnnose.models.w2i import W2I

# PROJECT
from src.utils.corpora import WikiCorpus, read_wiki_corpus, load_data
from src.utils.test import evaluate_model


def main() -> None:
    config_dict = manage_config()

    # Get config options
    model_paths = config_dict["general"]["models"]
    batch_size = config_dict["general"]["batch_size"]
    corpus_dir = config_dict["general"]["corpus_dir"]
    max_sentence_len = config_dict["general"]["max_sentence_len"]
    device = config_dict["general"]["device"]
    give_gold = config_dict["general"]["give_gold"]

    # Load data sets
    train_set, _ = load_data(corpus_dir, max_sentence_len)
    test_set = load_test_set(corpus_dir, max_sentence_len, train_set.vocab)

    # Load models
    models = {path: torch.load(path, map_location=device) for path in model_paths}
    vanilla_models = [model for path, model in models.items() if "vanilla" in path]
    recoding_models = [model for path, model in models.items() if "ppl" in path]

    # Evaluate
    print("Evaluating...\n")
    scores = defaultdict(lambda: np.array([]))

    for i, (vanilla_model, recoding_model) in enumerate(zip(vanilla_models, recoding_models)):
        print(f"\rEvaluating model {i+1} / {len(models)}...", end="", flush=True)
        model = recoding_model
        model.load_state_dict(vanilla_model.state_dict())
        perplexity = evaluate_model(model, test_set, batch_size, device=device, perplexity=True, give_gold=give_gold)
        scores["Vanilla with recoding"] = np.append(scores["Vanilla with recoding"], perplexity)
    print("\nEvaluation results:")
    for model, perplexities in scores.items():
        mean_perpl, std_perpl = perplexities.mean(), perplexities.std()
        print(f"{model} test perplexity: {mean_perpl:.4f} | Std. dev {std_perpl:.4f}")


def _grouping_function(path: str):
    """
    Defines how model scores are grouped by their path.
    """
    model_type = path[path.rfind("/") + 1:-1]

    return model_type


def load_test_set(corpus_dir: str, max_sentence_len: int, vocab: Optional[W2I] = None) -> WikiCorpus:
    """
    Load the test set.
    """
    test_set = read_wiki_corpus(corpus_dir, "test", max_sentence_len=max_sentence_len, vocab=vocab)

    return test_set


def manage_config() -> dict:
    """
    Parse a config file (if given), overwrite with command line arguments and return everything as dictionary
    of different config groups.
    """
    required_args = {"corpus_dir", "max_sentence_len", "batch_size", "models", "device", "give_gold"}
    arg_groups = {"general": required_args}
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
    from_cmd.add_argument("--models", nargs="+", help="Path to model files.")

    # Corpus options
    from_cmd.add_argument("--corpus_dir", type=str, help="Directory to corpus files.")
    from_cmd.add_argument("--max_sentence_len", type=int, help="Maximum sentence length when reading in the corpora.")

    # Evaluation options
    from_cmd.add_argument("--batch_size", type=int, help="Batch size while evaluating on the test set.")
    from_cmd.add_argument("--device", type=str, default="cpu", help="Device used for evaluation.")
    from_cmd.add_argument("--ignore_unk", action="store_true", default=None,
                          help="Whether to ignore the <unk> token during evaluation.")
    from_cmd.add_argument("--give_gold", action="store_true", default=None,
                          help="Whether recoding models are allowed access to the next gold token in order to "
                          "calculate the recoding signal.")

    return parser


if __name__ == "__main__":
    main()

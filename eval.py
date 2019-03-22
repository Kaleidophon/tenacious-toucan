"""
Script to evaluate a model.
"""

# STD
from argparse import ArgumentParser
from collections import defaultdict
import math
from typing import Optional

# EXT
import numpy as np
from rnnalyse.config.setup import ConfigSetup
import torch
from rnnalyse.models.w2i import W2I

# PROJECT
from src.corpus.corpora import WikiCorpus, read_wiki_corpus, load_data
from src.utils.test import evaluate_model


def main() -> None:
    config_dict = manage_config()

    # Get config options
    model_paths = config_dict["general"]["models"]
    batch_size = config_dict["general"]["batch_size"]
    corpus_dir = config_dict["general"]["corpus_dir"]
    max_sentence_len = config_dict["general"]["max_sentence_len"]
    device = config_dict["general"]["device"]

    # Load data sets
    train_set, _ = load_data(corpus_dir, max_sentence_len)
    test_set = load_test_set(corpus_dir, max_sentence_len, train_set.vocab)

    # Load models
    models = {path: torch.load(path, map_location=device) for path in model_paths}

    # Evaluate
    print("Evaluating...\n")
    scores = defaultdict(lambda: np.array([]))

    for i, (model_path, model) in enumerate(models.items()):
        print(f"\rEvaluating model {i+1} / {len(models)}...", end="", flush=True)
        loss = evaluate_model(model, test_set, batch_size, device=device)
        perplexity = math.exp(loss / test_set.num_tokens)
        scores[_grouping_function(model_path)] = np.append(scores[_grouping_function(model_path)], perplexity)

    print("\nEvaluation results:")
    for model, perplexities in scores.items():
        mean_perpl, std_perpl = perplexities.mean(), perplexities.std()
        print(f"{model} test perplexity: {mean_perpl:.2f} ± {std_perpl:.2f}")


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
    required_args = {"corpus_dir", "max_sentence_len", "batch_size", "models", "device"}
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

    return parser


if __name__ == "__main__":
    main()

"""
Estimate inherent data noise empirically following the idea of [1] §3.1.3, where we just measure the MSE of a trained
model on a held out validation set.

[1] https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8215650
"""

# STD
from argparse import ArgumentParser

# EXT
from tqdm import tqdm
from diagnnose.config.setup import ConfigSetup
import torch
from torch.autograd import Variable

# PROJECT
from src.utils.corpora import load_data, read_wiki_corpus, WikiCorpus
from src.models.language_model import LSTMLanguageModel
from src.utils.compatability import RNNCompatabilityMixin as CompatibleRNN


def main():
    config_dict = manage_config()

    # Get config options
    model_paths = config_dict["general"]["models"]
    corpus_dir = config_dict["general"]["corpus_dir"]
    max_sentence_len = config_dict["general"]["max_sentence_len"]
    device = config_dict["general"]["device"]

    # Load data sets
    train_set, _ = load_data(corpus_dir, max_sentence_len)
    validation_set = read_wiki_corpus(corpus_dir, "test", max_sentence_len=max_sentence_len, vocab=train_set.vocab)

    # Load models
    models = {path: torch.load(path, map_location=device) for path in model_paths}

    estimations = [estimate_noise(validation_set, model, config_dict) for model in models.values()]
    print(f"Estimated noise across {len(models)}: ", sum(estimations) / len(models))


def estimate_noise(validation_set: WikiCorpus, model: LSTMLanguageModel, config_dict: dict) -> float:
    batch_size = config_dict["general"]["batch_size"]
    device = config_dict["general"]["device"]

    loss = torch.nn.CrossEntropyLoss(reduction="sum").to(device)
    test_metric = 0
    global_norm = 0
    hidden = None
    validation_set.create_batches(batch_size, repeat=False, drop_last=False, device=device)

    model.eval()
    for batch, targets in tqdm(validation_set):
        # Batch and targets come out here with seq_len x batch_size
        # So invert batch here so batch dimension is first and flatten targets later
        batch.t_()
        batch_size, seq_len = batch.shape

        for t in range(seq_len):
            input_vars = batch[:, t].to(device)
            output_dist, hidden = model(input_vars, hidden)

            current_targets = targets[t, :].to(device)

            current_loss = loss(output_dist, current_targets).item()

            global_norm += current_targets.shape[0]
            test_metric += current_loss

        hidden = {l: CompatibleRNN.map(h, func=lambda h: Variable(h.data)) for l, h in hidden.items()}

    estimated_noise = test_metric / global_norm

    print(estimated_noise)

    return estimated_noise


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

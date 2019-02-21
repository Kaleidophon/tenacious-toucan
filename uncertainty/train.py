"""
Train the model with the uncertainty-based intervention mechanism.
"""

# STD
from argparse import ArgumentParser

# EXT
from rnnalyse.config.setup import ConfigSetup
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import NLLLoss

# PROJECT
from corpus.corpora import read_wiki_corpus
from uncertainty.language_model import LSTMLanguageModel
from uncertainty.uncertainty_recoding import UncertaintyMechanism


def main():
    # Manage config
    required_args = {"embedding_size", "hidden_size", "num_layers", "corpus_dir"}
    arg_groups = {
        "model": {"embedding_size", "hidden_size", "num_layers"},
        "train": {"weight_decay", "learning_rate", "batch_size", "num_epochs"},
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
    train_set = read_wiki_corpus(corpus_dir, "train")
    #valid_set = read_wiki_corpus(CORPUS_DIR, "valid", vocab=train_set.vocab)
    #test_set = read_wiki_corpus(CORPUS_DIR, "test", vocab=train_set.vocab)
    #torch.save(train_set, f"{CORPUS_DIR}/train.pt")
    #torch.save(valid_set, f"{CORPUS_DIR}/valid.pt")
    #torch.save(test_set, f"{CORPUS_DIR}/test.pt")

    #train_set = torch.load(f"{CORPUS_DIR}/train.pt")
    #valid_set = torch.load(f"{CORPUS_DIR}/valid.pt")
    #test_set = torch.load(f"{CORPUS_DIR}/test.pt")

    # Initialize model
    vocab_size = len(train_set.vocab)
    N = len(train_set)
    model = LSTMLanguageModel(vocab_size, **config_dict["model"])
    mechanism = UncertaintyMechanism(model, **config_dict["recoding"], data_length=N)
    #model = mechanism.apply()
    train_model(model, train_set, **config_dict["train"])


def train_model(model, dataset, learning_rate, num_epochs, batch_size, weight_decay):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    loss = NLLLoss()

    for batch_i, batch in enumerate(dataloader):
        out, activations = model(batch)
        a = 3


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

    return parser


if __name__ == "__main__":
    main()

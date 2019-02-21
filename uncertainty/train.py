"""
Train the model with the uncertainty-based intervention mechanism.
"""

# STD
from argparse import ArgumentParser
import time

# EXT
from rnnalyse.config.setup import ConfigSetup
import torch
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
    start = time.time()
    # TODO: Change this back, it's just for debugging
    train_set = read_wiki_corpus(corpus_dir, "valid")
    #valid_set = read_wiki_corpus(corpus_dir, "valid", vocab=train_set.vocab)
    #test_set = read_wiki_corpus(corpus_dir, "test", vocab=train_set.vocab)
    end = time.time()
    duration = end - start
    minutes, seconds = divmod(duration, 60)
    print(f"Data loading took {int(minutes)} minute(s), {seconds:.2f} second(s).")
    #torch.save(train_set, f"{corpus_dir}/train.pt")
    #torch.save(valid_set, f"{corpus_dir}/valid.pt")
    #torch.save(test_set, f"{corpus_dir}/test.pt")

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
    hidden = None

    for epoch in range(num_epochs):
        for batch_i, batch in enumerate(dataloader):
            seq_len = batch.shape[1]

            for t in range(seq_len):
                input_vars = batch[:, t].unsqueeze(1)  # Make input vars batch_size x 1
                out, hidden = model(input_vars, hidden)


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

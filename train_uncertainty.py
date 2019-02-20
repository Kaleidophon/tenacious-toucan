"""
Train the model with the uncertainty-based intervention mechanism.
"""

# EXT
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import NLLLoss

# PROJECT
from corpora import read_wiki_corpus
from language_model import SimpleLanguageModel
from uncertainty import UncertaintyMechanism


def main():
    # TODO: Make these arguments command line args and add config
    RNN_TYPE = "lstm"
    INPUT_SIZE = None
    HIDDEN_SIZE = 650
    NUM_LAYERS = 2
    PREDICTOR_LAYERS = [50, 50]
    WINDOW_SIZE = 4
    NUM_SAMPLES = 10
    DROPOUT_PROB = 0.2
    WEIGHT_DECAY = 0.3
    PRIOR_SCALE = 0.01
    LEARNING_RATE = 20
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    CORPUS_DIR = "data/corpora/enwiki"

    # Load data
    train_set, vocab = read_wiki_corpus(CORPUS_DIR, "train")
    #valid_set, _ = read_wiki_corpus(CORPUS_DIR, "valid", vocab)
    #test_set, _ = read_wiki_corpus(CORPUS_DIR, "test", vocab)

    # Initialize model
    vocab_size = len(vocab)
    N = len(train_set)
    model = SimpleLanguageModel(RNN_TYPE, vocab_size, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
    mechanism = UncertaintyMechanism(
        model, PREDICTOR_LAYERS, HIDDEN_SIZE, WINDOW_SIZE, NUM_LAYERS, DROPOUT_PROB, WEIGHT_DECAY, PRIOR_SCALE, N
    )
    #model = mechanism.apply()
    train_model(model, train_set, LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, WEIGHT_DECAY)


def train_model(model, dataset, learning_rate, num_epochs, batch_size, weight_decay):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    loss = NLLLoss()

    for batch_i, batch in enumerate(dataloader):
        out, activations = model(batch)
        a = 3


if __name__ == "__main__":
    main()

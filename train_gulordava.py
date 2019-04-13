import math
import time

import torch.nn as nn

import os
import torch
from collections import defaultdict
import logging
import codecs

from train import init_model, manage_config, init_argparser
from src.utils.compatability import RNNCompatabilityMixin as CompatibleRNN


class Dictionary(object):
    def __init__(self, path):
        self.word2idx = {}
        self.idx2word = []
        self.word2freq = defaultdict(int)

        vocab_path = os.path.join(path, 'vocab.txt')
        try:
            vocab = open(vocab_path, encoding="utf8").read()
            self.word2idx = {w: i for i, w in enumerate(vocab.split())}
            self.idx2word = [w for w in vocab.split()]
            self.vocab_file_exists = True
        except FileNotFoundError:
            logging.info("Vocab file not found, creating new vocab file.")
            self.create_vocab(os.path.join(path, 'train.txt'))
            codecs.open(vocab_path,"w", "utf-8").write("\n".join([w for w in self.idx2word]))

    def add_word(self, word):
        self.word2freq[word] += 1
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        #return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def create_vocab(self, path):
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split()
                for word in words:
                    self.add_word(word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary(path)
        self.train = tokenize(self.dictionary, os.path.join(path, 'train.txt'))
        self.valid = tokenize(self.dictionary, os.path.join(path, 'valid.txt'))
        self.test = tokenize(self.dictionary, os.path.join(path, 'test.txt'))


def tokenize(dictionary, path):
    """Tokenizes a text file for training or testing to a sequence of indices format
       We assume that training and test data has <eos> symbols """
    assert os.path.exists(path)
    with open(path, 'r', encoding="utf8") as f:
        ntokens = 0
        for line in f:
            words = line.split()
            ntokens += len(words)

    # Tokenize file content
    with open(path, 'r', encoding="utf8") as f:
        ids = torch.LongTensor(ntokens)
        token = 0
        for line in f:
            words = line.split()
            for word in words:
                if word in dictionary.word2idx:
                    ids[token] = dictionary.word2idx[word]
                else:
                    ids[token] = dictionary.word2idx["<unk>"]
                token += 1

    return ids


def repackage_hidden(h):
    """Detaches hidden states from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, seq_length):
    seq_len = min(seq_length, len(source) - 1 - i)
    data = source[i:i+seq_len]
    # predict the sequences shifted by one word
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def batchify(data, bsz, cuda):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if cuda:
        data = data.cuda()
    return data

###############################################################################
# Script start
###############################################################################

config_dict = manage_config()

logging.basicConfig(level=logging.INFO)

###############################################################################
# Load data
###############################################################################

logging.info("Loading data")
corpus_dir = config_dict["corpus"]["corpus_dir"]
batch_size = config_dict["train"]["batch_size"]
device = config_dict["train"]["device"]
cuda = device == "cuda"
bptt = config_dict["corpus"]["max_sentence_len"]
clip = config_dict["train"]["clip"]
log_interval = config_dict["train"]["print_every"]
learning_rate = config_dict["train"]["learning_rate"]
save = config_dict["train"]["model_save_path"]
epochs = config_dict["train"]["num_epochs"]

start = time.time()
corpus = Corpus(corpus_dir)
logging.info("( %.2f )" % (time.time() - start))
#logging.info(corpus.train)

logging.info("Batchying..")
eval_batch_size = 10
train_data = batchify(corpus.train, batch_size, cuda)
#logging.info("Train data size", train_data.size())
val_data = batchify(corpus.valid, eval_batch_size, cuda)
test_data = batchify(corpus.test, eval_batch_size, cuda)

ntokens = len(corpus.dictionary)

criterion = nn.CrossEntropyLoss()

###############################################################################
# Build the model
###############################################################################

logging.info("Building the model")

model = init_model(config_dict, vocab_size=len(corpus.dictionary), corpus_size=len(corpus.train))


###############################################################################
# Training code
###############################################################################


def process_seq(model, hidden, data):
    outputs = []

    for t in range(data.shape[0]):
        d = data[t, :]
        output_dist, hidden = model(d, hidden)
        outputs.append(output_dist)

    outputs = torch.cat(outputs)
    return outputs, hidden


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(eval_batch_size)

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i, bptt)
            #> output has size seq_length x batch_size x vocab_size

            output, hidden = process_seq(model, hidden, data)
            #output, hidden = model(data, hidden)

            #> output_flat has size num_targets x vocab_size (batches are stacked together)
            #> ! important, otherwise softmax computation (e.g. with F.softmax()) is incorrect
            output_flat = output.view(-1, ntokens)
            #output_candidates_info(output_flat.data, targets.data)
            total_loss += len(data) * nn.CrossEntropyLoss()(output_flat, targets).data
            hidden = repackage_hidden(hidden)

    return total_loss.item() /len(data_source - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    logging.info("Vocab size %d", ntokens)

    #hidden = model.init_hidden(batch_size)
    hidden = None

    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)
        # truncated BPP
        #hidden = repackage_hidden(hidden)
        if hidden is not None:
            hidden = {l: CompatibleRNN.map(h, func=lambda h: h.detach()) for l, h in hidden.items()}
        model.zero_grad()

        #output, hidden = model(data, hidden)
        output, hidden = process_seq(model, hidden, data)

        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / log_interval
            elapsed = time.time() - start_time
            logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // bptt, lr,
                elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

# Loop over epochs.
lr = learning_rate
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, epochs +1):
        epoch_start_time = time.time()

        train()

        val_loss = evaluate(val_data)
        logging.info('-' * 89)
        logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        logging.info('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    logging.info('-' * 89)
    logging.info('Exiting from training early')

# Load the best saved model.
with open(save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data)
logging.info('=' * 89)
logging.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
logging.info('=' * 89)

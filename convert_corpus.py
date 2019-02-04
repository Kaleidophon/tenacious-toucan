"""
Convert corpus to pickle file.
"""
import pickle

from corpus import read_gulordava_corpus


CORPUS_DIR = "./data/corpora/generated"
SAVE_PATH = "./data/corpora/labeled/num_agreement.pickle"

corpus = read_gulordava_corpus(CORPUS_DIR)

with open(SAVE_PATH, 'wb') as file:
    pickle.dump(corpus, file)

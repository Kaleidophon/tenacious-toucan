"""
Quick and dirty script to convert the necessary corpora into the expected data structures.
"""

# STD
import pickle

# PROJECT
from src.corpus.corpora import read_gulordava_corpus, read_giulianelli_corpus


# Convert Gulordava corpus

print("Converting Gulordava corpus...")
GULORDAVA_CORPUS_DIR = "./data/corpora/gulordava"
GULORDAVA_SAVE_PATH = "./data/corpora/labeled/num_agreement.pickle"

gulordava_corpus = read_gulordava_corpus(GULORDAVA_CORPUS_DIR)

with open(GULORDAVA_SAVE_PATH, 'wb') as file:
    pickle.dump(gulordava_corpus, file)

# Convert Giulianelli corpus

print("Converting Giulianelli corpus...")
GIULIANELLI_CORPUS_PATHS = [
    "./data/corpora/giulianelli/wd_k0_l5_m0_a3.tsv",
    "./data/corpora/giulianelli/wd_k0_l4-7_m0_a-.tsv",
    "./data/corpora/giulianelli/wd_k0_l5_m0_a-.tsv"
]
GIULIANELLI_SAVE_PATHS = [
    "./data/corpora/labeled/wd_k0_l5_m0_a3.pickle",
    "./data/corpora/labeled/wd_k0_l4-7_m0_a-.pickle",
    "./data/corpora/labeled/wd_k0_l5_m0_a-.pickle",
]

for corpus_path, save_path in zip(GIULIANELLI_CORPUS_PATHS, GIULIANELLI_SAVE_PATHS):
    guilianelli_corpus = read_giulianelli_corpus(corpus_path)

    with open(save_path, 'wb') as file:
        pickle.dump(guilianelli_corpus, file)

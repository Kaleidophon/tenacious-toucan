"""
Convert corpus to pickle file.
"""
import pickle

from corpora import read_gulordava_corpus, read_giulianelli_corpus


# Convert Gulordava corpus

print("Converting Gulordava corpus...")
#GULORDAVA_CORPUS_DIR = "./data/corpora/generated"
#GULORDAVA_SAVE_PATH = "./data/corpora/labeled/num_agreement.pickle"

#gulordava_corpus = read_gulordava_corpus(GULORDAVA_CORPUS_DIR)

#with open(GULORDAVA_SAVE_PATH, 'wb') as file:
#    pickle.dump(gulordava_corpus, file)

# Convert Giulianelli corpus

print("Converting Giulianelli corpus...")
GIULIANELLI_CORPUS_PATH = "./data/corpora/under_the_hood/wd_k0_l5_m0_a3.tsv"
GIULIANELLI_SAVE_PATH = "./data/corpora/labeled/wd_k0_l5_m0_a3.pickle"

guilianelli_corpus = read_giulianelli_corpus(GIULIANELLI_CORPUS_PATH)

with open(GIULIANELLI_SAVE_PATH, 'wb') as file:
    pickle.dump(guilianelli_corpus, file)

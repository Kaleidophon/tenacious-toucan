"""
Convert the Penn Treebank into a plain corpus and train / validation / test splits.
"""

# STD
import re
from typing import List
import random

# CONST
WORD_PATTERN = "\s([\w\d\.\,\;\:\\\?\!`'\-\*\$\§\=" + '"' + "]+)\s?\)"
PTB_PATH = "./data/corpora/ptb_raw/penn-wsj-line.txt"
PTB_OUT_DIR = "./data/corpora/ptb/"
SPLITS = (0.8, 0.1, 0.1)


def read_ptb(path: str) -> List[str]:
    lines = []

    with open(path, "r") as file:
        for raw_line in file.readlines():
            matches = re.findall(WORD_PATTERN, raw_line.strip())

            if len(matches) > 0:
                lines.append(" ".join(matches) + "\n")

    return lines


if __name__ == "__main__":
    # Read corpus
    lines = read_ptb(PTB_PATH)

    # Split corpus
    corpus_size = len(lines)
    train_set = lines[:int(corpus_size * SPLITS[0])]
    validation_set = lines[int(corpus_size * SPLITS[0]):int(corpus_size * (SPLITS[0] + SPLITS[1]))]
    test_set = lines[int(corpus_size * (SPLITS[0] + SPLITS[1])):]

    assert sum([len(train_set), len(validation_set), len(test_set)]) == corpus_size

    # Write splits
    def write_file(lines: List[str], path: str) -> None:
        with open(path, "w") as file:
            for line in lines:
                file.write(line)

    write_file(train_set, f"{PTB_OUT_DIR}train.txt")
    write_file(validation_set, f"{PTB_OUT_DIR}valid.txt")
    write_file(test_set, f"{PTB_OUT_DIR}test.txt")

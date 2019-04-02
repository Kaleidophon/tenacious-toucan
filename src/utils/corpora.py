"""
Read corpora that used in experiments.
"""

# STD
import time
from collections import defaultdict
from typing import List, Optional, Tuple
import os

# EXT
import torch
from torch import Tensor
from torch.utils.data import Dataset
from rnnalyse.models.w2i import W2I

# PROJECT
from src.utils.types import Device


def read_gulordava_corpus(corpus_dir: str) -> dict:
    """
    Read the corpus generated by [1] with 41 sentences, of which all content words were randomly replaced with another
    word of the same form nine time, resulting in 410 sentences in total.

    [1] https://arxiv.org/pdf/1803.11138.pdf

    Parameters
    ----------
    corpus_dir: str
        Directory to corpus files.

    Returns
    -------
    labelled_corpus: dict
        Corpus of labelled sentences as a dictionary from sentence id as key to LabelledSentence object as value.
    """
    def _read_file(path: str) -> List[str]:
        with open(path, "r") as f:
            return f.readlines()

    sentences = _read_file(f"{corpus_dir}/generated.text")
    sentence_info = _read_file(f"{corpus_dir}/generated.tab")[1:]  # Skip header line
    labelled_corpus = {}

    for i, sentence in enumerate(sentences):
        right_info, wrong_info = sentence_info[2*i], sentence_info[2*i+1]

        # Parse lines
        right_info, wrong_info = right_info.split("\t"), wrong_info.split("\t")
        constr_id, sent_id, correct_number, right_form, class_, type_ = right_info[1:7]
        len_context, len_prefix, sent = right_info[11:14]
        constr_id_wrong, sent_id_wrong, _, wrong_form, class_wrong, type_wrong = wrong_info[1:7]
        sent_wrong = wrong_info[13]

        assert class_ == "correct" and class_wrong == "wrong"
        assert constr_id == constr_id_wrong and sent_id == sent_id_wrong and sent == sent_wrong and type_ == type_wrong

        len_prefix, len_context = int(len_prefix), int(len_context)
        subj_pos = len_prefix - len_context
        verb_pos = len_prefix
        sentence = sent.split()

        misc_info = {
            "raw": sent,
            "subj_pos": subj_pos,
            "verb_pos": verb_pos,
            "right_form": right_form,
            "wrong_form": wrong_form,
            "correct_number": correct_number,
            "sent_id": sent_id,
            "constr_id": constr_id,
            "type": type_
        }

        labelled_sentence = {
            "sen": sentence, "labels": [0 if correct_number == "sing" else 1] * len(sentence), **misc_info
        }
        labelled_corpus[i] = labelled_sentence

    return labelled_corpus


def read_giulianelli_corpus(corpus_path: str) -> dict:
    """
    Read the corpus created in [1] with contains sentences from the Wikidata corpus used in [2], filtered by the
    number of words preceeding the subject, the number of words following the main verb, the context size and the number
    of helpful nouns or attractor between subject and main verb.

    Parameters
    ----------
    corpus_path: str
        Directory to corpus files.

    [1] http://aclweb.org/anthology/W18-5426
    [2] https://transacl.org/ojs/index.php/tacl/article/download/972/215
    """
    labelled_corpus = {}

    with open(corpus_path, "r") as corpus_file:
        for i, line in enumerate(corpus_file.readlines()):
            line = line.strip()

            # Parse line
            raw_sentence, raw_sentence_pos, label, subj_pos, verb_pos, num_helpful, num_attractors = line.split("\t")
            sentence, sentence_pos = raw_sentence.split(), raw_sentence_pos.split()
            subj_pos, verb_pos, num_helpful, num_attractors = map(int, [subj_pos, verb_pos, num_helpful, num_attractors])

            # Organize information
            misc_info = {
                "raw": raw_sentence,
                "subj_pos": subj_pos,
                "verb_pos": verb_pos,
                "pos_tags": sentence_pos,
                "num_helpful": num_helpful,
                "num_attractors": num_attractors
            }

            labelled_sentence = {
                "sen": sentence, "labels": [0 if label == "sing" else 1] * len(sentence), **misc_info
            }
            labelled_corpus[i] = labelled_sentence

    return labelled_corpus


class WikiCorpus(Dataset):
    """ Corpus Class used to train a PyTorch Language Model. """
    def __init__(self, indexed_sentences: List[Tensor], vocab: W2I, max_sentence_len: int):
        """
        Attributes
        ----------
        indexed_sentences: List[Tensor]
            List of indexed sentences as PyTorch sentences.
        vocab: W2I
            Vocabulary as W2I object.
        """
        self.indexed_sentences = torch.cat(indexed_sentences, dim=0)
        self.seq_len = max_sentence_len
        self.vocab = vocab
        self.batches = None
        self.repeat = None
        self.num_batches = 0

    def create_batches(self, batch_size: int, repeat: bool, drop_last: bool, device: Device) -> None:
        self.repeat = repeat

        # Work out how cleanly we can divide the dataset into batch-sized parts
        num_batched_steps = self.indexed_sentences.shape[0] // batch_size

        # Trim off any extra elements that wouldn't cleanly fit (remainders)
        self.indexed_sentences = self.indexed_sentences.narrow(0, 0, num_batched_steps * batch_size)

        # Evenly divide the data across the bsz batches.
        raw_batches = self.indexed_sentences.view(batch_size, -1).t().contiguous().to(device)

        # If the last batch would be too short and drop_last is true, remove it
        if num_batched_steps % self.seq_len > 0 and drop_last:
            num_batched_steps -= num_batched_steps % self.seq_len

        self.num_batches = int(num_batched_steps / self.seq_len)

        self.batches = [raw_batches[n: n + self.seq_len].t() for n in range(self.num_batches)]

    def __iter__(self):
        if self.batches is None:
            raise ValueError("Batches have not initialized yet. Call create_batches() first.")

        while True:
            for batch in self.batches:
                yield batch
            if not self.repeat:
                return

    def __len__(self):
        return len(self.indexed_sentences) if self.batches is None else len(self.batches)

    def __getitem__(self, item):
        return self.indexed_sentences[item] if self.batches is None else self.batches[item]


def read_wiki_corpus(corpus_dir: str, corpus_split: str, max_sentence_len: Optional[int] = 50,
                     vocab: Optional[dict] = None, stop_after: Optional[int] = None) -> WikiCorpus:
    """
    Read in the Wikipedia data set used by [1] to train a language model.

    [1] https://transacl.org/ojs/index.php/tacl/article/download/972/215

    Parameters
    ----------
    corpus_dir: str
        Directory to corpus files.
    corpus_split: str
        Training split which should be read in {"train", "valid", "test"}.
    max_sentence_len: int
        Maximum sentence length in corpus. Sentences will be padded up to this length and longer sentences will be
        discarded.
    vocab: dict or None
        Dictionary from type to id. If not given, the "vocab.txt" file will be read from corpus_dir to generate this
        data structure.
    stop_after: Optional[int]
        Stop reading after some number of lines.

    Returns
    -------
    dataset: WikiCorpus
        Returns the specified dataset
    """
    def _read_vocabulary(vocab_path: str) -> W2I:
        with open(vocab_path, "r") as vocab_file:
            idx, words = zip(*enumerate(line.strip() for line in vocab_file.readlines()))
            w2i = dict(zip(words, idx))
            w2i["<pad>"] = len(w2i)
            w2i = W2I(w2i)  # Return <unk> index if word is not in vocab

            return w2i

    assert corpus_split in ("train", "valid", "test"), "Invalid split selected!"

    if vocab is None:
        print(f"Reading vocabulary under {corpus_dir}/vocab.txt...")
        if os.path.exists(f"{corpus_dir}/vocab.txt"):
            vocab = _read_vocabulary(f"{corpus_dir}/vocab.txt")
        else:
            print("No vocabulary file found, building vocabulary from scratch...")
            vocab = defaultdict(lambda: len(vocab))

    # Read in corpus
    print(f"Reading corpus under {corpus_dir}/{corpus_split}.txt...")
    indexed_sentences = []

    with open(f"{corpus_dir}/{corpus_split}.txt", "r") as corpus_file:
        for i, line in enumerate(corpus_file.readlines()):
            line = line.strip()

            # Skip empty lines
            if line in ("", "<eos>"):
                continue

            tokens = line.split()
            if tokens[-1] != "<eos>":
                tokens.append("<eos>")

            indexed_sentence = torch.LongTensor(list(map(vocab.__getitem__, tokens)))  # Index lookup
            indexed_sentences.append(indexed_sentence)

            if stop_after is not None:
                if i > stop_after:
                    break

    # If vocab was build from scratch, convert
    if not isinstance(vocab, W2I):
        vocab = W2I(vocab)

    corpus = WikiCorpus(indexed_sentences, vocab, max_sentence_len)

    return corpus


def load_data(corpus_dir: str, max_sentence_len: int) -> Tuple[WikiCorpus, WikiCorpus]:
    """
    Load training and validation set.
    """
    start = time.time()
    train_set = read_wiki_corpus(corpus_dir, "train", max_sentence_len=max_sentence_len)
    valid_set = read_wiki_corpus(corpus_dir, "valid", max_sentence_len=max_sentence_len, vocab=train_set.vocab)
    end = time.time()
    duration = end - start
    minutes, seconds = divmod(duration, 60)

    print(f"Data loading took {int(minutes)} minute(s), {seconds:.2f} second(s).")

    return train_set, valid_set

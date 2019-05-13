"""
Analyse the behaviour of two models qualitative by looking at single sentences.
"""

# STD
from argparse import ArgumentParser
from collections import namedtuple, defaultdict
from typing import List

# EXT
from diagnnose.config.setup import ConfigSetup
import matplotlib.pyplot as plt
import torch
import matplotlib.backends.backend_pdf

# PROJECT
from src.models.language_model import LSTMLanguageModel
from src.utils.corpora import load_data, WikiCorpus
from src.utils.test import load_test_set
from src.utils.types import Device

# CONSTANTS
BASELINE_COLOR = "tab:blue"
RECODING_COLOR = "tab:orange"

# TYPES
ScoredSentence = namedtuple(
    "ScoredSentence", ["sentence", "first_scores", "first_std", "second_scores", "second_std", "diff"]
)


def main():
    config_dict = manage_config()

    corpus_dir = config_dict["general"]["corpus_dir"]
    max_sentence_len = config_dict["general"]["max_sentence_len"]
    device = config_dict["general"]["device"]
    model_paths = config_dict["general"]["models"]
    pdf_path = config_dict["general"]["pdf_path"]
    num_plots = config_dict["general"]["num_plots"]

    # Load data sets
    train_set, _ = load_data(corpus_dir, max_sentence_len)
    test_set = load_test_set(corpus_dir, max_sentence_len, train_set.vocab)

    # Load models
    models = {path: torch.load(path, map_location=device) for path in model_paths}
    sentence_scores = defaultdict(list)

    # Extract perplexity scores for every word of every sentence
    for path, model in models.items():
        sentence_ppls = extract_word_perplexities(model, test_set, device)
        # Aggregate perplexity scores by model
        sentence_scores[_grouping_function(path)].append(sentence_ppls)

    assert len(sentence_scores.keys()) < 3, "Cannot perform analysis with more than two models at once."

    # Iterate over all scores produced for one sentence by all models of the same type
    first_model, second_model = sentence_scores.keys()
    inverted_vocab = {idx: word for word, idx in test_set.vocab.items()}
    scored_sentences = []
    sentences_and_scores = zip(
        test_set.indexed_sentences, zip(*sentence_scores[first_model]), zip(*sentence_scores[second_model])
    )
    for sentence, first_scores, second_scores in sentences_and_scores:
        first_scores = torch.stack(first_scores)
        first_mean, first_std = first_scores.mean(dim=0), first_scores.std(dim=0)
        second_scores = torch.stack(second_scores)
        second_mean, second_std = second_scores.mean(dim=0), second_scores.std(dim=0)
        diff = (first_mean - second_mean).abs().mean()

        orig_sentence = list(map(inverted_vocab.__getitem__, sentence.numpy()))

        scored_sentences.append(
            ScoredSentence(
                sentence=orig_sentence, first_scores=first_mean.numpy(), first_std=first_std.numpy(),
                second_scores=second_mean.numpy(), second_std=second_std.numpy(), diff=diff.item()
            )
        )

    # Sort sentences descending by average perplexity assigned to individual time steps
    sorted_scored_sentences = sorted(scored_sentences, key=lambda sen: sen.diff, reverse=True)
    name_func = lambda name: "Vanilla" if "vanilla" in name else "Recoding"

    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)

    if num_plots is None or num_plots > len(sorted_scored_sentences):
        num_plots = len(sorted_scored_sentences)

    for scored_sentence in sorted_scored_sentences[:num_plots]:
        plot_perplexities(scored_sentence, list(map(name_func, sentence_scores.keys())), pdf=pdf)

    pdf.close()


def plot_perplexities(scored_sentence,  model_names, pdf=None):
    tokens = scored_sentence.sentence[:-1]  # Exclude <eos>
    x = range(len(tokens))
    fig, ax = plt.subplots()

    # Plot data
    first_high = (scored_sentence.first_scores + scored_sentence.first_std)
    first_low = (scored_sentence.first_scores - scored_sentence.first_std)
    second_high = (scored_sentence.second_scores + scored_sentence.second_std)
    second_low = (scored_sentence.second_scores - scored_sentence.second_std)
    ax.plot(x, scored_sentence.first_scores, label=model_names[0], color=BASELINE_COLOR)
    plt.fill_between(x, first_high,  scored_sentence.first_scores, alpha=0.4, color=BASELINE_COLOR)
    plt.fill_between(x,  scored_sentence.first_scores, first_low, alpha=0.4, color=BASELINE_COLOR)

    ax.plot(x, scored_sentence.second_scores, label=model_names[1], color=RECODING_COLOR)
    plt.fill_between(x, second_high, scored_sentence.second_scores, alpha=0.4, color=RECODING_COLOR)
    plt.fill_between(x, scored_sentence.second_scores, second_low, alpha=0.4, color=RECODING_COLOR)

    plt.xticks(x, tokens, fontsize=10)

    # Draw vertical lines
    for x_ in range(len(tokens)):
        plt.axvline(x=x_, alpha=0.8, color="gray", linewidth=0.25)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    plt.tight_layout()
    plt.legend(loc="upper left")

    if pdf:
        pdf.savefig(fig)

    plt.close()


def _grouping_function(path: str):
    """
    Defines how model scores are grouped by their path.
    """
    model_type = path[path.rfind("/") + 1:-1]

    return model_type


def extract_word_perplexities(model: LSTMLanguageModel, test_set: WikiCorpus, device: Device) -> List[torch.Tensor]:
    """
    Collect the perplexities of a model for all the words in a corpus.
    """
    def perplexity(tensor: torch.Tensor) -> torch.Tensor:
        return 2 ** (-tensor * tensor.log2())

    hidden = None
    all_sentence_ppls = []

    model.eval()
    with torch.no_grad():
        for sentence in test_set:
            # Batch and targets come out here with seq_len x batch_size
            # So invert batch here so batch dimension is first and flatten targets later
            word_ppls = []

            for t in range(sentence.shape[0] - 1):
                input_vars = sentence[t].unsqueeze(0).to(device)
                output_dist, hidden = model(input_vars, hidden, target_idx=sentence[t + 1].unsqueeze(0).to(device))

                # Calculate loss where the target is not <unk>
                current_word_ppls = torch.sigmoid(output_dist[:, sentence[t + 1]])
                word_ppls.append(current_word_ppls)

            # Concat and so that we have tensors containing all the target word perplexities of all batch sentences
            sentence_word_ppls = torch.cat(word_ppls, dim=0)
            sentence_word_ppls = perplexity(sentence_word_ppls)
            all_sentence_ppls.append(sentence_word_ppls)

    return all_sentence_ppls


def manage_config() -> dict:
    """
    Parse a config file (if given), overwrite with command line arguments and return everything as dictionary
    of different config groups.
    """
    required_args = {"corpus_dir", "max_sentence_len", "batch_size", "models", "device", "pdf_path", "num_plots"}
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
    from_cmd.add_argument("--device", type=str, default="cpu", help="Device used for evaluation.")

    # Plotting options
    from_cmd.add_argument("--pdf_path", type=str, help="Path to pdf with plots.")
    from_cmd.add_argument("--num_plots", type=int, help="Number of plots included in the pdf.")

    return parser


if __name__ == "__main__":
    main()

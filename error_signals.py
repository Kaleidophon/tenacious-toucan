"""
Analyze how recoding error signals and test metric change in one or multiple models with or without recoding at one
or multiple time steps.
"""

# STD
from argparse import ArgumentParser
from collections import namedtuple, defaultdict
import sys
from typing import List, Tuple, Union, Optional
import random

# EXT
from diagnnose.config.setup import ConfigSetup
import matplotlib.pyplot as plt
import torch
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.stats import ttest_ind
from torch.nn import functional as F

# PROJECT
from src.recoding.step import DummyPredictor
from src.models.language_model import LSTMLanguageModel
from src.models.recoding_lm import RecodingLanguageModel
from src.utils.corpora import load_data, Corpus
from src.utils.log import StatsCollector
from src.utils.test import load_test_set
from src.utils.types import Device

# CONSTANTS
PLOT_COLORS = {
    "out": ["darkblue", "coral"],
    "delta": ["darkviolet", "forestgreen"]
}
LATEX_LABELS = {
    "out": r"$\mathbf{o}_t$",
    "delta": r"$\delta_t$"
}

# TYPES
ScoredSentence = namedtuple("ScoredSentence", ["sentence", "first_scores", "second_scores"])
RecodingSteps = Union[List[int], str]

# TODO: Collect step size


def main() -> None:
    config_dict = manage_config()
    first_model_paths = config_dict["general"]["first"]
    second_model_paths = config_dict["general"]["second"]
    device = config_dict["general"]["device"]
    collectables = config_dict["general"]["collectables"]
    corpus_dir = config_dict["general"]["corpus_dir"]
    max_seq_len = config_dict["general"]["max_seq_len"]
    model_names = config_dict["general"]["model_names"]
    pdf_path = config_dict["optional"]["pdf_path"]
    stop_after = config_dict["optional"]["stop_after"]
    num_plots = config_dict["optional"]["num_plots"]
    first_recoding_steps = config_dict["optional"].get("first_recoding_steps", [])
    second_recoding_steps = config_dict["optional"].get("second_recoding_steps", [])
    ttest = config_dict["optional"].get("ttest", None)

    assert type(first_recoding_steps) == list or first_recoding_steps == ["never"], \
        f"Invalid recoding time steps specified, must be either empty list of ints or 'never', " \
        f"{str(first_recoding_steps)} found "
    assert type(second_recoding_steps) == list or second_recoding_steps == ["never"], \
        f"Invalid recoding time steps specified, must be either empty list of ints or 'never', " \
        f"{str(second_recoding_steps)} found "

    # Awkward spaghetti conversion of input args
    # Convert ["never"] to single string and otherwise the number in the list to str
    if len(first_recoding_steps) > 0:
        if first_recoding_steps[0] != "never":
            first_recoding_steps = list(map(int, first_recoding_steps))
        else:
            first_recoding_steps = first_recoding_steps[0]

    if len(second_recoding_steps) > 0:
        if second_recoding_steps[0] != "never":
            second_recoding_steps = list(map(int, second_recoding_steps))
        else:
            second_recoding_steps = second_recoding_steps[0]

    train_set, _ = load_data(corpus_dir, max_seq_len)
    vocab = defaultdict(lambda: "UNK", {idx: word for word, idx in train_set.vocab.items()})
    vocab[train_set.vocab.unk_idx] = "UNK"
    test_set = load_test_set(corpus_dir, max_seq_len, train_set.vocab, stop_after=stop_after)
    del train_set

    # Load the model whose weights will be used - if these models have a recoder, it will be removed
    first_models = (torch.load(path, map_location=device) for path in first_model_paths)
    second_models = (torch.load(path, map_location=device) for path in second_model_paths)

    # Collect data
    first_models_data = [
        extract_data(model, test_set, device, first_recoding_steps, collectables) for model in first_models
    ]
    second_models_data = [
        extract_data(model, test_set, device, second_recoding_steps, collectables) for model in second_models
    ]

    # Merge data
    scored_sentences = create_scored_sentences(first_models_data, second_models_data, test_set, vocab)

    # Plot
    pdf = PdfPages(pdf_path) if pdf_path is not None else None
    scored_sentences = scored_sentences if num_plots is None else random.sample(scored_sentences, num_plots)
    for scored_sentence in scored_sentences:
        plot_scores(scored_sentence, model_names, first_recoding_steps, second_recoding_steps, ttest, pdf)

    pdf.close()


def create_scored_sentences(first_models_data: List[List[dict]], second_models_data: List[List[dict]],
                            test_set: Corpus, vocab) -> List[ScoredSentence]:
    """
    Merge the measurements from several models and model runs into one single data structure.
    """
    first_sentence_data = zip(*first_models_data)
    second_sentence_data = zip(*second_models_data)
    scored_sentences = []

    for first_data, second_data, indexed_sentence in zip(first_sentence_data, second_sentence_data,
                                                         test_set.indexed_sentences):
        first_scores = {}
        for key in first_data[0].keys():
            # Concat data from multiple runs
            first_scores[key] = torch.stack([scores[key] for scores in first_data])

        second_scores = {}
        for key in second_data[0].keys():
            # Concat data from multiple runs
            second_scores[key] = torch.stack([scores[key] for scores in second_data])

        sentence = list(map(vocab.__getitem__, indexed_sentence.numpy()))
        scored_sentence = ScoredSentence(sentence, first_scores, second_scores)
        scored_sentences.append(scored_sentence)

    return scored_sentences


def extract_data(model: LSTMLanguageModel, test_set: Corpus, device: Device, recoding_steps: RecodingSteps,
                 collectables: List[str] = ("out", "out_prime", "delta", "delta_prime")) -> List[dict]:
    """
    Collect the perplexities of a model for all the words in a corpus.

    Parameters
    ----------
    model: LSTMLanguageModel
        Model for which the perplexities are being recorded.
    test_set: Corpus
        Test corpus the model is being tested on.
    device: Device
        The device the evaluation is being performed on.
    recoding_steps: RecodingSteps
        Define the time steps for which recoding should take place. If none are specified, recoding takes place at
        every time step.
    collectables: List[str]
        Define what data should be collected.
    """
    def perplexity(tensor: torch.Tensor) -> torch.Tensor:
        return - torch.log2(tensor)

    model.device = device
    all_sentence_data = []
    collector = StatsCollector()

    model.eval()
    model.diagnostics = True  # Return both the original and redecoded output distribution

    # Create dummy predictors
    dummy_predictors = {
        l: [DummyPredictor().to(device), DummyPredictor().to(device)]
        for l in range(model.num_layers)
    }

    # Don't recode when it is explicity specified to NEVER recode or if specific steps are specified which the current
    # step doesn't respond to
    dont_recode = lambda t: (recoding_steps == "never" or (t not in recoding_steps and len(recoding_steps) > 0))

    for sentence in test_set:
        hidden = None  # Reset hidden after every sentence to ensure comparability
        sentence_data = defaultdict(list)

        for t in range(sentence.shape[0] - 1):
            input_vars = sentence[t].unsqueeze(0).to(device)

            if isinstance(model, RecodingLanguageModel):
                # If no recoding is supposed to take place, switch step size predictors with ones always giving zeros
                if dont_recode(t):
                    model_predictors, model.mechanism.predictors = model.mechanism.predictors, dummy_predictors

                re_output_dist, output_dist, hidden = model(
                    input_vars, hidden, target_idx=sentence[t + 1].unsqueeze(0).to(device)
                )

                # Swap back
                if dont_recode(t):
                    model.mechanism.predictors = model_predictors

            else:
                output_dist, hidden = model(
                    input_vars, hidden, target_idx=sentence[t + 1].unsqueeze(0).to(device)
                )

            # Collect target word ppl
            if "out" in collectables:
                output_dist = F.softmax(output_dist)
                sentence_data["out"].append(perplexity(output_dist[:, sentence[t + 1]]).detach())

            if "out_prime" in collectables and isinstance(model, RecodingLanguageModel):
                re_output_dist = F.softmax(re_output_dist)
                sentence_data["out_prime"].append(perplexity(re_output_dist[:, sentence[t + 1]]).detach())

            # Perform another recoding step but 1.) set step size to 0 so we calculate the signal without actually
            # performing the update step and 2.) collect actual values later as they are intercepted by the stats
            # collector
            if "delta_prime" in collectables and isinstance(model, RecodingLanguageModel):
                # Swap out predictors so that the step size will be zero
                try:
                    model.mechanism.recoding_func(
                        input_vars, hidden, re_output_dist, device=device,
                        target_idx=sentence[t + 1].unsqueeze(0).to(device)
                    )
                except RuntimeError:
                    # Exploit an exception: Because the cell state hasn't been used for any computation, computing
                    # the gradient creates an exception. This can be used to halt the recoding after the new delta
                    # has already been collected
                    pass

        # Collect error signals - those were captured by StatsCollector
        if "delta" in collectables and isinstance(model, RecodingLanguageModel):
            # Because we applied two recoding steps in this case, deltas will be the first, third, fifth element
            # of the collected list and delta_primes the second, forth, sixth entry etc.
            if "delta_prime" in collectables:
                all_deltas = [
                    delta.unsqueeze(0) if len(delta.shape) == 0 else delta
                    for delta in StatsCollector._stats["deltas"]
                ]
                sentence_data["delta"] = all_deltas[::2]
                sentence_data["delta_prime"] = all_deltas[1::2]

            # In this case it's simply all the deltas the statscollector intercepted
            else:
                sentence_data["delta"] = [
                    delta.unsqueeze(0) if len(delta.shape) == 0 else delta
                    for delta in StatsCollector._stats["deltas"]
                ]

        # Concat and clean
        for key, data in sentence_data.items():
            data = torch.cat(data)
            data[data != data] = sys.maxsize
            sentence_data[key] = data

        all_sentence_data.append(sentence_data)
        collector.wipe()

    return all_sentence_data


def plot_scores(scored_sentence, model_names, first_recoding_steps: RecodingSteps, second_recoding_steps: RecodingSteps,
                ttest: Optional[str] = None, pdf: Optional[PdfPages] = None) -> None:
    """
    Plot the word perplexity scores of two models for the same sentence and potentially add it to a pdf.

    Parameters
    ----------
    scored_sentence: namedtuple
        Named tuple with data about how multiple runs of the same model behaved w.r.t. to a single sentence.
    model_names: list
        List of length two with the names of the models for the plot legend.
    first_recoding_steps: RecodingSteps
        Steps for which recoding takes places for the first model. These time steps will be marked accordingly. If
        None, recoding takes place at all time steps and no time step is marked for readability.
    second_recoding_steps: RecodingSteps
        Same as above but for the second model.
    ttest: Optional[str]
        Perform a t-test to determine whether the surprisal / error signals value differences between the two models
        are statistically significant.
    pdf: Optional[PdfPages]
        PDF the plot is written to if given. Otherwise the plot is shown on screen.
    """
    def zip_lists(list1, list2) -> List:
        new_list = []

        for el1, el2 in zip(list1, list2):
            new_list += [el1] + [el2]

        return new_list

    def zip_arrays(array1, array2) -> np.array:
        new_array = np.empty((array1.shape[0], 0))
        if len(array1.shape) == 3: array1 = array1.squeeze(2)
        if len(array2.shape) == 3: array2 = array2.squeeze(2)

        for i in range(array1.shape[1]):
            new_array = np.concatenate((new_array, array1[:, i][..., np.newaxis]), axis=1)
            new_array = np.concatenate((new_array, array2[:, i][..., np.newaxis]), axis=1)

        return new_array

    tokens = scored_sentence.sentence[:-1]  # Exclude <eos>
    tokens = zip_lists(tokens, [""] * len(tokens))
    x = range(len(tokens))
    fig, ax1 = plt.subplots()

    # Enable latex use
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=15)

    # Determine axes
    data_keys = set(scored_sentence.first_scores.keys()) | set(scored_sentence.second_scores.keys())
    data2ax = {"out": ax1, "delta": ax1}

    if "out" in data_keys and "delta" in data_keys:
        ax2 = ax1.twinx()
        # Surprisal scores will be on axis 1 and error signals on axis 2 UNLESS surprisal scores are not plotted,
        # then error signals are plotted on primary axis
        data2ax["delta"] = ax2
        ax1.set_ylabel(r"Surprisal $-\log_2(\mathbf{o}_t)$")

        ax2.set_ylabel(r"Error signal $\delta_t$")
    elif "delta" in data_keys:
        ax1.set_ylabel(r"Error signal $\delta_t$")
    else:
        ax1.set_ylabel(r"Surprisal $-\log_2(\mathbf{o}_t)$")

    def _produce_data(key, prime_key, model_scores) -> Tuple[np.array, np.array, np.array, np.array]:
        scores = model_scores[key].numpy()
        # If prime data is available, interlace with current data so that prime data is placed halfway between
        # two time steps
        if prime_key in model_scores:
            scores = zip_arrays(scores, model_scores[prime_key].numpy())
        else:
            scores = zip_arrays(scores, scores)

        mean, std = scores.mean(axis=0), scores.std(axis=0)
        high, low = mean + std, mean - std

        return mean, high, low, scores

    significant_steps = []
    for key in ["out", "delta"]:
        prime_key = key + "_prime"

        if key in scored_sentence.first_scores:
            first_mean, first_high, first_low, first_scores = _produce_data(
                key, prime_key, scored_sentence.first_scores
            )
            data2ax[key].plot(x, first_mean, label=f"{LATEX_LABELS[key]} {model_names[0]}", color=PLOT_COLORS[key][0])
            data2ax[key].fill_between(x, first_high, first_mean, alpha=0.4, color=PLOT_COLORS[key][0])
            data2ax[key].fill_between(x, first_mean, first_low, alpha=0.4, color=PLOT_COLORS[key][0])

        if key in scored_sentence.second_scores:
            second_mean, second_high, second_low, second_scores = _produce_data(
                key, prime_key, scored_sentence.second_scores
            )
            data2ax[key].plot(x, second_mean, label=f"{LATEX_LABELS[key]} {model_names[1]}", color=PLOT_COLORS[key][1])
            data2ax[key].fill_between(x, second_high, second_mean, alpha=0.4, color=PLOT_COLORS[key][1])
            data2ax[key].fill_between(x, second_mean, second_low, alpha=0.4, color=PLOT_COLORS[key][1])

        # Perform t-test
        if ttest == key and key in scored_sentence.first_scores and key in scored_sentence.second_scores:
            # Should t-test also be performed for recoded outputs / deltas? That only makes sense if at least one model
            # is a recoding model
            do_prime_steps = prime_key in data_keys

            # Perform t-test for every time step
            for i in range(0, len(tokens), 1 if do_prime_steps else 2):
                _, p_value = ttest_ind(first_scores[:, i], second_scores[:, i], equal_var=False)

                # If significant, add marker to time step
                if p_value <= 0.05:
                    significant_steps.append(i)

    plt.xticks(x, tokens, fontsize=15)

    # Draw vertical lines
    for x_ in range(0, len(tokens), 2):
        t = x_ / 2

        # Mark recoding for first model
        if first_recoding_steps != "never" and t in first_recoding_steps:
            plt.axvline(
                x=x_, color=PLOT_COLORS["delta"][0], linewidth=1, linestyle="dashed",
                label=f"Recoding {model_names[0]}"
            )

        # Mark recoding for second model
        elif second_recoding_steps != "never" and t in second_recoding_steps:
            plt.axvline(
                x=x_, color=PLOT_COLORS["delta"][1], linewidth=1, linestyle="dashed",
                label=f"Recoding {model_names[1]}"
            )

        # Mark non-recoding time stop OR if all time steps use recoding
        else:
            plt.axvline(x=x_, alpha=0.8, color="gray", linewidth=0.25)

    # Plot significant points
    if len(significant_steps) > 0:
        plt.scatter(
            significant_steps, np.zeros(len(significant_steps)) - 0.2, marker="^", color="black", alpha=0.6
        )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if "out" in data_keys and "delta" in data_keys:
        handles1, labels1 = ax1.get_legend_handles_labels()
        handels2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(
            handles1 + handels2, labels1 + labels2,
            loc='upper center', bbox_to_anchor=(0.5, -0.35), fancybox=True, ncol=2, prop={"size": 15}
        )
    else:
        plt.legend(
            loc='upper center', bbox_to_anchor=(0.5, -0.35), fancybox=True, ncol=2, prop={"size": 15}
        )

    plt.tight_layout()

    if pdf:
        pdf.savefig(fig)
    else:
        plt.show()

    plt.close()


def manage_config() -> dict:
    """
    Parse a config file (if given), overwrite with command line arguments and return everything as dictionary
    of different config groups.
    """
    required_args = {"corpus_dir", "max_seq_len", "first", "second", "device", "collectables",
                     "model_names"}
    arg_groups = {
        "general": required_args,
        "optional": {"stop_after", "pdf_path", "num_plots", "first_recoding_steps", "second_recoding_steps", "ttest"}
    }
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
    from_cmd.add_argument("--first", nargs="+", help="Path to model files.")
    from_cmd.add_argument("--second", nargs="+", help="Path to model files to compare first models to.")

    # Corpus options
    from_cmd.add_argument("--corpus_dir", type=str, help="Directory to corpus files.")
    from_cmd.add_argument("--max_seq_len", type=int, help="Maximum sentence length when reading in the corpora.")
    from_cmd.add_argument("--stop_after", type=int, help="Read corpus up to a certain number of lines.", default=None)

    # Evaluation options
    from_cmd.add_argument("--device", type=str, default="cpu", help="Device used for evaluation.")
    from_cmd.add_argument("--collectables", type=str, nargs="+", choices=["out", "out_prime", "delta", "delta_prime"],
                          help="Information that should be plotted.\nTarget word probability (out)\nTarget word "
                               "probability after recoding (out_prime)\nRecoding error signal (delta)\nError signal "
                               "after recoding (delta_prime)", default=["out", "out_prime", "delta", "delta_prime"])
    from_cmd.add_argument("--first_recoding_steps", nargs="+", default=[],
                          help="Explicitly define the time steps when recoding is supposed to take place for the first"
                               "model. If none are specified, recoding takes place at all time steps. If 'never' is "
                               "specified, it is used at no time step.")
    from_cmd.add_argument("--second_recoding_steps", nargs="+", default=[],
                          help="Explicitly define the time steps when recoding is supposed to take place for the second"
                               "model. If none are specified, recoding takes place at all time steps.")
    from_cmd.add_argument("--ttest", type=str, default=None, choices=["out", "delta"],
                          help="Perform a t-test to determine whether the surprisal / error signals value differences "
                               "between the two models are statistically significant.")

    # Plotting options
    from_cmd.add_argument("--pdf_path", type=str, help="Path to pdf with plots.")
    from_cmd.add_argument("--num_plots", type=int, help="Number of plots included in the pdf.", default=None)
    from_cmd.add_argument("--model_names", nargs=2, type=str, help="Name of models being compared")

    return parser


if __name__ == "__main__":
    main()

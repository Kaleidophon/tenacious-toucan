"""
Analyze how recoding error signals and test metric change in one or multiple models with or without recoding at one
or multiple time steps.
"""

# STD
from argparse import ArgumentParser
from collections import namedtuple, defaultdict
from typing import List, Any

# EXT
from diagnnose.config.setup import ConfigSetup
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from matplotlib.backends.backend_pdf import PdfPages

# PROJECT
from src.models.language_model import LSTMLanguageModel
from src.recoding.step import AbstractStepPredictor
from src.utils.corpora import load_data, Corpus
from src.utils.log import StatsCollector
from src.utils.test import load_test_set
from src.utils.types import Device, StepSize
from src.utils.compatability import RNNCompatabilityMixin as CompatibleRNN

# CONSTANTS
BASELINE_COLOR = "tab:blue"
RECODING_COLOR = "tab:orange"

# TYPES
ScoredSentence = namedtuple(
    "ScoredSentence", ["sentence", "first_scores", "first_std", "second_scores", "second_std", "diff"]
)


class DummyPredictor(AbstractStepPredictor):
    """
    Dummy predictor which always predicts a step size of zero. Useful when you want to avoid recoding to take place.
    """

    def forward(self, hidden: Tensor, out: Tensor, device: torch.device, **additional: Any) -> StepSize:
        """
        Prediction step.

        Parameters
        ----------
        hidden: Tensor
            Current hidden state used to determine step size.
        out: Tensor
            Output Tensor of current time step.
        device: torch.device
            Torch device the model is being trained on (e.g. "cpu" or "cuda").

        Returns
        -------
        step_size: StepSize
            Batch size x 1 tensor of predicted step sizes per batch instance or one single float for the whole batch.
        """
        return torch.Tensor([0]).to(device)


def main() -> None:
    config_dict = manage_config()

    # TODO: Load models

    # TODO: Collect data
    # TODO: Also enable optional argument of where recoding should take place

    # TODO: Plot


def extract_data(model: LSTMLanguageModel, test_set: Corpus, device: Device,
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
    collectables: List[str]
        Define what data should be collected.
    """
    # TODO: Add functionality to only recode at certain time steps?
    # TODO: Make baseline-proof
    def perplexity(tensor: torch.Tensor) -> torch.Tensor:
        return tensor ** -tensor  # 2 ^ (-x log2 x) = 2 ^ (log2 x ^ -x) = x ^ -x

    hidden = None
    all_sentence_data = []
    collector = StatsCollector()

    model.eval()
    model.diagnostics = True  # Return both the original and redecoded output distribution

    # Create dummy predictors
    dummy_predictors = {
        l: [DummyPredictor().to(device), DummyPredictor().to(device)]
        for l in range(model.num_layers)
    }

    for sentence in test_set:
        sentence_data = defaultdict(list)

        for t in range(sentence.shape[0] - 1):
            input_vars = sentence[t].unsqueeze(0).to(device)
            re_output_dist, output_dist, hidden = model(
                input_vars, hidden, target_idx=sentence[t + 1].unsqueeze(0).to(device)
            )

            # Collect target word ppl
            if "out" in collectables:
                sentence_data["out"].append(perplexity(output_dist[:, sentence[t + 1]]))

            if "out_prime" in collectables:
                sentence_data["out_prime"].append(perplexity(re_output_dist[:, sentence[t + 1]]))

            # Perform another recoding step but 1.) set step size to 0 so we calculate the signal without actually
            # performing the update step and 2.) collect actual values later as they are intercepted by the stats
            # collector
            if "delta_prime" in collectables:
                # Swap out predictors so that the step size will be zero
                model_predictors, model.mechanism.predictors = model.mechanism.predictors, dummy_predictors

                model.mechanism.recoding_func(
                    input_vars, hidden, re_output_dist, device=device,  # we are now using recoded hiddens here
                    target_idx=sentence[t + 1].unsqueeze(0).to(device)
                )

                # Swap back
                model.mechanism.predictors = model_predictors

        # Collect error signals - those were captured by StatsCollector
        if "delta" in collectables:
            # Because we applied two recoding steps in this case, deltas will be the first, third, fifth element
            # of the collected list and delta_primes the second, forth, sixth entry etc.
            if "delta_prime" in collectables:
                all_deltas = StatsCollector._stats["deltas"]
                sentence_data["delta"] = all_deltas[::2]
                sentence_data["delta_prime"] = all_deltas[1::2]

            # In this case it's simply all the deltas the statscollector intercepted
            else:
                sentence_data["delta"] = StatsCollector._stats["deltas"]

        # Concat and so that we have tensors containing all the target word perplexities of all batch sentences
        all_sentence_data.append(sentence_data)
        collector.wipe()

        hidden = {l: CompatibleRNN.map(h, func=lambda h: h.detach()) for l, h in hidden.items()}

    return all_sentence_data


def plot_perplexities(scored_sentence,  model_names, pdf=None) -> None:
    """
    Plot the word perplexity scores of two models for the same sentence and potentially add it to a pdf.

    Parameters
    ----------
    scored_sentence: namedtuple
        Named tuple with data about how multiple runs of the same model behaved w.r.t. to a single sentence.
    model_names: list
        List of length two with the names of the models for the plot legend.
    pdf: Optional[PdfPages]
        PDF the plot is written to if given. Otherwise the plot is shown on screen.
    """
    # TODO: Rewrite
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
    else:
        plt.show()

    plt.close()


def manage_config() -> dict:
    """
    Parse a config file (if given), overwrite with command line arguments and return everything as dictionary
    of different config groups.
    """
    # TODO: Rewrite
    required_args = {"corpus_dir", "max_seq_len", "batch_size", "models", "device", "pdf_path", "num_plots"}
    arg_groups = {"general": required_args, "optional": {"stop_after"}}
    argparser = init_argparser()
    config_object = ConfigSetup(argparser, required_args, arg_groups)
    config_dict = config_object.config_dict

    return config_dict


def init_argparser() -> ArgumentParser:
    # TODO: Rewrite
    parser = ArgumentParser()

    from_config = parser.add_argument_group('From config file', 'Provide full experiment setup via config file')
    from_config.add_argument('-c', '--config', help='Path to json file containing classification config.')

    from_cmd = parser.add_argument_group('From commandline', 'Specify experiment setup via commandline arguments')

    # Model options
    from_cmd.add_argument("--models", nargs="+", help="Path to model files.")

    # Corpus options
    from_cmd.add_argument("--corpus_dir", type=str, help="Directory to corpus files.")
    from_cmd.add_argument("--max_seq_len", type=int, help="Maximum sentence length when reading in the corpora.")
    from_cmd.add_argument("--stop_after", type=int, help="Read corpus up to a certain number of lines.")

    # Evaluation options
    from_cmd.add_argument("--device", type=str, default="cpu", help="Device used for evaluation.")

    # Plotting options
    from_cmd.add_argument("--pdf_path", type=str, help="Path to pdf with plots.")
    from_cmd.add_argument("--num_plots", type=int, help="Number of plots included in the pdf.")

    return parser


if __name__ == "__main__":
    main()

"""
Run an LSTM language models with interventions.
"""

# STD
from argparse import ArgumentParser
from collections import defaultdict

# EXT
import torch
from torch.nn.functional import log_softmax
from lm_diagnoser.classifiers.dc_trainer import DCTrainer
from lm_diagnoser.activations.initial import InitStates
from lm_diagnoser.interventions.language_model import (
    LanguageModelInterventionMechanism, SubjectLanguageModelInterventionMechanism
)
from lm_diagnoser.models.intervention_lstm import InterventionLSTM
from lm_diagnoser.models.import_model import import_model_from_json
from lm_diagnoser.corpora.import_corpus import convert_to_labeled_corpus
from lm_diagnoser.typedefs.corpus import LabeledCorpus
from lm_diagnoser.config.setup import ConfigSetup
from scipy.stats import shapiro, ttest_ind

# PROJECT
from corpus import read_gulordava_corpus


def main():
    # Manage config
    required_args = {'model', 'vocab', 'lm_module', 'corpus_path', 'classifiers'}
    arg_groups = {
        'model': {'model', 'vocab', 'lm_module', 'device'},
        'corpus': {'corpus_path'},
        'interventions': {'step_size', 'classifiers', 'init_states'},
    }
    argparser = init_argparser()
    config_object = ConfigSetup(argparser, required_args, arg_groups)
    config_dict = config_object.config_dict

    # Load data: Corpus, models, diagnostic classifiers
    corpus_path = config_dict["corpus"]["corpus_path"]

    if corpus_path.endswith(".pickle"):
        corpus = convert_to_labeled_corpus(corpus_path)
    else:
        corpus = read_gulordava_corpus(config_dict["corpus"]["corpus_path"])

    basic_model = import_model_from_json(
        **config_dict["model"], model_class=InterventionLSTM
    )
    subj_intervention_model = import_model_from_json(
        **config_dict["model"], model_class=InterventionLSTM
    )
    global_intervention_model = import_model_from_json(
        **config_dict["model"], model_class=InterventionLSTM
    )

    # Load classifiers and apply intervention mechanisms
    step_size = config_dict["interventions"]["step_size"]
    classifier_paths = config_dict["interventions"]["classifiers"]
    init_states_path = config_dict["interventions"]["init_states"]
    classifiers = {path: DCTrainer.load_classifier(path) for path in classifier_paths}
    subj_mechanism = SubjectLanguageModelInterventionMechanism(subj_intervention_model, classifiers, step_size)
    global_mechanism = LanguageModelInterventionMechanism(global_intervention_model, classifiers, step_size)
    subj_intervention_model = subj_mechanism.apply()
    global_intervention_model = global_mechanism.apply()
    init_states = InitStates(basic_model, init_states_path)

    # 1. Experiment: Replicate Gulordava findings
    # In what percentage of cases does the LM assign a higher probability to the grammatically correct sentence?
    #replicate_gulordava(basic_model, corpus, init_states=init_states)

    # 2. Experiment: Assess the influence of interventions on LM perplexity
    measure_influence_on_perplexity(basic_model, subj_intervention_model, global_intervention_model, corpus, init_states)

    # 3. Experiment: Check to what extend the accuracy of Diagnostic Classifiers increases after having interventions
    # on the subject position / on every position
    # TODO

    # 4. Experiment: Repeat the 1. Experiment but measure the influence of interventions on the subject position /
    # on every position
    # TODO


def replicate_gulordava(model: InterventionLSTM,
                        corpus: LabeledCorpus,
                        init_states: InitStates) -> None:
    """
    Replicate the Language Model number prediction accuracy experiment from [1]. In this experiment, a language models
    is facing a sentence in which the main verb is presented in its singular and plural form, one of which is
    ungrammatical given the numerosity of the sentence's subject. The LM is then expected to assign a higher probability
    to the grammatical sentence. Finally, the percentage of cases in which this was the case is reported.

    [1] https://arxiv.org/pdf/1803.11138.pdf
    """
    print("\n\nReplicating Gulordava Number Agreement experiment...")

    # Calculate scores
    scores = {"original": [], "generated": []}

    for i, (sentence_id, labelled_sentence) in enumerate(corpus.items()):

        if i % 5 == 0:
            print(f"\rProcessing sentence #{i+1}...", end="", flush=True)

        # Get necessary information
        sentence = labelled_sentence.sen
        target_pos = labelled_sentence.misc_info["verb_pos"]
        right_form = labelled_sentence.misc_info["right_form"]
        wrong_form = labelled_sentence.misc_info["wrong_form"]
        sentence_type = labelled_sentence.misc_info["type"]

        # Retrieve word indices
        right_index = model.w2i[right_form]
        wrong_index = model.w2i[wrong_form]

        # Feed sentence into RNN
        activations = init_states.states

        for pos, token in enumerate(sentence):

            out, activations = model.forward(token, activations)

            # After processing the sentence up to the verb in question, check which of the verb forms is assigned
            # a higher probability
            if pos == target_pos - 1:
                if out[right_index] > out[wrong_index]:
                    scores[sentence_type].append(1)
                else:
                    scores[sentence_type].append(0)

    original_acc = sum(scores["original"]) / len(scores["original"])
    nonce_acc = sum(scores["generated"]) / len(scores["generated"])

    print("")
    print(f"Original accuracy: {round(original_acc * 100, 1):.1f}")
    print(f"Nonce accuracy: {round(nonce_acc * 100, 1):.1f}")


def measure_influence_on_perplexity(basic_model: InterventionLSTM,
                                    subj_intervention_model: InterventionLSTM,
                                    global_intervention_model: InterventionLSTM,
                                    corpus: LabeledCorpus,
                                    init_states: InitStates) -> None:
    """
    Check whether interventions - be it only on the subject position or all positions - influence the perplexity
    of the Language Model in a statistically significant way.
    """
    perplexities = defaultdict(list)
    w2i = basic_model.w2i  # Vocabulary is shared between models
    unk_index = basic_model.unk_idx
    basic_activations, subj_activations, global_activations = init_states.states, init_states.states, init_states.states

    print("\n\nAssessing influence of interventions on perplexities...")
    print("Gathering perplexity scores for corpus...")

    for sentence_id, sentence in corpus.items():

        if sentence_id % 5 == 0:
            print(f"\rProcessing sentence #{sentence_id+1}...", end="", flush=True)

        basic_perplexity, subj_perplexity, global_perplexity = 0, 0, 0

        # Get necessary sentence data
        sen = sentence.sen
        labels = sentence.labels
        subj_pos = sentence.misc_info["subj_pos"]

        for pos, token in enumerate(sen):
            basic_out, basic_activations = basic_model(token, basic_activations)
            subj_out, subj_activations = subj_intervention_model(
                token, subj_activations, label=labels[pos], is_subj_pos=subj_pos == pos
            )
            global_out, global_activations = global_intervention_model(token, global_activations, label=labels[pos])

            # "Batchify" to speed up expensive log-softmax
            token_index = unk_index if token not in w2i else w2i[token]
            outs = torch.stack((basic_out, subj_out, global_out))
            vocab_probs = log_softmax(outs, dim=0)
            token_probs = vocab_probs[:, token_index]
            basic_prob, subj_prob, global_prob = torch.split(token_probs, 1, dim=0)
            basic_perplexity += basic_prob
            subj_perplexity += subj_prob
            global_perplexity += global_prob

        # Save sentence perplexities
        perplexities["basic"].append(basic_perplexity.detach().numpy()[0])
        perplexities["subj"].append(subj_perplexity.detach().numpy()[0])
        perplexities["global"].append(global_perplexity.detach().numpy()[0])

    print("Test whether perplexities are normally distributed...")
    _, p_basic = shapiro(perplexities["basic"])
    _, p_subj = shapiro(perplexities["subj"])
    _, p_global = shapiro(perplexities["global"])
    print(f"Basic: {p_basic:.2f} | Subj: {p_subj:.2f} | Global: {p_global:.2f}\n")

    print("Test whether the difference in perplexity is stat. significant after interventions...")
    _, p_basic_subj = ttest_ind(perplexities["basic"], perplexities["subj"], equal_var=False)
    _, p_basic_global = ttest_ind(perplexities["basic"], perplexities["subj"], equal_var=False)
    print(f"Basic - Subj: {p_basic_subj:.2f} | Basic - Global: {p_basic_global:.2f}\n\n")


def init_argparser() -> ArgumentParser:
    parser = ArgumentParser()

    from_config = parser.add_argument_group('From config file',
                                            'Provide full experiment setup via config file')
    from_config.add_argument('-c', '--config',
                             help='Path to json file containing classification config.')

    from_cmd = parser.add_argument_group('From commandline',
                                         'Specify experiment setup via commandline arguments')
    from_cmd.add_argument('--models', type=str, help='Location of json file with models setup')
    from_cmd.add_argument('--corpus', type=str, help='Location of corpus')
    from_cmd.add_argument('--classifiers', nargs="+", help='Location of diagnostic classifiers')
    from_cmd.add_argument('--step_size', type=float, help="Step-size for weakly supervised interventions.")
    from_cmd.add_argument('--init_states', type=float, help="Path to states to initialize the Language Model with.")

    return parser


if __name__ == "__main__":
    main()

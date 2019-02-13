"""
Extract activations from a model, almost completely and shamelessly copied from
https://github.com/dieuwkehupkes/diagnosing_lms/tree/interventions.
"""

# STD
from argparse import ArgumentParser

# EXT
from rnnalyse.config.setup import ConfigSetup
from rnnalyse.extractors.base_extractor import Extractor
from rnnalyse.models.language_model import LanguageModel
from rnnalyse.typedefs.corpus import LabeledCorpus, LabeledSentence
from rnnalyse.models.import_model import import_model_from_json
from rnnalyse.corpora.import_corpus import convert_to_labeled_corpus


def init_argparser() -> ArgumentParser:
    parser = ArgumentParser()

    # create group to load config from a file
    from_config = parser.add_argument_group('From config file',
                                            'Provide full experiment setup via config file')
    from_config.add_argument('-c', '--config',
                             help='Path to json file containing extraction config.')

    # create group to provide info via commandline arguments
    # Required args are not set to be required here as they can come from --config
    from_cmd = parser.add_argument_group('From commandline',
                                         'Specify experiment setup via commandline arguments')
    from_cmd.add_argument('--model',
                          help='Path to model parameters')
    from_cmd.add_argument('--vocab',
                          help='Path to model vocabulary')
    from_cmd.add_argument('--lm_module',
                          help='Path to folder containing model module')
    from_cmd.add_argument('--corpus_path',
                          help='Path to labeled corpus')
    # TODO: Provide explanation of activation names
    from_cmd.add_argument('--activation_names',
                          help='Activations to be extracted', nargs='*')
    from_cmd.add_argument('--output_dir',
                          help='Path to folder to which extracted embeddings will be written.')
    from_cmd.add_argument('--device',
                          help='(optional) Torch device name on which model will be run.'
                               'Defaults to cpu.')
    from_cmd.add_argument('--init_lstm_states_path',
                          help='(optional) Location of initial lstm states of the model. '
                               'If no path is provided zero-initialized states will be used at the'
                               'start of each sequence.')
    from_cmd.add_argument('--print_every', type=int,
                          help='(optional) Print extraction progress every n steps.'
                               'Defaults to 20.')
    from_cmd.add_argument('--cutoff', type=int,
                          help='(optional) Stop extraction after n sentences. '
                               'Defaults to -1 to extract entire corpus.')

    return parser


def subj_selection_func(pos: int, token: str, sentence: LabeledSentence):
    """ Select activations only when they occur on the subject's position. """
    return pos == sentence.misc_info["subj_pos"]


def pos_4_selection_func(pos: int, token: str, sentence: LabeledSentence):
    """ Select activations only on position 4. """
    return pos == 4


if __name__ == "__main__":
    required_args = {'model', 'vocab', 'lm_module', 'corpus_path', 'activation_names', 'output_dir'}
    arg_groups = {
        'model': {'model', 'vocab', 'lm_module', 'device'},
        'corpus': {'corpus_path'},
        'init_extract': {'activation_names', 'output_dir', 'init_lstm_states_path'},
        'extract': {'cutoff', 'print_every'},
    }
    argparser = init_argparser()

    config_object = ConfigSetup(argparser, required_args, arg_groups)
    config_dict = config_object.config_dict

    model: LanguageModel = import_model_from_json(**config_dict['model'])
    corpus: LabeledCorpus = convert_to_labeled_corpus(**config_dict['corpus'])

    extractor = Extractor(model, corpus, **config_dict['init_extract'])
    extractor.extract(**config_dict['extract'], selection_func=pos_4_selection_func)
    extractor.extract_average_eos_activations(print_every=config_dict['extract']['print_every'])

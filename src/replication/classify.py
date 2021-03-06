"""
Train Diagnostic Classifier on the extracted activations of a model, almost completely and shamelessly copied from
https://github.com/dieuwkehupkes/diagnosing_lms/tree/interventions.
"""

# STD
from argparse import ArgumentParser

# EXT
from diagnnose.classifiers.dc_trainer import DCTrainer
from diagnnose.config.setup import ConfigSetup


def init_argparser() -> ArgumentParser:
    parser = ArgumentParser()

    # create group to load config from a file
    from_config = parser.add_argument_group('From config file',
                                            'Provide full experiment setup via config file')
    from_config.add_argument('-c', '--config',
                             help='Path to json file containing classification config.')

    # create group to provide info via commandline arguments
    # Required args are not set to be required here as they can come from --config
    from_cmd = parser.add_argument_group('From commandline',
                                         'Specify experiment setup via commandline arguments')
    from_cmd.add_argument('--activations_dir',
                          help='Path to folder containing activations to train on.')
    from_cmd.add_argument('--activation_names',
                          help='Activation names to train on.', nargs='*')
    from_cmd.add_argument('--output_dir',
                          help='Path to folder to  which trained classifiers and '
                               'results will be written.')
    from_cmd.add_argument('--classifier_type',
                          help='Classifier type, as of now \'logreg\' or \'svm\'')
    from_cmd.add_argument('--labels',
                          help='(optional) Path to pickle file containing activation labels.'
                               'Defaults to lables.pickle which is generated by the Labeler.')
    from_cmd.add_argument('--train_subset_size', type=int,
                          help='(optional) Size of training set to train on.'
                               'Defaults to -1, i.e. the full training set.')
    from_cmd.add_argument('--train_test_split', type=float,
                          help='(optional) Percentage of data set split into train/test set.'
                               'Defaults to 0.9, indicating a 90/10 train/test split.')

    return parser


if __name__ == '__main__':
    required_args = {'activations_dir', 'activation_names', 'output_dir', 'classifier_type'}
    arg_groups = {
        'dc_trainer': {'activations_dir', 'activation_names', 'output_dir',
                       'classifier_type', 'labels'},
        'classify': {'train_subset_size', 'train_test_split'},
    }
    argparser = init_argparser()

    config_object = ConfigSetup(argparser, required_args, arg_groups)
    config_dict = config_object.config_dict

    dc_trainer = DCTrainer(**config_dict['dc_trainer'])
    dc_trainer.train(**config_dict['classify'])

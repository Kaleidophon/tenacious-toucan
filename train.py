"""
Train the model with the uncertainty-based intervention mechanism.
"""

# PROJECT
from src.utils.corpora import load_data
from src.utils.trainer import train_model, init_model
from src.utils.trainer import manage_config


def main():
    config_dict = manage_config()

    # Load data
    corpus_dir = config_dict["corpus"]["corpus_dir"]
    max_seq_len = config_dict["corpus"]["max_seq_len"]
    train_set, valid_set = load_data(corpus_dir, max_seq_len)

    # Initialize model
    model = init_model(config_dict, vocab_size=len(train_set.vocab), corpus_size=len(train_set))

    # Train
    log_dir = config_dict["logging"]["log_dir"]
    train_model(model, train_set, **config_dict['train'], valid_set=valid_set, log_dir=log_dir)

if __name__ == "__main__":
    main()

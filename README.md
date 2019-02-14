## tenacious-toucan (Working title)

Master thesis in Artificial Intelligence written during Summer term 2019 at the Universiteit van Amsterdam.

### Description

TODO

### Usage

To run any code in this repository, first clone the repository and install the requirements:

    git clone https://github.com/Kaleidophon/tenacious-toucan.git
    pip3 install -r requirements.txt

#### Replication of Gulianelli et al. (2018)

To replicate the finding of Gulianelli et al. (2018), simply run the following the following script:

    sh replicate_giulianelli.py
    
This script performs the following steps:

1. Converting the corpora into the right format. 
2. Downloading the trained language model by Gulordava et al. (2018).
3. Extracting the activations from a corpus using the very same model.
4. Training classifiers on these activations that try to predict the numerosity of the verb based on hidden activations.
5. Replicating the experiments described in the paper (number agreement accuracy with and without interventions and 
whether these influence the perplexity values in a statistically significant way)

Any settings for these steps can be changed by either changing the corresponding config files inside the `config` directory
or running the corresponding scripts individually and supplying arguments via the command line (command line arguments override
config options).

This script might take while to execute, to better run it in the background and get yourself a cup of coffee in the 
meantime :coffee:

Lastly, it should be noted that the training of the Diagnostic Classifiers introduces some randomness into the 
replication, because the training splits are generated randomly everytime, and differently trained Diagnostic Classifiers
therefore also impact the effects of interventions in a different way. To repeat the experiments with different classifiers
again, delete the results in `data/classifiers/giulianelli/models` and `data/classifiers/giulianelli/preds` and execute 
`classify.py` and `replication.py` again.


### References

Giulianelli, Mario, et al. "Under the Hood: Using Diagnostic Classifiers to Investigate and Improve how Language Models Track Agreement Information." arXiv preprint arXiv:1808.08079 (2018).

Gulordava, Kristina, et al. "Colorless green recurrent networks dream hierarchically." arXiv preprint arXiv:1803.11138 (2018).
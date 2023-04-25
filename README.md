# NLP Robustness Experiment for Random feature noise injection.

This is the github repository for testing model's robustness and gather benchmark results from different methods.

## Creating the environment.
To create the environment for the experiment, run this command.

`conda env create -f environment.yml`
## Running the experiment.
For each of the dataset, there is a TextAttack script for executing the attacks. In order to run the method, simply run the TextAttack script for that dataset

`python textattack_agnews.py --parallel`

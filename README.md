# NLP Robustness Experiment for Random feature noise injection.

This is the github repository for testing model's robustness and gather benchmark results from different methods.

## Creating the environment.
To create the environment for the experiment, run this command.

`conda env create -f environment.yml`
## Running the experiment.
# Setting up the weight structure.
If one has access to wget and the google drive server directly, they can download the weight directly from this link:
https://drive.google.com/file/d/1swjLh5UuCKXMGq9xaaB0oiUgho3UfC5F/view?usp=share_link

After downloading the weights, place the model.zip file in the github repo's directory and unzip the file.

`unzip model.zip`
# Downloading data set.
For AGNEWS, we used the dataset that comes from huggingface. For IMDB, please:
 - download the dataset in this website:
   - https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
 - create a "data" folder and put the downloaded file in that path.
 - untar the files using this command:
   - `tar -zxvf aclImdb_v1.tar.gz`
# Running the experiments.
(This only covers the experiments related to mask and safer defense method.) To the the experiments, there are two ways to do it:
 1. Run the run_experiment_baidu.sh.
   - `chmod +x run_experiment_baidu.sh | ./run_experiment_baidu.sh`

 2. Run using python commands:
   - `python for_baidu/baidu_agnews.py --model ["bert"|"roberta] --defense ["mask"|"safer"] --parallel (to run multiple process) -nd (number of process/GPUs, default:2)`
   - you can adjust the number of workers for each GPU by adding `-nd` for the commands in the bash script.
   - you can also limit the GPUs by passing CUDA_VISIBLE_DEVICES=(GPU IDs) before the commands.

# Package the results
Once the experiment is finish, we can package the results by zipping the `noise_defense_attack_result/paper_default setting` folder.

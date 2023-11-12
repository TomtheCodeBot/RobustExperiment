from textattack.datasets import HuggingFaceDataset
import os
os.environ['CURL_CA_BUNDLE'] = ''
from typing import List, Dict
import random
from textattack.attack_recipes import (
    TextFoolerJin2019,
    BERTAttackLi2020,
    HotFlipEbrahimi2017,
    DeepWordBugGao2018,
    TextBuggerLi2018,
)
from sklearn.metrics import accuracy_score

from textattack.transformations import WordSwapEmbedding
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders.universal_sentence_encoder import (
    UniversalSentenceEncoder,
)
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.goal_functions import UntargetedClassification
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from textattack.transformations import WordSwapEmbedding, WordSwapWordNet, WordSwapMaskedLM
from utils.dataloader import load_train_test_imdb_data

from model.BERTNoiseDefend import BertForSequenceClassification

from model.RoBERTaNoiseDefend import RobertaForSequenceClassification

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from utils.preprocessing import clean_text_imdb
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from model.robustNB import RobustNaiveBayesClassifierPercentage
from utils.bert_vectorizer import BertVectorizer
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack import Attack
from textattack.models.wrappers import (
    ModelWrapper,
    PyTorchModelWrapper,
    SklearnModelWrapper,
    HuggingFaceModelWrapper,
)
from tqdm import tqdm
from textattack.models.helpers import LSTMForClassification
import torch
import argparse
from textattack import Attacker, AttackArgs
from textattack.datasets import Dataset
import datasets
import numpy as np
import os
import model as model_lib
from model.TextDefenseExtraWrapper import wrapping_model
import json
class CustomModelWrapper(PyTorchModelWrapper):
    def __init__(self, model, tokenizer):
        super(CustomModelWrapper, self).__init__(model, tokenizer)

    def __call__(self, text_input_list):
        inputs_dict = self.tokenizer(
            text_input_list,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        inputs_dict.to(model_device)

        with torch.no_grad():
            outputs = self.model(**inputs_dict)

        if isinstance(outputs, tuple):
            return outputs[-1]  # model-h,model-bh

        if isinstance(outputs, torch.Tensor):
            return outputs  # baseline

        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence models return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            return outputs
        else:
            # HuggingFace classification models return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.
            return outputs.logits

def build_attacker_from_textdefender(model: HuggingFaceModelWrapper,args) -> Attack:
    if args["attack_method"] == 'hotflip':
        return HotFlipEbrahimi2017.build(model)
    if args["attack_method"] == 'textfooler':
        attacker = TextFoolerJin2019.build(model)
    elif args["attack_method"] == 'bertattack':
        attacker = BERTAttackLi2020.build(model)
        attacker.transformation = WordSwapMaskedLM(method="bert-attack", max_candidates=args["k_neighbor"])
    elif args["attack_method"] == 'textbugger':
        attacker = TextBuggerLi2018.build(model)
    else:
        attacker = TextFoolerJin2019.build(model)

    if args["attack_method"] in ['textfooler', 'pwws', 'textbugger', 'pso']:
        attacker.transformation = WordSwapEmbedding(max_candidates=args["k_neighbor"])
        for constraint in attacker.constraints:
            if isinstance(constraint, WordEmbeddingDistance) or isinstance(constraint, UniversalSentenceEncoder):
                attacker.constraints.remove(constraint)
                
    attacker.constraints.append(MaxWordsPerturbed(max_percent=args["modify_ratio"]))
    use_constraint = UniversalSentenceEncoder(
        threshold=args["similarity"],
        metric="cosine"
    )
    attacker.constraints.append(use_constraint)
    print(attacker.constraints)
    attacker.goal_function = UntargetedClassification(model, query_budget=args["k_neighbor"])
    return Attack(attacker.goal_function, attacker.constraints + attacker.pre_transformation_constraints,
                  attacker.transformation, attacker.search_method)
    

    return attacker
def build_attacker(model, args):
    if args["attack_method"] == "textfooler":
        attacker = TextFoolerJin2019.build(model)
    elif args["attack_method"] == "textbugger":
        attacker = TextBuggerLi2018.build(model)
    elif args["attack_method"] == "deepwordbug":
        attacker = DeepWordBugGao2018.build(model)
    elif args["attack_method"] == "bertattack":
        attacker = BERTAttackLi2020.build(model)
    else:
        attacker = TextFoolerJin2019.build(model)
    if args["modify_ratio"] != 0:
        print(args["modify_ratio"])
        attacker.constraints.append(MaxWordsPerturbed(max_percent=args["modify_ratio"]))
    if args["similarity"] != 0:
        print("AGHHHHHHHHHHHHHH")
        attacker.constraints.append(
            UniversalSentenceEncoder(threshold=args["similarity"], metric="cosine")
        )
    return attacker


def str2bool(strIn):
    if strIn.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif strIn.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


def set_seed(seed=42):
    if seed is not None:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True


def attack(args, wrapper, name, dataset):
    attackMethod = args.attack_method
    attack_args_dict = {
        "attack_method": attackMethod,
        "attack_times": 1,
        "attack_examples": int(args.attack_examples),
        "modify_ratio": float(args.modify_ratio),
        "similarity": float(args.similarity),
        "k_neighbor": int(args.k_neighbor),
        "log_path": "{}/{}/{}_{}-{}.txt".format(
            args.load_path, attackMethod, args.attack_dataset, args.modify_ratio, name
        ),
    }
    attack = build_attacker_from_textdefender(wrapper, attack_args_dict)
    attack_args = AttackArgs(
        num_examples=attack_args_dict["attack_examples"],
        log_to_txt=attack_args_dict["log_path"],
        csv_coloring_style="file",
    )
    attacker = Attacker(attack, dataset, attack_args)
    attacker.attack_dataset()


def gen_dataset(instances):
    test_instances = instances
    test_dataset = []
    for instance in range(len(test_instances)):
        test_dataset.append(
            (test_instances["text"][instance], int(test_instances["label"][instance]))
        )
    dataset = Dataset(test_dataset, shuffle=True)
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parameter")
    parser.add_argument("-a", "--attack", type=str2bool, nargs="?", const=False)
    parser.add_argument("-lp", "--load_path", default=None)
    parser.add_argument("-am", "--attack_method", default=None)
    parser.add_argument(
        "-ad", "--attack_dataset", default="test"
    )  # attack dataset & accuracy dataset
    parser.add_argument("-ae", "--attack_examples", default=1000)
    parser.add_argument("-mr", "--modify_ratio", default=0.1)
    parser.add_argument("-sm", "--similarity", default=0.84)
    parser.add_argument("-kn", "--k_neighbor", default=50)
    parser.add_argument("-en", "--ensemble_num", default=16)
    parser.add_argument("-eb", "--ensemble_batch_size", default=32)
    parser.add_argument("-rms", "--random_mask_rate", default=0.3)
    parser.add_argument("-spf", "--safer_pertubation_file", default="/home/ubuntu/TextDefender/dataset/imdb/perturbation_constraint_pca0.8_100.pkl")
    parser.add_argument("-md", "--model", default="bert")
    parser.add_argument("-df", "--defense", default="mask")
    args = parser.parse_args()
    
    clean_accuracy={}
    num_repetitions = 3
    batch=100
    
    yelp_dataset = datasets.load_dataset("yelp_polarity")
    train_data = yelp_dataset["train"]
    test_data = yelp_dataset["test"]    
    test_labels = np.array(test_data["label"])
    device = "cuda"
    bert_input = list(test_data["text"])
    batch_iter = len(bert_input)//batch + (1 if len(bert_input)%batch!=0 else 0 )
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",use_fast=True)
    tokenizer_roberta = AutoTokenizer.from_pretrained("roberta-base",use_fast=True)
    tokenizer.model_max_length=256
    
    tokenizer = AutoTokenizer.from_pretrained("/vinserver_user/duy.hc/tmd/yelp_polarity/bert-base-uncased/epoch_9",use_fast=True)
    config = AutoConfig.from_pretrained("/vinserver_user/duy.hc/tmd/yelp_polarity/bert-base-uncased/epoch_9")
    model = BertForSequenceClassification(config)
    state = AutoModelForSequenceClassification.from_pretrained("/vinserver_user/duy.hc/tmd/yelp_polarity/bert-base-uncased/epoch_9")
    model.load_state_dict(state.state_dict())
    model.eval()
    BERT = HuggingFaceModelWrapper(model,tokenizer)
    BERT.to("cuda")
    y_pred_BERT = []
    for i in tqdm(range(0,len(bert_input)//batch)):
        y_pred_BERT.extend(torch.argmax(torch.tensor(BERT(bert_input[i*batch:(i+1)*batch])),dim=-1).tolist())
    acc = accuracy_score(test_labels, y_pred_BERT)
    print(f"IMDB BERT: {acc*100:.2f}%")
    clean_accuracy["IMDB_BERT"] = f"{acc*100:.2f}%"
    
    y_pred_BERT = []
    for i in tqdm(range(0,len(bert_input)//batch)):
        y_pred_BERT.extend(torch.argmax(BERT(bert_input[i*batch:(i+1)*batch]),dim=-1).tolist())
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_BERT)
    print(f"IMDB BERT (with noise module): {acc*100:.2f}%")
    clean_accuracy["IMDB_BERT"] = f"{acc*100:.2f}%"
    noise_position={
        'input_noise':[0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1],
        'pre_att_cls':[0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1],
        'pre_att_all':[0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1],
        "post_att_cls":[0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1],
        "post_att_all":[0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1], 
        'last_cls':[0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1], 
        'logits':[0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1]
    }
    positions = [ 'input_noise', 'pre_att_cls', 'pre_att_all',"post_att_cls","post_att_all", 'last_cls', 'logits']
    noise_position={
        'input_noise':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
        'pre_att_cls':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
        'pre_att_all':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
        "post_att_cls":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
        "post_att_all":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], 
        'last_cls':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], 
        'logits':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    }
    positions = [ 'input_noise', 'pre_att_cls', 'pre_att_all',"post_att_cls","post_att_all", 'last_cls', 'logits']
    #noise_position={
    #    'input_noise':[1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2],
    #    'pre_att_cls':[1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2],
    #    'pre_att_all':[1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2],
    #    "post_att_cls":[1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2],
    #    "post_att_all":[1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2], 
    #    'last_cls':[1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2], 
    #    'logits':[1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
    #}
    #positions = [ 'post_att_cls', 'last_cls', 'logits']
    for repetitions in range(0,num_repetitions):
        for position in positions:
            for noise in noise_position[position]:
                model.change_defense(defense_cls="random_noise",def_position=position,noise_sigma=noise,defense=True)
                BERT = HuggingFaceModelWrapper(model,tokenizer)
                BERT.to("cuda")
                y_pred_BERT = []
                for i in tqdm(range(0,len(bert_input)//batch)):
                    y_pred_BERT.extend(torch.argmax(BERT(bert_input[i*batch:(i+1)*batch]),dim=-1).tolist())
                # Evaluation
                acc = accuracy_score(test_labels, y_pred_BERT)
                print(f"IMDB_BERT_{'random_noise'}_{position}_{str(noise)} = {acc*100:.2f}%")
                clean_accuracy[f"IMDB_BERT-WITH-0.1-SCALE_{'random_noise'}_{position}_{str(noise)}"] = f"{acc*100:.2f}%"
                print(clean_accuracy)
        
        # Serializing json
        json_object = json.dumps(clean_accuracy, indent=4)
        
        # Writing to sample.json
        with open(f"IMDB_BERT_0.1_scale_clean_from_0_{repetitions}.json", "w") as outfile:
            outfile.write(json_object)
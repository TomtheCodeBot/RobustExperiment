from textattack.datasets import HuggingFaceDataset
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
import model
import json
import model as model_wrap
from model.TextDefenseExtraWrapper import wrapping_model
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
    args = parser.parse_args()
    batch=1500
    sst2_dataset = datasets.load_dataset("ag_news")
    train_data = sst2_dataset["train"]
    test_data = sst2_dataset["test"]    
    test_labels = np.array(test_data["label"])
    clean_accuracy={}
    num_repetitions = 3
    bert_input = list(test_data["text"])
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-ag-news",use_fast=True)
    tokenizer.model_max_length=128
    tokenizer_roberta = AutoTokenizer.from_pretrained("roberta-base",use_fast=True)
    
    
    config = AutoConfig.from_pretrained("textattack/bert-base-uncased-ag-news")
    model = BertForSequenceClassification(config)
    state = AutoModelForSequenceClassification.from_pretrained(
        "textattack/bert-base-uncased-ag-news"
    )
    model.load_state_dict(state.state_dict())
    model.to("cuda")
    model.eval()
    BERT = HuggingFaceModelWrapper(model, tokenizer)
    y_pred_BERT = []
    batch_iter = len(bert_input)//batch + (1 if len(bert_input)%batch!=0 else 0 )
    for i in tqdm(range(0,batch_iter)):
        y_pred_BERT.extend(torch.argmax(BERT(bert_input[i*batch:(i+1)*batch]),dim=-1).tolist())
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_BERT)
    print(f"AGNEWS BERT (with noise module): {acc*100:.2f}%")
    clean_accuracy["AGNEWS_BERT"] = f"{acc*100:.2f}%"
    
    #ascc_model = model.TextDefense_model_builder("bert","bert-base-uncased","ascc",device,dataset_name="agnews")
    #load_path = "model/weights/VinAI_weights/tmd_ckpts/TextDefender/saved_models/agnews_bert/ascc-len128-epo10-batch32-best.pth"
    #print(ascc_model.load_state_dict(torch.load(load_path,map_location = device), strict=False))
    #tokenizer.model_max_length=128
    #ascc_model = wrapping_model(ascc_model,tokenizer,"ascc")
    #y_pred_BERT = []
    #for i in tqdm(range(0,len(bert_input)//batch)):
    #    y_pred_BERT.extend(torch.argmax(torch.tensor(ascc_model(bert_input[i*batch:(i+1)*batch])),dim=-1).tolist())
    #acc = accuracy_score(test_labels, y_pred_BERT)
    #print(f"AGNEWS BERT ASCC: {acc*100:.2f}%")
    
    #dne_model = model.TextDefense_model_builder("bert","bert-base-uncased","dne",device,dataset_name="agnews")
    #load_path = "model/weights/tmd_ckpts/TextDefender/saved_models/agnews_bert/dne-len128-epo10-batch32-best.pth"
    #print(dne_model.load_state_dict(torch.load(load_path,map_location = device), strict=False))
    #dne_model = wrapping_model(dne_model,tokenizer,"dne",batch_size=batch)
    #bert_input = list(test_data["text"])
    #y_pred_BERT = []
    #for i in tqdm(range(0,len(bert_input)//batch)):
    #    y_pred_BERT.extend(torch.argmax(torch.tensor(dne_model(bert_input[i*batch:(i+1)*batch])),dim=-1).tolist())
    #acc = accuracy_score(dne_model, y_pred_BERT)
    #print(f"AGNEWS BERT DNE: {acc*100:.2f}%")
    
    #tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/RobustExperiment/model/weights/VinAI_weights/tmd_ckpts/manifold_defense/models/roberta-base-agnews",use_fast=True)
    #config = AutoConfig.from_pretrained("/home/ubuntu/RobustExperiment/model/weights/VinAI_weights/tmd_ckpts/manifold_defense/models/roberta-base-agnews")
    #model = RobertaForSequenceClassification(config)
    #state = AutoModelForSequenceClassification.from_pretrained("/home/ubuntu/RobustExperiment/model/weights/VinAI_weights/tmd_ckpts/manifold_defense/models/roberta-base-agnews")
    #model.load_state_dict(state.state_dict())
    #model.eval()
    #RoBERTa = HuggingFaceModelWrapper(model,tokenizer)
    #RoBERTa.to("cuda")
    #y_pred_BERT = []
    #for i in tqdm(range(0,len(bert_input)//batch)):
    #    y_pred_BERT.extend(torch.argmax(RoBERTa(bert_input[i*batch:(i+1)*batch]),dim=-1).tolist())
    ## Evaluation
    #acc = accuracy_score(test_labels, y_pred_BERT)
    #print(f"AGNEWS RoBERTa (with noise module): {acc*100:.2f}%")
    #clean_accuracy["AGNEWS_RoBERTa"] = f"{acc*100:.2f}%"
    
    #load_path = "/home/ubuntu/RobustExperiment/model/weights/VinAI_weights/bert-base-uncased-ag-news"
    #gm_path = "/home/ubuntu/RobustExperiment/model/weights/VinAI_weights/tmd_ckpts/tmd/outputs/infogan_bert_agnews/manifold-defense/42b0465v/checkpoints/epoch=99-step=10599.ckpt"
    #dne_model = model_wrap.TextDefense_model_builder("bert",load_path,"tmd",gm_path = gm_path,device="cuda")
    #tokenizer_tmd = AutoTokenizer.from_pretrained("/home/ubuntu/RobustExperiment/model/weights/VinAI_weights/bert-base-uncased-ag-news",use_fast=True)
    #dne_model = wrapping_model(dne_model,tokenizer_tmd,"tmd")
    #y_pred_BERT = []
    #for i in tqdm(range(0,len(bert_input)//batch)):
    #    y_pred_BERT.extend(torch.argmax(dne_model(bert_input[i*batch:(i+1)*batch]),dim=-1).tolist())
    ## Evaluation
    #acc = accuracy_score(test_labels, y_pred_BERT)
    #print(f"AGNEWS TMD (with noise module): {acc*100:.2f}%")
    
    #freelb_model = model_wrap.TextDefense_model_builder("bert","bert-base-uncased","freelb",device,dataset_name="agnews")
    #load_path = "/home/ubuntu/TextDefender/saved_models/ag_news_bert/freelb-len128-epo5-batch32-advstep5-advlr0.03-norm0.0-best.pth"
    #tokenizer.model_max_length=128
    #print(freelb_model.load_state_dict(torch.load(load_path,map_location = device), strict=False))
    #BERT_FREELB = wrapping_model(freelb_model,tokenizer,"freelb")
    #y_pred_BERT = []
    #for i in tqdm(range(0,len(bert_input)//batch)):
    #    y_pred_BERT.extend(torch.argmax(BERT_FREELB(bert_input[i*batch:(i+1)*batch]),dim=-1).tolist())
    ## Evaluation
    #acc = accuracy_score(test_labels, y_pred_BERT)
    #print(f"AGNEWS BERT_FREELB (with noise module): {acc*100:.2f}%")
    
    #infobert_model = model_wrap.TextDefense_model_builder("bert","bert-base-uncased","infobert",device,dataset_name="agnews")
    #load_path = "/home/ubuntu/TextDefender/saved_models/ag_news_bert/infobert-len128-epo5-batch32-advstep3-advlr0.04-norm0-best.pth"
    #tokenizer.model_max_length=128
    #print(infobert_model.load_state_dict(torch.load(load_path,map_location = device), strict=False))
    #BERT_INFOBERT = wrapping_model(infobert_model,tokenizer,"infobert")
    #y_pred_BERT = []
    #for i in tqdm(range(0,len(bert_input)//batch)):
    #    y_pred_BERT.extend(torch.argmax(BERT_INFOBERT(bert_input[i*batch:(i+1)*batch]),dim=-1).tolist())
    ## Evaluation
    #acc = accuracy_score(test_labels, y_pred_BERT)
    #print(f"AGNEWS BERT_INFOBERT (with noise module): {acc*100:.2f}%")
    
    #ascc_model = model.TextDefense_model_builder("roberta","roberta-base","ascc",device,dataset_name="agnews")
    #load_path = "/home/ubuntu/RobustExperiment/model/weights/VinAI_weights/tmd_ckpts/TextDefender/saved_models/agnews_roberta/ascc-len128-epo10-batch32-best.pth"
    #print(ascc_model.load_state_dict(torch.load(load_path,map_location = device), strict=False))
    #tokenizer_roberta.model_max_length=128
    #ascc_model = wrapping_model(ascc_model,tokenizer_roberta,"ascc")
    #y_pred_BERT = []
    #for i in tqdm(range(0,len(bert_input)//batch)):
    #    y_pred_BERT.extend(torch.argmax(torch.tensor(ascc_model(bert_input[i*batch:(i+1)*batch])),dim=-1).tolist())
    #acc = accuracy_score(test_labels, y_pred_BERT)
    #print(f"AGNEWS ROBERTA ASCC: {acc*100:.2f}%")
    
    
    #load_path = "/home/ubuntu/RobustExperiment/model/weights/VinAI_weights/tmd_ckpts/manifold_defense/models/roberta-base-agnews"
    #gm_path = "/home/ubuntu/RobustExperiment/model/weights/VinAI_weights/tmd_ckpts/manifold_defense/outputs/infogan_roberta_agnews/6us3wbhr/checkpoints/epoch=99-step=10599.ckpt"
    #tmd = model.TextDefense_model_builder("roberta",load_path,"tmd",gm_path = gm_path,device="cuda",dataset_name="agnews")
    #tokenizer = AutoTokenizer.from_pretrained(load_path,use_fast=True)
    #ROBERTA_TMD = wrapping_model(tmd,tokenizer,"tmd")
    #y_pred_BERT = []
    #for i in tqdm(range(0,len(bert_input)//batch)):
    #    y_pred_BERT.extend(torch.argmax(ROBERTA_TMD(bert_input[i*batch:(i+1)*batch]),dim=-1).tolist())
    ## Evaluation
    #acc = accuracy_score(test_labels, y_pred_BERT)
    #print(f"AGNEWS ROBERTA TMD (with noise module): {acc*100:.2f}%")
    
    #noise_position={
    #    'input_noise':[0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1],
    #    'pre_att_cls':[0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1],
    #    'pre_att_all':[0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1],
    #    "post_att_cls":[0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1],
    #    "post_att_all":[0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1], 
    #    'last_cls':[0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1], 
    #    'logits':[0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1]
    #}
    #positions = [ 'input_noise', 'pre_att_cls', 'pre_att_all',"post_att_cls","post_att_all", 'last_cls', 'logits']
    
    #noise_position={
    #    'input_noise':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    #    'pre_att_cls':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    #    'pre_att_all':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    #    "post_att_cls":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    #    "post_att_all":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], 
    #    'last_cls':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], 
    #    'logits':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    #}
    #positions = [ 'pre_att_all','pre_att_cls','post_att_cls',"post_att_all", 'last_cls', 'logits']
    noise_position={
        'input_noise':[1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2],
        'pre_att_cls':[1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2],
        'pre_att_all':[1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2],
        "post_att_cls":[1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2],
        "post_att_all":[1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2], 
        'last_cls':[1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2], 
        'logits':[1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
    }
    positions = [ 'pre_att_cls','post_att_cls', 'last_cls', 'logits']
    for repetitions in range(0,num_repetitions):
        for position in positions:
            for noise in noise_position[position]:
                model.change_defense(defense_cls="random_noise",def_position=position,noise_sigma=noise,defense=True)
                y_pred_BERT = []
                for i in tqdm(range(0,batch_iter)):
                    y_pred_BERT.extend(torch.argmax(BERT(bert_input[i*batch:(i+1)*batch]),dim=-1).tolist())
                # Evaluation
                acc = accuracy_score(test_labels, y_pred_BERT)
                print(f"AGNEWS_BERT_{'random_noise'}_{position}_{str(noise)} = {acc*100:.2f}%")
                clean_accuracy[f"AGNEWS_BERT_{'random_noise'}_{position}_{str(noise)}"] = f"{acc*100:.2f}%"
                print(clean_accuracy)
        
        # Serializing json
        json_object = json.dumps(clean_accuracy, indent=4)
        # Writing to sample.json
        with open(f"AGNEWS_0.1_scale_max_2_clean_accuracy_{repetitions}.json", "w") as outfile:
            outfile.write(json_object)
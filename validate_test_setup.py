import torch
from textattack.datasets import HuggingFaceDataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from typing import List, Dict
import random
from textattack.attack_recipes import TextFoolerJin2019,DeepWordBugGao2018,TextBuggerLi2018
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.goal_functions import UntargetedClassification
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from utils.dataloader import load_train_test_imdb_data
from model.BERTNoiseDefend import BertForSequenceClassification
from transformers import AutoModelForSequenceClassification,AutoTokenizer,AutoConfig
from utils.preprocessing import clean_text_imdb
from sklearn.feature_extraction.text import CountVectorizer
from textattack.models.wrappers import PyTorchModelWrapper,HuggingFaceModelWrapper
import argparse
from textattack import Attacker,AttackArgs
from textattack.datasets import Dataset
import datasets
import numpy as np
import os 

import json
 
class CustomModelWrapper(PyTorchModelWrapper):
    def __init__(self,model,tokenizer):
        super(CustomModelWrapper,self).__init__(model,tokenizer)

    def __call__(self,text_input_list):
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

        if isinstance(outputs,tuple):
            return outputs[-1]#model-h,model-bh

        if isinstance(outputs,torch.Tensor):
            return outputs#baseline

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

def build_attacker(model,args):
    if (args['attack_method'] == 'textfooler'):
        attacker=TextFoolerJin2019.build(model)
    elif (args['attack_method'] == 'textbugger'):
        attacker=TextBuggerLi2018.build(model)
    elif (args['attack_method'] == 'deepwordbug'):
        attacker=DeepWordBugGao2018.build(model)
    else:
        attacker=TextFoolerJin2019.build(model)
    if(args['modify_ratio']!=0):
        attacker.constraints.append(MaxWordsPerturbed(max_percent=args['modify_ratio']))
    return attacker

def str2bool(strIn):
    if strIn.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif strIn.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        print(strIn)
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
def set_seed(seed=42):
    if seed is not None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True
        
def attack(args,wrapper,name,dataset):
    attackMethod=args.attack_method
    attack_args_dict = {'attack_method': attackMethod, 'attack_times': 1,'attack_examples':int(args.attack_examples),'modify_ratio':float(args.modify_ratio),
            'log_path': '{}/{}/{}_{}-{}.txt'.format(args.load_path, attackMethod,args.attack_dataset,args.modify_ratio,name)}
    attack = build_attacker(wrapper, attack_args_dict)
    attack_args = AttackArgs(num_examples=attack_args_dict['attack_examples'], log_to_txt=attack_args_dict['log_path'], csv_coloring_style="file")
    attacker=Attacker(attack,dataset,attack_args)
    attacker.attack_dataset()
    
def gen_dataset(instances):
    test_instances=instances
    test_dataset=[]
    for instance in range(len(test_instances)):
        test_dataset.append((test_instances["text"][instance],int(test_instances["label"][instance])))
    dataset=Dataset(test_dataset,shuffle=True)
    return dataset

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="parameter")
    parser.add_argument('-hd','--hidden_size',default=500)
    parser.add_argument('-be','--beta',default='0.1')
    parser.add_argument('-b','--batch_size',default=64)
    parser.add_argument('-s','--seed',default=0)
    parser.add_argument('-d','--dropout',default=0.1)
    parser.add_argument('-g','--gpu_num',default=2)
    parser.add_argument('-a','--attack',type=str2bool,nargs='?',const=False)
    parser.add_argument('-l','--load',type=str2bool,nargs='?',const=False)
    parser.add_argument('-t', '--test', type=str2bool, nargs='?', const=False)
    parser.add_argument('-lp', '--load_path', default=None)
    parser.add_argument('-am', '--attack_method', default=None)
    parser.add_argument('-ai', '--adv_init_mag', default=0.08)
    parser.add_argument('-as', '--adv_steps', default=3)
    parser.add_argument('-al', '--adv_lr', default=0.04)
    parser.add_argument('-amn','--adv_max_norm',default=None)
    parser.add_argument('-ad','--attack_dataset',default='test')#attack dataset & accuracy dataset
    parser.add_argument('-ae','--attack_examples',default=200)
    parser.add_argument('-mr','--modify_ratio',default=0.15)
    parser.add_argument('-e', '--epoch', default=10)
    parser.add_argument('-se','--save_epoch',type=str2bool,nargs='?',const=False)
    parser.add_argument('-c','--class_num',default=2)
    parser.add_argument('-dp','--dataset_path',default='dataset/sst2')
    parser.add_argument('-sp','--save_path',default='')
    args = parser.parse_args()
    
    vectorizer = CountVectorizer(stop_words="english",
                            preprocessor=clean_text_imdb, 
                            min_df=0)

    """train_data , test_data = load_train_test_imdb_data("/home/ubuntu/RobustExperiment/data/aclImdb")
    vectorizer = CountVectorizer(stop_words="english",
                                preprocessor=clean_text_imdb, 
                                min_df=0)
    training_features = vectorizer.fit_transform(train_data["text"])
    training_labels = train_data["label"]
    test_features = vectorizer.transform(test_data["text"])
    test_labels = test_data["label"]"""
    sst2_dataset = datasets.load_dataset("ag_news")
    train_data = sst2_dataset["train"]
    test_data = sst2_dataset["test"]
    training_features = vectorizer.fit_transform(train_data["text"])
    training_labels = np.array(train_data["label"])
    test_features = vectorizer.transform(test_data["text"])
    test_labels = np.array(test_data["label"])
    
    """tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb",use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
    BERT = HuggingFaceModelWrapper(model,tokenizer)
    BERT.to("cuda")
    batch = 100
    bert_input = list(test_data["text"])
    y_pred_BERT = []
    for i in tqdm(range(0,len(bert_input)//batch)):
        y_pred_BERT.extend(torch.argmax(BERT(bert_input[i*batch:(i+1)*batch]),dim=-1).tolist())
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_BERT)
    print(f"AGNEWS BERT (no noise module): {acc*100:.2f}%")
    
    
    config = AutoConfig.from_pretrained("textattack/bert-base-uncased-imdb")
    model = BertForSequenceClassification(config)
    state = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
    model.load_state_dict(state.state_dict())
    model.eval()
    BERT = HuggingFaceModelWrapper(model,tokenizer)
    BERT.to("cuda")
    
    batch = 100
    bert_input = list(test_data["text"])
    y_pred_BERT = []
    for i in tqdm(range(0,len(bert_input)//batch)):
        y_pred_BERT.extend(torch.argmax(BERT(bert_input[i*batch:(i+1)*batch]),dim=-1).tolist())
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_BERT)
    print(f"AGNEWS BERT (with noise module): {acc*100:.2f}%")
    
    
    sst2_dataset = datasets.load_dataset("ag_news")
    train_data = sst2_dataset["train"]
    test_data = sst2_dataset["test"]
    training_features = vectorizer.fit_transform(train_data["text"])
    training_labels = np.array(train_data["label"])
    test_features = vectorizer.transform(test_data["text"])
    test_labels = np.array(test_data["label"])
    """
    tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-ag-news",use_fast=True)
    batch = 100
    bert_input = list(test_data["text"])
    num_repetitions = 3
    clean_accuracy={}
    """
    model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
    BERT = HuggingFaceModelWrapper(model,tokenizer)
    BERT.to("cuda")
    
    y_pred_BERT = []
    for i in tqdm(range(0,len(bert_input)//batch)):
        y_pred_BERT.extend(torch.argmax(BERT(bert_input[i*batch:(i+1)*batch]),dim=-1).tolist())
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_BERT)
    print(f"AGNEWS BERT (no noise module): {acc*100:.2f}%")
    """
    
    config = AutoConfig.from_pretrained("textattack/bert-base-uncased-ag-news")
    model = BertForSequenceClassification(config)
    state = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-ag-news")
    model.load_state_dict(state.state_dict())
    model.eval()
    BERT = HuggingFaceModelWrapper(model,tokenizer)
    print("123")
    BERT.to("cuda:1")
    print("123")
    y_pred_BERT = []
    for i in tqdm(range(0,len(bert_input)//batch)):
        y_pred_BERT.extend(torch.argmax(BERT(bert_input[i*batch:(i+1)*batch]),dim=-1).tolist())
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_BERT)
    print(f"AGNEWS BERT (with noise module): {acc*100:.2f}%")
    clean_accuracy["AGNEWS_BERT-WITH-0.1-SCALE"] = f"{acc*100:.2f}%"
    """noise_position={
        'input_noise':[0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1],
        'pre_att_cls':[0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1],
        'pre_att_all':[0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1],
        "post_att_cls":[0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1],
        "post_att_all":[0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1], 
        'last_cls':[0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1], 
        'logits':[0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1]
    }
    positions = [ 'input_noise', 'pre_att_cls', 'pre_att_all',"post_att_cls","post_att_all", 'last_cls', 'logits']"""
    noise_position={
        'input_noise':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
        'pre_att_cls':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
        'pre_att_all':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
        "post_att_cls":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
        "post_att_all":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], 
        'last_cls':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], 
        'logits':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    }
    positions = [ 'pre_att_all',"post_att_all", 'last_cls', 'logits']
    for repetitions in range(0,num_repetitions):
        for position in positions:
            for noise in noise_position[position]:
                model.change_defense(defense_cls="random_noise",def_position=position,noise_sigma=noise,defense=True)
                y_pred_BERT = []
                for i in tqdm(range(0,len(bert_input)//batch)):
                    y_pred_BERT.extend(torch.argmax(BERT(bert_input[i*batch:(i+1)*batch]),dim=-1).tolist())
                # Evaluation
                acc = accuracy_score(test_labels, y_pred_BERT)
                print(f"AGNEWS_BERT-WITH-0.1-SCALE_{'random_noise'}_{position}_{str(noise)} = {acc*100:.2f}%")
                clean_accuracy[f"AGNEWS_BERT-WITH-0.1-SCALE_{'random_noise'}_{position}_{str(noise)}"] = f"{acc*100:.2f}%"
                print(clean_accuracy)
        
        # Serializing json
        json_object = json.dumps(clean_accuracy, indent=4)
        
        # Writing to sample.json
        with open(f"AGNEWS_0.1_scale_clean_accuracy_{repetitions}.json", "w") as outfile:
            outfile.write(json_object)
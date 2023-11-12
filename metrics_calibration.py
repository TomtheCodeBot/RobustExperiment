"""
Metrics for assessing the quality of predictive uncertainty quantification.
"""
from typing import Any, NoReturn, Tuple, Union, Optional
from argparse import Namespace

import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm
from scipy.special import softmax

import sys
sys.path.insert(1, '.')
import os
os.environ['CURL_CA_BUNDLE'] = ''
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

from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack import Attack
from textattack.models.wrappers import (
    ModelWrapper,
    PyTorchModelWrapper,
    SklearnModelWrapper,
    HuggingFaceModelWrapper,
)
from textattack.models.helpers import LSTMForClassification
import torch
import argparse
from textattack import Attacker, AttackArgs
from textattack.datasets import Dataset
import datasets
import numpy as np
import os
import model as model_lib
from textattack.shared import AttackedText, utils
from tqdm import tqdm
from model.TextDefenseExtraWrapper import wrapping_model


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
    
    
def ece_score(y_pred, y_test, n_bins=15):
    py = softmax(y_pred, axis=1) if y_pred.max() > 1 else y_pred

    py = np.array(py)
    y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)
if __name__ == "__main__":
    
    sst2_dataset = datasets.load_dataset("ag_news")
    train_data = sst2_dataset["train"]
    test_data = sst2_dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased", use_fast=True
    )
    
    tokenizer.model_max_length = 128
    device = "cuda"
    
    config = AutoConfig.from_pretrained("textattack/bert-base-uncased-ag-news")
    model = BertForSequenceClassification(config)
    state = AutoModelForSequenceClassification.from_pretrained(
        "textattack/bert-base-uncased-ag-news"
    )
    model.load_state_dict(state.state_dict(), strict=False)
    model.to("cuda")
    model.eval()
    BERT = HuggingFaceModelWrapper(model, tokenizer)
    
    args = {}
    args["attack_method"] = "bertattack"
    args["k_neighbor"] = 50
    args["modify_ratio"] = 0.3
    args["similarity"] = 0.84
    #attack  =  build_attacker_from_textdefender(BERT,args)
    important_ind = []
    std = []
    from sklearn.model_selection import train_test_split
    train_data,_,train_label,_=train_test_split(test_data["text"],test_data["label"],train_size=1000)
    train_data = list(train_data)
    
    ece_list =[]
    for  _ in range(10):
        #model.change_defense(defense_cls="random_noise",def_position="post_att_cls",noise_sigma=0.8,defense=True)
        pred = []
        for i in tqdm(range(len(train_data))):
        
            res = BERT([train_data[i]])
            pred.append(res.cpu().numpy())
        y_pred = np.concatenate(pred,axis=0)
        print(y_pred.shape)
        ece_list.append(ece_score(y_pred,list(train_label)))
    import statistics
    mean = statistics.mean(ece_list)
    stdev = statistics.stdev(ece_list)


    print(f"Mean: {mean}")
    print(f"Standard Deviation: {stdev}")
    
    
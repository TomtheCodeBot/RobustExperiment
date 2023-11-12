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
    
if __name__ == "__main__":
    
    sst2_dataset = datasets.load_dataset("ag_news")
    train_data = sst2_dataset["train"]
    test_data = sst2_dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased", use_fast=True
    )
    
    tokenizer_roberta = AutoTokenizer.from_pretrained(
        "roberta-base", use_fast=True
    )
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
    attack  =  build_attacker_from_textdefender(BERT,args)
    important_ind = []
    for i in tqdm(range(len(test_data["text"][:100]))):
        example = AttackedText(test_data["text"][i])
        attack.goal_function.init_attack_example(example,test_data["label"][i])
        indices = attack.search_method._get_index_order(example)
        important_ind.append(indices[0][0])
    important_ind2 = []
    #model.change_defense(defense_cls="random_noise",def_position="post_att_cls",noise_sigma=0.8,defense=True)
    BERT = HuggingFaceModelWrapper(model, tokenizer)
    attack  =  build_attacker_from_textdefender(BERT,args)
    for i in tqdm(range(len(test_data["text"][:100]))):
        example = AttackedText(test_data["text"][i])
        attack.goal_function.init_attack_example(example,test_data["label"][i])
        indices = attack.search_method._get_index_order(example)
        important_ind2.append(indices[0][0])
    print(np.sum(np.array(important_ind2)==np.array(important_ind)))
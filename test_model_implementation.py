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

from textattack.transformations import WordSwapEmbedding, WordSwapWordNet, WordSwapMaskedLM
from utils.dataloader import load_train_test_imdb_data

from model.BERTNoiseDefend import BertForSequenceClassification

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
from textattack.models.helpers import LSTMForClassification
import torch
import argparse
from textattack import Attacker, AttackArgs
from textattack.datasets import Dataset
import datasets
import numpy as np
import os
import model
from model.TextDefenseExtraWrapper import wrapping_model 
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",use_fast=True)
    dne_model = model.TextDefense_model_builder("bert","bert-base-uncased","dne",)
    load_path = "model/weights/VinAI_weights/tmd_ckpts/TextDefender/saved_models/imdb_bert/dne-len256-epo10-batch32-best.pth"
    print(dne_model.load_state_dict(torch.load(load_path,map_location = 'cpu'), strict=False))
    dne_model = wrapping_model(dne_model,tokenizer,"dne")
    print(dne_model(["helllooooo my god man I hope this work"]))
    dne_model = model.TextDefense_model_builder("bert","bert-base-uncased","ascc")
    load_path = "model/weights/VinAI_weights/tmd_ckpts/TextDefender/saved_models/imdb_bert/ascc-len256-epo10-batch32-best.pth"
    print(dne_model.load_state_dict(torch.load(load_path,map_location = 'cpu'), strict=False))
    dne_model = wrapping_model(dne_model,tokenizer,"ascc")
    print(dne_model(["helllooooo my god man I hope this work"]))
    load_path = "/home/ubuntu/RobustExperiment/model/weights/VinAI_weights/bert-base-uncased-ag-news"
    gm_path = "/home/ubuntu/RobustExperiment/model/weights/VinAI_weights/tmd_ckpts/tmd/outputs/infogan_bert_agnews/manifold-defense/42b0465v/checkpoints/epoch=99-step=10599.ckpt"
    dne_model = model.TextDefense_model_builder("bert",load_path,"tmd",gm_path = gm_path)
    dne_model = wrapping_model(dne_model,tokenizer,"tmd")
    print(dne_model(["helllooooo my god man I hope this work"]))
    
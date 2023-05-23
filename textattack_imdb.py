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
from utils.preprocessing import clean_text_imdb
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from model.robustNB import RobustNaiveBayesClassifierPercentage
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
        num_workers_per_device=int(args.num_workers_per_device),
        parallel=args.parallel
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
    parser.add_argument("-nd", "--num_workers_per_device", default=2)
    parser.add_argument('-pr', '--parallel',action='store_true')
    parser.add_argument("-en", "--ensemble_num", default=16)
    parser.add_argument("-eb", "--ensemble_batch_size", default=32)
    parser.add_argument("-rms", "--random_mask_rate", default=0.3)
    parser.add_argument("-spf", "--safer_pertubation_file", default="/home/ubuntu/TextDefender/dataset/imdb/perturbation_constraint_pca0.8_100.pkl")
    
    args = parser.parse_args()

    device = "cuda"
    train_data, test_data = load_train_test_imdb_data(
        "data/aclImdb"
    )

    #tokenizer = AutoTokenizer.from_pretrained(
    #    "textattack/bert-base-uncased-imdb", use_fast=True
    #)
    config = AutoConfig.from_pretrained("/home/duy/RobustExperiment/model/weights/bert-base-uncased-imdb")
    tokenizer_tmd = AutoTokenizer.from_pretrained(
        "/home/duy/RobustExperiment/model/weights/bert-base-uncased-imdb", use_fast=True
    )
    model = BertForSequenceClassification(config)
    state = AutoModelForSequenceClassification.from_pretrained(
        "/home/duy/RobustExperiment/model/weights/bert-base-uncased-imdb"
    )
    model.load_state_dict(state.state_dict())
    model.to("cuda")
    model.eval()
    BERT = HuggingFaceModelWrapper(model, tokenizer_tmd)
    
    #ascc_model = model_lib.TextDefense_model_builder("bert","bert-base-uncased","ascc",device)
    #load_path = "model/weights/tmd_ckpts/TextDefender/saved_models/imdb_bert/ascc-len256-epo10-batch32-best.pth"
    #print(ascc_model.load_state_dict(torch.load(load_path,map_location = device), strict=False))
    #ascc_model.to("cuda")
    #BERT_ASCC = wrapping_model(ascc_model,tokenizer,"ascc")
    
    #tokenizer = AutoTokenizer.from_pretrained(
    #    "bert-base-uncased", use_fast=True
    #)
    #mask_model = model_lib.TextDefense_model_builder("bert","bert-base-uncased","mask",device)
    #load_path = "/home/duy/TextDefender/saved_models/imdb_bert/mask-len256-epo10-batch32-rate0.3-best.pth"
    #tokenizer.model_max_length=256
    #print(mask_model.load_state_dict(torch.load(load_path,map_location = device), strict=False))
    #BERT_MASK = wrapping_model(mask_model,tokenizer,"mask",ensemble_num=args.ensemble_num,batch_size=args.ensemble_batch_size,ran_mask=args.random_mask_rate)
    
    #tokenizer = AutoTokenizer.from_pretrained(
    #    "bert-base-uncased", use_fast=True
    #)
    #safer_model = model_lib.TextDefense_model_builder("bert","bert-base-uncased","safer",device)
    #load_path = "/home/khoa/duyhc/RobustExperimen/model/weights/tmd_ckpts/TextDefender/saved_models/imdb_bert/safer-len256-epo10-batch32-best.pth"
    #tokenizer.model_max_length=256
    #print(safer_model.load_state_dict(torch.load(load_path,map_location = device), strict=False))
    #BERT_SAFER = wrapping_model(safer_model,tokenizer,"safer",ensemble_num=args.ensemble_num,batch_size=args.ensemble_batch_size,safer_aug_set=args.safer_pertubation_file)
    
    #freelb_model = model_lib.TextDefense_model_builder("bert","bert-base-uncased","freelb",device)
    #load_path = "/home/ubuntu/TextDefender/saved_models/imdb_bert/freelb-len256-epo10-batch32-advstep5-advlr0.03-norm0.0-best.pth"
    #tokenizer.model_max_length=256
    #print(freelb_model.load_state_dict(torch.load(load_path,map_location = device), strict=False))
    #freelb_model.to("cuda")
    #BERT_FREELB = wrapping_model(freelb_model,tokenizer,"freelb")
    
    #info_model = model_lib.TextDefense_model_builder("bert","bert-base-uncased","infobert",device)
    #load_path = "/home/ubuntu/TextDefender/saved_models/imdb_bert/infobert-len256-epo10-batch32-advstep3-advlr0.04-norm0-best.pth"
    #tokenizer.model_max_length=256
    #print(info_model.load_state_dict(torch.load(load_path,map_location = device), strict=False))
    #info_model.to("cuda")
    #BERT_INFOBERT = wrapping_model(info_model,tokenizer,"infobert")
    
    #load_path = "/home/duy/RobustExperiment/model/weights/bert-base-uncased-imdb"
    #gm_path = "/home/khoa/duyhc/RobustExperimen/model/weights/tmd_ckpts/tmd/outputs/infogan_bert_imdb/manifold-defense/yutbyyz5/checkpoints/epoch=99-step=2199.ckpt"
    #tmd = model_lib.TextDefense_model_builder("bert",load_path,"tmd",gm_path = gm_path,device="cuda")
    #tokenizer = AutoTokenizer.from_pretrained("/home/duy/RobustExperiment/model/weights/bert-base-uncased-imdb",use_fast=True)
    #BERT_TMD = wrapping_model(tmd,tokenizer,"tmd")
    
    #config = AutoConfig.from_pretrained("model/weights/tmd_ckpts/manifold_defense/models/roberta-base-imdb")
    #model_roberta = RobertaForSequenceClassification(config)
    #tokenizer_tmd_roberta = AutoTokenizer.from_pretrained(
    #   "model/weights/tmd_ckpts/manifold_defense/models/roberta-base-imdb", use_fast=True
    #)
    #state = AutoModelForSequenceClassification.from_pretrained(
    #    "model/weights/tmd_ckpts/manifold_defense/models/roberta-base-imdb"
    #)
    #model_roberta.load_state_dict(state.state_dict())
    #model_roberta.to("cuda")
    #model_roberta.eval()
    #ROBERTA = HuggingFaceModelWrapper(model_roberta, tokenizer_tmd_roberta)
    
    #load_path = "/home/khoa/duyhc/RobustExperimen/model/weights/tmd_ckpts/manifold_defense/models/roberta-base-imdb"
    #gm_path = "/home/khoa/duyhc/RobustExperimen/model/weights/tmd_ckpts/manifold_defense/outputs/infogan_roberta_imdb/bvi8ln2v/checkpoints/epoch=99-step=2199.ckpt"
    #tmd = model_lib.TextDefense_model_builder("roberta",load_path,"tmd",gm_path = gm_path,device="cuda",dataset_name="imdb")
    #tokenizer = AutoTokenizer.from_pretrained(load_path,use_fast=True)
    #ROBERTA_TMD = wrapping_model(tmd,tokenizer,"tmd")
    
    #tokenizer_roberta = AutoTokenizer.from_pretrained(
    #    "roberta-base", use_fast=True
    #)
    #ascc_roberta_model = model_lib.TextDefense_model_builder("roberta","roberta-base","ascc",device)
    #load_path = "/home/khoa/duyhc/RobustExperimen/model/weights/tmd_ckpts/TextDefender/saved_models/imdb_roberta/ascc-len256-epo10-batch32-best.pth"
    #print(ascc_roberta_model.load_state_dict(torch.load(load_path,map_location = device), strict=False))
    #ascc_roberta_model.to("cuda")
    #tokenizer_roberta.model_max_length=256
    #ROBERTA_ASCC = wrapping_model(ascc_roberta_model,tokenizer_roberta,"ascc")
    
    
    #tokenizer_roberta = AutoTokenizer.from_pretrained(
    #    "roberta-base", use_fast=True
    #)
    #infobert_roberta_model = model_lib.TextDefense_model_builder("roberta","roberta-base","infobert",device)
    #load_path = "/home/ubuntu/TextDefender/saved_models/imdb_roberta/infobert-len256-epo10-batch32-advstep3-advlr0.04-norm0-best.pth"
    #print(infobert_roberta_model.load_state_dict(torch.load(load_path,map_location = device), strict=False))
    #infobert_roberta_model.to("cuda")
    #tokenizer_roberta.model_max_length=256
    #ROBERTA_INFOBERT = wrapping_model(infobert_roberta_model,tokenizer_roberta,"infobert")
    
    #tokenizer_roberta = AutoTokenizer.from_pretrained(
    #    "roberta-base", use_fast=True
    #)
    #freelb_roberta_model = model_lib.TextDefense_model_builder("roberta","roberta-base","freelb",device)
    #load_path = "/home/ubuntu/TextDefender/saved_models/imdb_roberta/freelb-len256-epo10-batch32-advstep5-advlr0.03-norm0.0-best.pth"
    #print(freelb_roberta_model.load_state_dict(torch.load(load_path,map_location = device), strict=False))
    #freelb_roberta_model.to("cuda")
    #tokenizer_roberta.model_max_length=256
    #ROBERTA_FREELB = wrapping_model(freelb_roberta_model,tokenizer_roberta,"freelb")
    
    with torch.no_grad():
        
        #noise_pos = { "pre_att_all": [0.1,0.2], "post_att_all": [0.1,0.2, 0.3]}
        #noise_pos_roberta = { "pre_att_all": [0.1,0.2], "post_att_all": [0.2, 0.3]}
        
        noise_pos = { "post_att_cls": [0.55]}
        #noise_pos_roberta = {"post_att_cls": [0.9, 1], "pre_att_cls": [0.3,0.4]}
        #noise_pos_roberta = {"post_att_cls": [1.3]}
        
        list_attacks = ["textbugger","bertattack"]
        for i in range(0, 1):
            set_seed(i)
            dataset = gen_dataset(test_data)
            args.load_path = (
                f"noise_defense_attack_result/paper_default setting/IMDB/{i}/"
            )
            for attack_method in list_attacks:
                args.attack_method = attack_method
                #attack(args, BERT, "BERT", dataset)
                for key in noise_pos.keys():
                    for noise_intensity in noise_pos[key]:
                        model.change_defense(defense_cls="random_noise",def_position=key,noise_sigma=noise_intensity,defense=True)
                        BERT = HuggingFaceModelWrapper(model, tokenizer_tmd)
                        attack(args, BERT, f"BERT_{key}_{noise_intensity}", dataset)
                model.change_defense(defense=False)
                #attack(args, BERT_ASCC, "BERT_ASCC", dataset)
                #attack(args, BERT_FREELB, "BERT_FREELB", dataset)
                #attack(args, BERT_INFOBERT, "BERT_INFOBERT", dataset)
                #attack(args, BERT_TMD, "BERT_TMD", dataset)
                #attack(args, BERT_MASK, "BERT_MASK", dataset)
                
                #attack(args, ROBERTA, "ROBERTA", dataset)
                #for key in noise_pos_roberta.keys():
                #    for noise_intensity in noise_pos_roberta[key]:
                #        model_roberta.change_defense(defense_cls="random_noise",def_position=key,noise_sigma=noise_intensity,defense=True)
                #        attack(args, ROBERTA, f"ROBERTA_{key}_{noise_intensity}", dataset)
                #model_roberta.change_defense(defense=False)
                #attack(args, ROBERTA_TMD, "ROBERTA_TMD", dataset)
                #attack(args, ROBERTA_ASCC, "ROBERTA_ASCC", dataset)
                #attack(args, ROBERTA_INFOBERT, "ROBERTA_INFOBERT", dataset)
                #attack(args, ROBERTA_FREELB, "ROBERTA_FREELB", dataset)
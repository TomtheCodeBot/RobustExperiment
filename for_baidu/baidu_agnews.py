import sys
sys.path.insert(1, '.')
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
from model.TextDefenseExtraWrapper import wrapping_model
import time
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
    attacker.goal_function = UntargetedClassification(model, query_budget=args["k_neighbor"] // (1 if args['same_niters'] else args['ensemble_num']))
    return Attack(attacker.goal_function, attacker.constraints + attacker.pre_transformation_constraints,
                  attacker.transformation, attacker.search_method)

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
    if args["consine_sim"] != 0:
        print("AGHHHHHHHHHHHHHH")
        attacker.constraints.append(
            UniversalSentenceEncoder(threshold=args["consine_sim"], metric="cosine")
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
        "ensemble_num": args.ensemble_num,
        "same_niters": args.same_niters,
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
    for instance in iter(test_instances):
        test_dataset.append(
            (instance['text'], int(instance['label']))
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
    parser.add_argument("-mr", "--modify_ratio", default=0.3)
    parser.add_argument("-sm", "--similarity", default=0.84)
    parser.add_argument("-kn", "--k_neighbor", default=50)
    parser.add_argument("-nd", "--num_workers_per_device", default=2)
    parser.add_argument('-pr', '--parallel',action='store_true')
    parser.add_argument("-en", "--ensemble_num", default=100, type=int)
    parser.add_argument("-eb", "--ensemble_batch_size", default=32)
    parser.add_argument("-rmr", "--random_mask_rate", default=0.9)
    parser.add_argument("-spf", "--safer_pertubation_file", default="/home/ubuntu/TextDefender/dataset/ag_news/perturbation_constraint_pca0.8_100.pkl")
    parser.add_argument("-md", "--model", default="bert")
    parser.add_argument("-df", "--defense", default="mask")
    parser.add_argument("--noise", default=0, type=float)
    parser.add_argument("--same-niters", action="store_true")
    args = parser.parse_args()
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
    
    if args.model == "bert":
        print("bert")
        if args.defense == "mask":
            print("mask")
            mask_model = model_lib.TextDefense_model_builder("bert","bert-base-uncased","mask",device,dataset_name="agnews")
            load_path = "model/weights/agnews/mask-len128-epo10-batch32-rate0.9-best.pth"
            tokenizer.model_max_length=128
            print(mask_model.load_state_dict(torch.load(load_path,map_location = device), strict=False))
            BERT_MASK = wrapping_model(mask_model,tokenizer,"mask",ensemble_num=args.ensemble_num,batch_size=args.ensemble_batch_size,ran_mask=args.random_mask_rate,safer_aug_set=None)
        elif args.defense == "safer":
            #print("safer")
            #safer_model = model_lib.TextDefense_model_builder("bert","bert-base-uncased","safer",device,dataset_name="agnews")
            #load_path = "model/weights/tmd_ckpts/TextDefender/saved_models/agnews_bert/safer-len128-epo10-batch32-best.pth"
            #tokenizer.model_max_length=128
            #print(safer_model.load_state_dict(torch.load(load_path,map_location = device), strict=False))
            #BERT_SAFER = wrapping_model(safer_model,tokenizer,"safer",ensemble_num=args.ensemble_num,batch_size=args.ensemble_batch_size,safer_aug_set="model/weights/agnews/perturbation_constraint_pca0.8_100.pkl")
            config = AutoConfig.from_pretrained("textattack/bert-base-uncased-ag-news")
            safer_model = BertForSequenceClassification(config)
            state = AutoModelForSequenceClassification.from_pretrained(
    	        "textattack/bert-base-uncased-ag-news"
    	    )
            tokenizer.model_max_length=128
            safer_model.load_state_dict(state.state_dict(), strict=False)
            safer_model.change_defense(defense_cls="random_noise",def_position="post_att_cls",noise_sigma=args.noise,defense=True)
            safer_model.to("cuda")
            safer_model.eval()
            BERT_SAFER = wrapping_model(safer_model,tokenizer,"safer",ensemble_num=args.ensemble_num,batch_size=args.ensemble_batch_size,safer_aug_set="model/weights/agnews/perturbation_constraint_pca0.8_100.pkl")

        elif args.defense == "advfooler":
            config = AutoConfig.from_pretrained("textattack/bert-base-uncased-ag-news")
            model = BertForSequenceClassification(config)
            state = AutoModelForSequenceClassification.from_pretrained(
    	        "textattack/bert-base-uncased-ag-news"
    	    )
            tokenizer.model_max_length=128
            model.load_state_dict(state.state_dict(), strict=False)
            model.change_defense(defense_cls="random_noise",def_position="post_att_cls",noise_sigma=args.noise,defense=True)
            model.to("cuda")
            model.eval()
            BERT_FOOLER = wrapping_model(model,tokenizer,"ensemble",ensemble_num=args.ensemble_num,batch_size=args.ensemble_batch_size)
    if args.model == "roberta":
        print("roberta")
        if args.defense == "mask":
            print("mask")
            mask_model = model_lib.TextDefense_model_builder("roberta","roberta-base","mask",device,dataset_name="agnews")
            load_path = "model/weights/tmd_ckpts/agnews/roberta_mask-len128-epo10-batch32-rate0.9-best.pth"
            tokenizer_roberta.model_max_length=128
            print(mask_model.load_state_dict(torch.load(load_path,map_location = device), strict=False))
            ROBERTA_MASK = wrapping_model(mask_model,tokenizer_roberta,"mask",ensemble_num=args.ensemble_num,batch_size=args.ensemble_batch_size,ran_mask=args.random_mask_rate,safer_aug_set=None,mask_token="<mask>")
        
        if args.defense == "safer":
            print("safer")
            safer_model = model_lib.TextDefense_model_builder("roberta","roberta-base","safer",device,dataset_name="agnews")
            load_path = "model/weights/tmd_ckpts/TextDefender/saved_models/agnews_roberta/safer-len128-epo10-batch32-best.pth"
            tokenizer_roberta.model_max_length=128
            print(safer_model.load_state_dict(torch.load(load_path,map_location = device), strict=False))
            ROBERTA_SAFER = wrapping_model(safer_model,tokenizer_roberta,"safer",ensemble_num=args.ensemble_num,batch_size=args.ensemble_batch_size,safer_aug_set="model/weights/tmd_ckpts/agnews/perturbation_constraint_pca0.8_100.pkl")
    
    with torch.no_grad():
        #noise_pos = {"pre_att_all": [0.2,0.3],"post_att_all": [0.2,0.3,0.4]}
        #noise_pos_roberta = {"post_att_all": [0.2,0.3]}
        
        noise_pos = {"pre_att_cls": [0.6,0.7],"post_att_cls": [0.8,0.9,1]}
        
        list_attacks = [args.attack_method]
        for i in range(0, 1):
            set_seed(i)
            dataset = gen_dataset(test_data)
            args.load_path = (
                f"noise_defense_attack_result/rebuttal_trueiter/AGNEWS/{i}/"
            )
            for attack_method in list_attacks:
                args.attack_method = attack_method
                if args.model == "bert":
                    if args.defense == "mask":
                        attack(args, BERT_MASK, f"BERT_MASK_{str(args.ensemble_num)}", dataset)
                
                    if args.defense == "safer":
                        attack(args, BERT_SAFER, f"BERT_SAFER_untrained_{str(args.ensemble_num)}", dataset)

                    if args.defense == "advfooler":
                        attack(args, BERT_FOOLER, "BERT_FOOLER", dataset)
                if args.model == "roberta":
                    if args.defense == "mask":
                        attack(args, ROBERTA_MASK, "ROBERTA_MASK", dataset)
                
                    if args.defense == "safer":
                        attack(args, ROBERTA_SAFER, "ROBERTA_SAFER", dataset)

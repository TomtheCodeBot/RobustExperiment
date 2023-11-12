from transformers import AutoTokenizer,AutoModelForSequenceClassification,AutoConfig
from model.BERTNoiseDefend import BertForSequenceClassification
from tqdm import tqdm
from textattack.models.wrappers import (
    ModelWrapper,
    PyTorchModelWrapper,
    SklearnModelWrapper,
    HuggingFaceModelWrapper,
)
import torch
from sklearn.metrics import accuracy_score
import copy
import pandas as pd
import numpy as np
import json
def extract_adversarial_examples(file,seperator=None):
    adv_samples = []
    if seperator == None:
        seperator = "---------------------------------------------"
    with open(file) as f:
        doc_lines = f.readlines()
    firstline = True
    skip = False
    first_lines = []
    label_list = []
    label = 0
    for line in doc_lines:
        if skip:
            label = int(line[2])
            skip=False
            continue
        if "Number of successful attacks" in line:
            break
        line = line.replace("\n","")
        if line == "":
            continue
        if seperator in line:
            firstline = True
            skip = True
            continue
        if firstline:
            first_line = line.replace("[[","").replace("]]","")
            firstline = False
            continue
        label_list.append(label)
        first_lines.append(first_line)
        adv_samples.append(line.replace("[[","").replace("]]",""))
    return adv_samples,label_list,first_lines

def get_result_from_run(file):
    with open(file) as f:
        doc_lines = f.readlines()[-9:-1]
        print(doc_lines)
    return {
        "successful_attacks":int(doc_lines[0].split(":")[-1].replace("\n","")),
        "failed_attacks":int(doc_lines[1].split(":")[-1].replace("\n","")),
        "skipped_attacks":int(doc_lines[2].split(":")[-1].replace("\n",""))
    }

def recalculate(curr_result,failed_attacks):
    curr_result["successful_attacks"] -= failed_attacks
    curr_result["failed_attacks"] += failed_attacks
    
    curr_result["AuA"] = curr_result["failed_attacks"] / 1000
    curr_result["ASR"] = curr_result["successful_attacks"] / (curr_result["successful_attacks"]+curr_result["failed_attacks"])
    return curr_result


def most_frequent(List):
    print(List.count(max(set(List), key = List.count)))
    return max(set(List), key = List.count)
if __name__ == "__main__":
    file = "/vinserver_user/duy.hc/RobustExperiment/noise_defense_attack_result/test_3/AGNEWS/0/hotflip/test_0.3-BERT.txt"
    batch =16
    iter = 5
    adv_samples,label_list,_ = extract_adversarial_examples(file)
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",use_fast=True)
    config = AutoConfig.from_pretrained("textattack/bert-base-uncased-ag-news")
    model = BertForSequenceClassification(config)
    state = AutoModelForSequenceClassification.from_pretrained(
        "textattack/bert-base-uncased-ag-news"
    )
    model.load_state_dict(state.state_dict(), strict=False)
    model.to("cuda")
    model.eval()
    model.change_defense(defense_cls="random_noise",def_position="post_att_cls",noise_sigma=0.8,defense=True)
    BERT = HuggingFaceModelWrapper(model, tokenizer)
    batch_num = len(adv_samples)//batch
    batch_num += 1 if len(adv_samples)%batch else 0
    batch_answer = []
    
    for _ in range(iter):
        y_pred_BERT = []
        for i in tqdm(range(0,batch_num)):
            y_pred_BERT.extend(torch.argmax(torch.tensor(BERT(adv_samples[i*batch:(i+1)*batch])),dim=-1).tolist())
        batch_answer.append(copy.deepcopy(y_pred_BERT))
    y_pred_BERT = []
    for i in range(len(batch_answer[0])):
        y_pred_BERT.append(most_frequent([batch_answer[x][i] for x in range(len(batch_answer))]))
    acc = accuracy_score(label_list, y_pred_BERT)
    print(f"AGNEWS: {acc*100:.2f}%")
    extra_failed_attacks = np.sum(np.array(label_list) == np.array(y_pred_BERT)).item()
    
    curr_result = get_result_from_run(file)
    print(curr_result["successful_attacks"]+curr_result["failed_attacks"]+curr_result["skipped_attacks"])
    print(json.dumps(recalculate(curr_result,extra_failed_attacks),sort_keys=True, indent=4))

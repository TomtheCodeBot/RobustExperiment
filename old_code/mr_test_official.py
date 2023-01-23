from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from model.robustNB import RobustNaiveBayesClassifierPercentage
from model.LSTMWrapper import OpenAttackMNB
from model.GRUWrapper import loadGRUModel,RNN
from utils.preprocessing import clean_text_imdb
from utils.dataloader import load_mr_data
from utils.attacking_platform import attack_platform
from model.BERTWrapper import OpenAttackBert

import OpenAttack as oa
import ssl
import numpy as np
from tqdm import tqdm
import pathlib
import json
import torch
import os
from utils.bert_vectorizer import BertVectorizer

os.environ["CUDA_VISIBLE_DEVICES"] = "1d"
import tensorflow as tf
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
    
def select_correct_samples(x,y_pred,y_true,sample_amount=1000,random_seed:int =0):
    np.random.seed(random_seed)
    y_pred = np.squeeze(np.asarray(y_pred))
    y_true = np.squeeze(np.asarray(y_true))
    correct_samples = x[y_pred==y_true]
    correct_labels = y_true[y_pred==y_true]
    sample_number = np.random.choice(len(correct_labels), sample_amount,replace=False)
    
    return correct_samples.iloc[sample_number],correct_labels[sample_number]
def attack_function_deepwordbug(victim,x,y_true,percentage_mod=0.15):
    json_res = {
            "Total Attacked Instances": len(x),
            "Successful Instances": 0,
            "Attack Success Rate": 0,
            "Avg. Victim Model Queries": 0,
            "Avg. Word Modif. Rate": 0
        }
    tokenizer = oa.text_process.tokenizer.PunctTokenizer()
    for i in tqdm(range(len(x))):
        attacker = oa.attackers.DeepWordBugAttacker(power = int(len(tokenizer.do_tokenize(x.iloc[i]))*percentage_mod))
        goal = oa.attack_assist.goal.ClassifierGoal(y_true[i],False)
        victim.set_context(x.iloc[i], None)
        res,amount = attacker.attack(victim,x.iloc[i],goal)
        if res is not None:
            orig = len(tokenizer.tokenize(x.iloc[i],pos_tagging=False))
            mod_rate = amount/orig
            if mod_rate<=percentage_mod:
                json_res["Avg. Word Modif. Rate"]+=mod_rate
                json_res["Successful Instances"]+=1
                json_res["Avg. Victim Model Queries"]+=victim.context.invoke
    json_res["Avg. Word Modif. Rate"] = json_res["Avg. Word Modif. Rate"]/json_res["Successful Instances"]
    json_res["Avg. Victim Model Queries"] = json_res["Avg. Victim Model Queries"]/json_res["Successful Instances"]
    return json_res
def attack_function_others(victim,attacker,x,y_true,percentage_mod=0.2):
    json_res = {
            "Total Attacked Instances": len(x),
            "Successful Instances": 0,
            "Attack Success Rate": 0,
            "Avg. Victim Model Queries": 0,
            "Avg. Word Modif. Rate": 0
        }
    tokenizer = oa.text_process.tokenizer.PunctTokenizer()
    for i in tqdm(range(len(x))):
        goal = oa.attack_assist.goal.ClassifierGoal(y_true[i],False)
        victim.set_context(x.iloc[i], None)
        res,amount = attacker.attack(victim,x.iloc[i],goal)
        if res is not None:
            orig = len(tokenizer.tokenize(x.iloc[i],pos_tagging=False))
            mod_rate = amount/orig
            if mod_rate<=percentage_mod:
                json_res["Avg. Word Modif. Rate"]+=mod_rate
                json_res["Successful Instances"]+=1
                json_res["Avg. Victim Model Queries"]+=victim.context.invoke
    json_res["Avg. Word Modif. Rate"] = json_res["Avg. Word Modif. Rate"]/json_res["Successful Instances"]
    json_res["Avg. Victim Model Queries"] = json_res["Avg. Victim Model Queries"]/json_res["Successful Instances"]
    return json_res

def result_output(result,dataset_name,attacker_name,victim_name):
    pathlib.Path(f'/home/ubuntu/Robustness_Gym/result_official/{dataset_name}/{attacker_name}').mkdir(parents=True, exist_ok=True) 

    with open(f"/home/ubuntu/Robustness_Gym/result_official/{dataset_name}/{attacker_name}/{victim_name}.json", "w") as outfile:
        json.dump(result, outfile)

def run_test(victim,attacker,y_pred,input_test,input_label,dataset_name,attacker_name,victim_name,test_time=3,sample_amount=1000):
    json_res = {
            "Total Attacked Instances": 0,
            "Successful Instances": 0,
            "Attack Success Rate": 0,
            "Avg. Victim Model Queries": 0,
            "Avg. Word Modif. Rate": 0
        }
    for i in range(test_time):
        x_test,y_test = select_correct_samples(input_test,y_pred,input_label,sample_amount,i)
        result = attack_function_others(victim,attacker,x_test,y_test)
        for k in json_res.keys():
            json_res[k]+=result[k]
    for k in json_res.keys():
        json_res[k]/=test_time
    json_res["Attack Success Rate"] = json_res["Successful Instances"]/json_res["Total Attacked Instances"]
    result_output(json_res,dataset_name,attacker_name,victim_name)
    print(json_res)
    return

def run_test_deepwordbug(victim,attacker,y_pred,input_test,input_label,dataset_name,attacker_name,victim_name,test_time=3,sample_amount=1000):
    json_res = {
            "Total Attacked Instances": 0,
            "Successful Instances": 0,
            "Attack Success Rate": 0,
            "Avg. Victim Model Queries": 0,
            "Avg. Word Modif. Rate": 0
        }
    for i in range(test_time):
        x_test,y_test = select_correct_samples(input_test,y_pred,input_label,sample_amount,i)
        result = attack_function_deepwordbug(victim,x_test,y_test)
        for k in json_res.keys():
            json_res[k]+=result[k]
    for k in json_res.keys():
        json_res[k]/=test_time
    json_res["Attack Success Rate"] = json_res["Successful Instances"]/json_res["Total Attacked Instances"]
    result_output(json_res,dataset_name,attacker_name,victim_name)
    print(json_res)
    return 
def dataset_mapping(x):
    return {
        "x": x["text"],
        "y": 1 if x["label"] > 0.5 else 0,
    }

if __name__ == "__main__":
    train_data , test_data = load_mr_data()
    vectorizer = CountVectorizer(stop_words="english",
                                preprocessor=clean_text_imdb, 
                                min_df=0)

    training_features = vectorizer.fit_transform(train_data["text"])
    training_labels = train_data["label"]
    test_features = vectorizer.transform(test_data["text"])
    test_labels = test_data["label"]

    RNB = RobustNaiveBayesClassifierPercentage(100)
    RNB.fit(training_features, training_labels)
    y_pred_RNB = RNB.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB)
    
    
    print("RNB Accuracy on the MR dataset: {:.2f}".format(acc*100))

    RNB_50 = RobustNaiveBayesClassifierPercentage(50)
    RNB_50.fit(training_features, training_labels)
    y_pred_RNB_50 = RNB_50.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_50)
    
    
    print("RNB_50 Accuracy on the MR dataset: {:.2f}".format(acc*100))

    RNB_25 = RobustNaiveBayesClassifierPercentage(25)
    RNB_25.fit(training_features, training_labels)
    y_pred_RNB_25 = RNB_25.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_25)
    
    
    print("RNB_25 Accuracy on the MR dataset: {:.2f}".format(acc*100))

    RNB_15 = RobustNaiveBayesClassifierPercentage(15)
    RNB_15.fit(training_features, training_labels)
    y_pred_RNB_15 = RNB_15.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_15)
    
    print("RNB_15 Accuracy on the MR dataset: {:.2f}".format(acc*100))
    
    RNB_5 = RobustNaiveBayesClassifierPercentage(5)
    RNB_5.fit(training_features, training_labels)
    y_pred_RNB_5 = RNB_5.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_5)
    
    print("RNB_5 Accuracy on the MR dataset: {:.2f}".format(acc*100))

    
    
    MNB = MultinomialNB(alpha=1.0)
    MNB.fit(training_features, training_labels)
    y_pred_MNB = MNB.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_MNB)

    print("MNB Accuracy on the MR dataset: {:.2f}".format(acc*100))

    LR = LogisticRegression(random_state=0)
    LR.fit(training_features, training_labels)
    y_pred_LR = LR.predict(test_features)
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_LR)

    print("LR Accuracy on the MR dataset: {:.2f}".format(acc*100))
    BERT = OpenAttackBert("/home/ubuntu/Robustness_Gym/model/weights/BERT/MR/ONNX")
    y_pred_BERT = BERT.get_pred(list(test_data["text"]))
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_BERT)
    print("BERT Accuracy on the MR dataset: {:.2f}".format(acc*100))
    """LSTM = loadLSTMModel("/home/ubuntu/Robustness_Gym/model/weights/LSTM/MR_WordTokenizer")
    y_pred_LSTM = LSTM.get_pred(list(test_data["text"]))
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_LSTM)
    
    print("LSTM Accuracy on the MR dataset: {:.2f}".format(acc*100))
    
    GRU = loadGRUModel("/home/ubuntu/Robustness_Gym/model/weights/GRU/MR_WordTokenizer",map_location=torch.device("cpu"))
    y_pred_GRU = GRU.get_pred(list(test_data["text"]))
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_GRU)
    
    print("GRU Accuracy on the GRU dataset: {:.2f}".format(acc*100))
    bert_vectorizer = BertVectorizer()
    training_features = bert_vectorizer.transform(train_data["text"].tolist())
    test_features = bert_vectorizer.transform(test_data["text"].tolist())
    RNB_BERT = RobustNaiveBayesClassifierPercentage(100)
    RNB_BERT.fit(training_features, training_labels)
    y_pred_RNB_BERT = RNB_BERT.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT)
    
    
    print("RNB_BERT Accuracy on the MR dataset: {:.2f}".format(acc*100))

    RNB_BERT_50 = RobustNaiveBayesClassifierPercentage(50)
    RNB_BERT_50.fit(training_features, training_labels)
    y_pred_RNB_BERT_50 = RNB_BERT_50.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT_50)
    
    
    print("RNB_BERT_50 Accuracy on the MR dataset: {:.2f}".format(acc*100))

    RNB_BERT_25 = RobustNaiveBayesClassifierPercentage(25)
    RNB_BERT_25.fit(training_features, training_labels)
    y_pred_RNB_BERT_25 = RNB_BERT_25.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT_25)
    
    
    print("RNB_BERT_25 Accuracy on the MR dataset: {:.2f}".format(acc*100))

    RNB_BERT_15 = RobustNaiveBayesClassifierPercentage(15)
    RNB_BERT_15.fit(training_features, training_labels)
    y_pred_RNB_BERT_15 = RNB_BERT_15.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT_15)
    
    print("RNB_BERT_15 Accuracy on the MR dataset: {:.2f}".format(acc*100))
    
    RNB_BERT_5 = RobustNaiveBayesClassifierPercentage(5)
    RNB_BERT_5.fit(training_features, training_labels)
    y_pred_RNB_BERT_5 = RNB_BERT_5.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT_5)
    
    print("RNB_BERT_5 Accuracy on the MR dataset: {:.2f}".format(acc*100))
    del training_features
    del test_features
    attacker = oa.attackers.TextBuggerAttacker()
    
    victim = OpenAttackMNB(RNB,vectorizer)
    run_test(victim,attacker,y_pred_RNB,test_data["text"],test_labels,"MR","TextBugger","RNB")
    
    victim = OpenAttackMNB(MNB,vectorizer)
    run_test(victim,attacker,y_pred_MNB,test_data["text"],test_labels,"MR","TextBugger","MNB")
    
    victim = OpenAttackMNB(LR,vectorizer)
    run_test(victim,attacker,y_pred_LR,test_data["text"],test_labels,"MR","TextBugger","LR")
    
    
    victim = OpenAttackMNB(RNB_50,vectorizer)
    run_test(victim,attacker,y_pred_RNB_50,test_data["text"],test_labels,"MR","TextBugger","RNB_50")
    
    victim = OpenAttackMNB(RNB_25,vectorizer)
    run_test(victim,attacker,y_pred_RNB_25,test_data["text"],test_labels,"MR","TextBugger","RNB_25")
    
    victim = OpenAttackMNB(RNB_15,vectorizer)
    run_test(victim,attacker,y_pred_RNB_15,test_data["text"],test_labels,"MR","TextBugger","RNB_15")
    
    victim = OpenAttackMNB(RNB_5,vectorizer)
    run_test(victim,attacker,y_pred_RNB_5,test_data["text"],test_labels,"MR","TextBugger","RNB_5")
    
    victim = LSTM
    run_test(victim,attacker,y_pred_LSTM,test_data["text"],test_labels,"MR","TextBugger","LSTM")
    attacker = oa.attackers.PWWSAttacker()
    victim = OpenAttackMNB(RNB,vectorizer)
     
    run_test(victim,attacker,y_pred_RNB,test_data["text"],test_labels,"MR","PWWS","RNB")
    
    victim = OpenAttackMNB(RNB_50,vectorizer)
    run_test(victim,attacker,y_pred_RNB_50,test_data["text"],test_labels,"MR","PWWS","RNB_50")
    
    victim = OpenAttackMNB(RNB_25,vectorizer)
    run_test(victim,attacker,y_pred_RNB_25,test_data["text"],test_labels,"MR","PWWS","RNB_25")
    
    victim = OpenAttackMNB(RNB_15,vectorizer)
    run_test(victim,attacker,y_pred_RNB_15,test_data["text"],test_labels,"MR","PWWS","RNB_15")
    
    victim = OpenAttackMNB(RNB_5,vectorizer)
    run_test(victim,attacker,y_pred_RNB_5,test_data["text"],test_labels,"MR","PWWS","RNB_5")
    victim = OpenAttackMNB(MNB,vectorizer)
    run_test(victim,attacker,y_pred_MNB,test_data["text"],test_labels,"MR","PWWS","MNB")
    
    victim = OpenAttackMNB(LR,vectorizer)
    run_test(victim,attacker,y_pred_LR,test_data["text"],test_labels,"MR","PWWS","LR")
    
    victim = LSTM
    run_test(victim,attacker,y_pred_LSTM,test_data["text"],test_labels,"MR","PWWS","LSTM")
    
    attacker = oa.attackers.TextFoolerAttacker()
    
    victim = OpenAttackMNB(RNB,vectorizer)
    run_test(victim,attacker,y_pred_RNB,test_data["text"],test_labels,"MR","TextFooler","RNB")
    
    victim = OpenAttackMNB(RNB_50,vectorizer)
    run_test(victim,attacker,y_pred_RNB_50,test_data["text"],test_labels,"MR","TextFooler","RNB_50")
    
    victim = OpenAttackMNB(RNB_25,vectorizer)
    run_test(victim,attacker,y_pred_RNB_25,test_data["text"],test_labels,"MR","TextFooler","RNB_25")
    
    victim = OpenAttackMNB(RNB_15,vectorizer)
    run_test(victim,attacker,y_pred_RNB_15,test_data["text"],test_labels,"MR","TextFooler","RNB_15")
    
    victim = OpenAttackMNB(RNB_5,vectorizer)
    run_test(victim,attacker,y_pred_RNB_5,test_data["text"],test_labels,"MR","TextFooler","RNB_5")
    victim = OpenAttackMNB(MNB,vectorizer)
    run_test(victim,attacker,y_pred_MNB,test_data["text"],test_labels,"MR","TextFooler","MNB")
    
    victim = OpenAttackMNB(LR,vectorizer)
    run_test(victim,attacker,y_pred_LR,test_data["text"],test_labels,"MR","TextFooler","LR")
    
    victim = LSTM
    run_test(victim,attacker,y_pred_LSTM,test_data["text"],test_labels,"MR","TextFooler","LSTM")
    
    victim = OpenAttackMNB(RNB,vectorizer)
    attacker = oa.attackers.DeepWordBugAttacker()
    run_test_deepwordbug(victim,attacker,y_pred_RNB,test_data["text"],test_labels,"MR","DeepWordBug","RNB")
    
    victim = OpenAttackMNB(RNB_50,vectorizer)
    run_test_deepwordbug(victim,attacker,y_pred_RNB_50,test_data["text"],test_labels,"MR","DeepWordBug","RNB_50")
    
    victim = OpenAttackMNB(RNB_25,vectorizer)
    run_test_deepwordbug(victim,attacker,y_pred_RNB_25,test_data["text"],test_labels,"MR","DeepWordBug","RNB_25")
    
    victim = OpenAttackMNB(RNB_15,vectorizer)
    run_test_deepwordbug(victim,attacker,y_pred_RNB_15,test_data["text"],test_labels,"MR","DeepWordBug","RNB_15")
    
    victim = OpenAttackMNB(RNB_5,vectorizer)
    run_test_deepwordbug(victim,attacker,y_pred_RNB_5,test_data["text"],test_labels,"MR","DeepWordBug","RNB_5")
    victim = OpenAttackMNB(MNB,vectorizer)
    run_test_deepwordbug(victim,attacker,y_pred_MNB,test_data["text"],test_labels,"MR","DeepWordBug","MNB")
    
    victim = OpenAttackMNB(LR,vectorizer)
    run_test_deepwordbug(victim,attacker,y_pred_LR,test_data["text"],test_labels,"MR","DeepWordBug","LR")
    
    victim = LSTM
    run_test_deepwordbug(victim,attacker,y_pred_LSTM,test_data["text"],test_labels,"MR","DeepWordBug","LSTM")
    
    attacker = oa.attackers.TextBuggerAttacker()
    victim = GRU
    run_test(victim,attacker,y_pred_GRU,test_data["text"],test_labels,"MR","TextBugger","GRU")
    run_test_deepwordbug(victim,"replaceone",y_pred_GRU,test_data["text"],test_labels,"MR","DeepWordBug","GRU")
    
    attacker = oa.attackers.TextFoolerAttacker()
    run_test(victim,attacker,y_pred_GRU,test_data["text"],test_labels,"MR","TextFooler","GRU")
    
    attacker = oa.attackers.PWWSAttacker()
    run_test(victim,attacker,y_pred_GRU,test_data["text"],test_labels,"MR","PWWS","GRU")
    
    attacker = oa.attackers.TextBuggerAttacker()
    victim = OpenAttackMNB(RNB_BERT,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT,test_data["text"],test_labels,"MR","TextBugger","RNB_BERT")
    
    victim = OpenAttackMNB(RNB_BERT_50,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT_50,test_data["text"],test_labels,"MR","TextBugger","RNB_BERT_50")
    
    victim = OpenAttackMNB(RNB_BERT_25,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT_25,test_data["text"],test_labels,"MR","TextBugger","RNB_BERT_25")
    
    victim = OpenAttackMNB(RNB_BERT_15,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT_15,test_data["text"],test_labels,"MR","TextBugger","RNB_BERT_15")
    
    victim = OpenAttackMNB(RNB_BERT_5,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT_5,test_data["text"],test_labels,"MR","TextBugger","RNB_BERT_5")
    
    victim = OpenAttackMNB(RNB_BERT,bert_vectorizer)
    run_test_deepwordbug(victim,"replaceone",y_pred_RNB_BERT,test_data["text"],test_labels,"MR","DeepWordBug","RNB_BERT")
    
    victim = OpenAttackMNB(RNB_BERT_50,bert_vectorizer)
    run_test_deepwordbug(victim,"replaceone",y_pred_RNB_BERT_50,test_data["text"],test_labels,"MR","DeepWordBug","RNB_BERT_50")
    
    victim = OpenAttackMNB(RNB_BERT_25,bert_vectorizer)
    run_test_deepwordbug(victim,"replaceone",y_pred_RNB_BERT_25,test_data["text"],test_labels,"MR","DeepWordBug","RNB_BERT_25")
    
    victim = OpenAttackMNB(RNB_BERT_15,bert_vectorizer)
    run_test_deepwordbug(victim,"replaceone",y_pred_RNB_BERT_15,test_data["text"],test_labels,"MR","DeepWordBug","RNB_BERT_15")
    
    victim = OpenAttackMNB(RNB_BERT_5,bert_vectorizer)
    run_test_deepwordbug(victim,"replaceone",y_pred_RNB_BERT_5,test_data["text"],test_labels,"MR","DeepWordBug","RNB_BERT_5")
    
    attacker = oa.attackers.TextFoolerAttacker()
    victim = OpenAttackMNB(RNB_BERT,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT,test_data["text"],test_labels,"MR","TextFooler","RNB_BERT")
    
    victim = OpenAttackMNB(RNB_BERT_50,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT_50,test_data["text"],test_labels,"MR","TextFooler","RNB_BERT_50")
    
    victim = OpenAttackMNB(RNB_BERT_25,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT_25,test_data["text"],test_labels,"MR","TextFooler","RNB_BERT_25")
    
    victim = OpenAttackMNB(RNB_BERT_15,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT_15,test_data["text"],test_labels,"MR","TextFooler","RNB_BERT_15")
    
    victim = OpenAttackMNB(RNB_BERT_5,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT_5,test_data["text"],test_labels,"MR","TextFooler","RNB_BERT_5")
    
    attacker = oa.attackers.PWWSAttacker()
    victim = OpenAttackMNB(RNB_BERT,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT,test_data["text"],test_labels,"MR","PWWS","RNB_BERT")
    
    victim = OpenAttackMNB(RNB_BERT_50,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT_50,test_data["text"],test_labels,"MR","PWWS","RNB_BERT_50")
    
    victim = OpenAttackMNB(RNB_BERT_25,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT_25,test_data["text"],test_labels,"MR","PWWS","RNB_BERT_25")
    
    victim = OpenAttackMNB(RNB_BERT_15,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT_15,test_data["text"],test_labels,"MR","PWWS","RNB_BERT_15")
    
    victim = OpenAttackMNB(RNB_BERT_5,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT_5,test_data["text"],test_labels,"MR","PWWS","RNB_BERT_5")"""
    
    attacker = oa.attackers.TextBuggerAttacker()
    victim = BERT
    run_test(victim,attacker,y_pred_BERT,test_data["text"],test_labels,"MR","TextBugger","BERT")
    run_test_deepwordbug(victim,"replaceone",y_pred_BERT,test_data["text"],test_labels,"MR","DeepWordBug","BERT")
    attacker = oa.attackers.TextFoolerAttacker()
    run_test(victim,attacker,y_pred_BERT,test_data["text"],test_labels,"MR","TextFooler","BERT")
    attacker = oa.attackers.PWWSAttacker()
    run_test(victim,attacker,y_pred_BERT,test_data["text"],test_labels,"MR","PWWS","BERT")
    
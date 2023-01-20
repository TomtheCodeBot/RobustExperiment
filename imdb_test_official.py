from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from model.robustNB import RobustNaiveBayesClassifierPercentage
from model.LSTMWrapper import OpenAttackMNB
from model.LSTMWrapper import loadLSTMModel
from model.BERTWrapper import OpenAttackBert
from model.SMARTBERTWrapper import OpenAttackSMARTBERT,OpenAttackSMARTBERTONNX
from model.GRUWrapper import loadGRUModel,RNN
from utils.preprocessing import clean_text_imdb
from utils.dataloader import load_train_test_imdb_data
from utils.attacking_platform import attack_platform
from utils.bert_vectorizer import BertVectorizer
import OpenAttack as oa
import ssl
import numpy as np
from tqdm import tqdm
import pathlib
import json
import pandas as pd
import torch
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
    
def select_correct_samples(x,y_pred,y_true,sample_amount=200,random_seed:int =0):
    np.random.seed(random_seed)
    y_pred = np.squeeze(np.asarray(y_pred))
    y_true = np.squeeze(np.asarray(y_true))
    correct_samples = x[y_pred==y_true]
    correct_labels = y_true[y_pred==y_true]
    sample_number = np.random.choice(len(correct_labels), sample_amount,replace=False)
    
    return correct_samples.iloc[sample_number],correct_labels[sample_number]
def attack_function_deepwordbug(victim,x,y_true,attack_mode = "replaceone",percentage_mod=0.15):
    json_res = {
            "Total Attacked Instances": len(x),
            "Successful Instances": 0,
            "Attack Success Rate": 0,
            "Avg. Victim Model Queries": 0,
            "Avg. Word Modif. Rate": 0
        }
    tokenizer = oa.text_process.tokenizer.PunctTokenizer()
    successful = []
    for i in tqdm(range(len(x))):
        attacker = oa.attackers.DeepWordBugAttacker(power = int(len(tokenizer.do_tokenize(x.iloc[i]))*percentage_mod),scoring=attack_mode,transform = "swap",)
        goal = oa.attack_assist.goal.ClassifierGoal(y_true[i],False)
        victim.set_context(x.iloc[i], None)
        res,amount = attacker.attack(victim,x.iloc[i],goal)
        if res is not None:
            org= tokenizer.tokenize(x.iloc[i],pos_tagging=False)
            orig = len(org)
            mod_rate = amount/orig
            if mod_rate<=percentage_mod:
                successful.append([" ".join(org),res])
                json_res["Avg. Word Modif. Rate"]+=mod_rate
                json_res["Successful Instances"]+=1
                json_res["Avg. Victim Model Queries"]+=victim.context.invoke
    panda_frame = pd.DataFrame(successful, columns=['orig', 'augmented'])
    panda_frame.to_csv("/home/ubuntu/Robustness_Gym/out.csv")
    print(json_res)
    if json_res["Successful Instances"]!=0:
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
    successful = []
    for i in tqdm(range(len(x))):
        goal = oa.attack_assist.goal.ClassifierGoal(y_true[i],False)
        victim.set_context(x.iloc[i], None)
        res,amount = attacker.attack(victim,x.iloc[i],goal)
        if res is not None:
            org= tokenizer.tokenize(x.iloc[i],pos_tagging=False)
            orig = len(org)
            mod_rate = amount/orig
            if mod_rate<=percentage_mod:
                successful.append([" ".join(org),res])
                json_res["Avg. Word Modif. Rate"]+=mod_rate
                json_res["Successful Instances"]+=1
                json_res["Avg. Victim Model Queries"]+=victim.context.invoke
    panda_frame = pd.DataFrame(successful, columns=['orig', 'augmented'])
    panda_frame.to_csv("/home/ubuntu/Robustness_Gym/out.csv")
    if json_res["Successful Instances"]!=0:
        json_res["Avg. Word Modif. Rate"] = json_res["Avg. Word Modif. Rate"]/json_res["Successful Instances"]
        json_res["Avg. Victim Model Queries"] = json_res["Avg. Victim Model Queries"]/json_res["Successful Instances"]
    return json_res

def result_output(result,dataset_name,attacker_name,victim_name):
    pathlib.Path(f'/home/ubuntu/Robustness_Gym/result_official/{dataset_name}/{attacker_name}').mkdir(parents=True, exist_ok=True) 

    with open(f"/home/ubuntu/Robustness_Gym/result_official/{dataset_name}/{attacker_name}/{victim_name}.json", "w") as outfile:
        json.dump(result, outfile)

def run_test(victim,attacker,y_pred,input_test,input_label,dataset_name,attacker_name,victim_name,test_time=3,sample_amount=200):
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

def run_test_deepwordbug(victim,attack_mode,y_pred,input_test,input_label,dataset_name,attacker_name,victim_name,test_time=3,sample_amount=200):
    json_res = {
            "Total Attacked Instances": 0,
            "Successful Instances": 0,
            "Attack Success Rate": 0,
            "Avg. Victim Model Queries": 0,
            "Avg. Word Modif. Rate": 0
        }
    for i in range(test_time):
        x_test,y_test = select_correct_samples(input_test,y_pred,input_label,sample_amount,i)
        result = attack_function_deepwordbug(victim,x_test,y_test,attack_mode)
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
    train_data , test_data = load_train_test_imdb_data("/home/ubuntu/Robustness_Gym/data/aclImdb")
    vectorizer = CountVectorizer(stop_words="english",
                                preprocessor=clean_text_imdb, 
                                min_df=0)
    training_features = vectorizer.fit_transform(train_data["text"])
    training_labels = train_data["label"]
    test_features = vectorizer.transform(test_data["text"])
    test_labels = test_data["label"]

    """RNB = RobustNaiveBayesClassifierPercentage(100)
    RNB.fit(training_features, training_labels)
    y_pred_RNB = RNB.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB)
    
    
    print("RNB Accuracy on the IMDB dataset: {:.2f}".format(acc*100))

    RNB_50 = RobustNaiveBayesClassifierPercentage(50)
    RNB_50.fit(training_features, training_labels)
    y_pred_RNB_50 = RNB_50.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_50)
    
    
    print("RNB_50 Accuracy on the IMDB dataset: {:.2f}".format(acc*100))

    RNB_25 = RobustNaiveBayesClassifierPercentage(25)
    RNB_25.fit(training_features, training_labels)
    y_pred_RNB_25 = RNB_25.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_25)
    
    
    print("RNB_25 Accuracy on the IMDB dataset: {:.2f}".format(acc*100))

    RNB_15 = RobustNaiveBayesClassifierPercentage(15)
    RNB_15.fit(training_features, training_labels)
    y_pred_RNB_15 = RNB_15.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_15)
    
    print("RNB_15 Accuracy on the IMDB dataset: {:.2f}".format(acc*100))
    
    RNB_5 = RobustNaiveBayesClassifierPercentage(5)
    RNB_5.fit(training_features, training_labels)
    y_pred_RNB_5 = RNB_5.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_5)
    
    print("RNB_5 Accuracy on the IMDB dataset: {:.2f}".format(acc*100))

    
    
    MNB = MultinomialNB(alpha=1.0)
    MNB.fit(training_features, training_labels)
    y_pred_MNB = MNB.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_MNB)

    print("MNB Accuracy on the IMDB dataset: {:.2f}".format(acc*100))

    LR = LogisticRegression(random_state=0)
    LR.fit(training_features, training_labels)
    y_pred_LR = LR.predict(test_features)
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_LR)

    print("LR Accuracy on the IMDB dataset: {:.2f}".format(acc*100))"""
    
    SMART_BERT = OpenAttackSMARTBERT("/home/ubuntu/mt-dnn/checkpoint_imdb/model_4.pt")
    y_pred_SMART_BERT = SMART_BERT.get_pred(list(test_data["text"]))
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_SMART_BERT)
    print("SMART_BERT Accuracy on the IMDB dataset: {:.2f}".format(acc*100))
    
    """BERT = OpenAttackBert("/home/ubuntu/Robustness_Gym/model/weights/BERT/IMDB/ONNX")
    y_pred_BERT = BERT.get_pred(list(test_data["text"]))
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_BERT)
    print("BERT Accuracy on the IMDB dataset: {:.2f}".format(acc*100))
    LSTM = loadLSTMModel("/home/ubuntu/Robustness_Gym/model/weights/LSTM/IMDB_WordTokenizer")
    y_pred_LSTM = LSTM.get_pred(list(test_data["text"]))
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_LSTM)
    
    print("LSTM Accuracy on the IMDB dataset: {:.2f}".format(acc*100))
    attacker = oa.attackers.TextBuggerAttacker()
    
    GRU = loadGRUModel("/home/ubuntu/Robustness_Gym/model/weights/GRU/IMDB_WordTokenizer",map_location=torch.device("cuda"))
    y_pred_GRU = GRU.get_pred(list(test_data["text"]))
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_GRU)
    
    print("GRU Accuracy on the IMDB dataset: {:.2f}".format(acc*100))
    bert_vectorizer = BertVectorizer()
    training_features = bert_vectorizer.transform(train_data["text"])
    training_labels = train_data["label"]
    test_features = bert_vectorizer.transform(test_data["text"])
    test_labels = test_data["label"]

    RNB_BERT = RobustNaiveBayesClassifierPercentage(100)
    RNB_BERT.fit(training_features, training_labels)
    y_pred_RNB_BERT = RNB_BERT.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT)
    
    
    print("RNB_BERT Accuracy on the IMDB dataset: {:.2f}".format(acc*100))

    RNB_BERT_50 = RobustNaiveBayesClassifierPercentage(50)
    RNB_BERT_50.fit(training_features, training_labels)
    y_pred_RNB_BERT_50 = RNB_BERT_50.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT_50)
    
    
    print("RNB_BERT_50 Accuracy on the IMDB dataset: {:.2f}".format(acc*100))

    RNB_BERT_25 = RobustNaiveBayesClassifierPercentage(25)
    RNB_BERT_25.fit(training_features, training_labels)
    y_pred_RNB_BERT_25 = RNB_BERT_25.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT_25)
    
    
    print("RNB_BERT_25 Accuracy on the IMDB dataset: {:.2f}".format(acc*100))

    RNB_BERT_15 = RobustNaiveBayesClassifierPercentage(15)
    RNB_BERT_15.fit(training_features, training_labels)
    y_pred_RNB_BERT_15 = RNB_BERT_15.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT_15)
    
    print("RNB_BERT_15 Accuracy on the IMDB dataset: {:.2f}".format(acc*100))
    
    RNB_BERT_5 = RobustNaiveBayesClassifierPercentage(5)
    RNB_BERT_5.fit(training_features, training_labels)
    y_pred_RNB_BERT_5 = RNB_BERT_5.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT_5)
    
    print("RNB_BERT_5 Accuracy on the IMDB dataset: {:.2f}".format(acc*100))
    del training_features
    del test_features
    attacker = oa.attackers.TextBuggerAttacker()
    victim = OpenAttackMNB(RNB,vectorizer)
    run_test(victim,attacker,y_pred_RNB,test_data["text"],test_labels,"IMDB","TextBugger","RNB")
    
    victim = OpenAttackMNB(RNB_50,vectorizer)
    run_test(victim,attacker,y_pred_RNB_50,test_data["text"],test_labels,"IMDB","TextBugger","RNB_50")
    
    victim = OpenAttackMNB(RNB_25,vectorizer)
    run_test(victim,attacker,y_pred_RNB_25,test_data["text"],test_labels,"IMDB","TextBugger","RNB_25")
    
    victim = OpenAttackMNB(RNB_15,vectorizer)
    run_test(victim,attacker,y_pred_RNB_15,test_data["text"],test_labels,"IMDB","TextBugger","RNB_15")
    
    victim = OpenAttackMNB(RNB_5,vectorizer)
    run_test(victim,attacker,y_pred_RNB_5,test_data["text"],test_labels,"IMDB","TextBugger","RNB_5")
    victim = OpenAttackMNB(MNB,vectorizer)
    run_test(victim,attacker,y_pred_MNB,test_data["text"],test_labels,"IMDB","TextBugger","MNB")
    
    victim = OpenAttackMNB(LR,vectorizer)
    run_test(victim,attacker,y_pred_LR,test_data["text"],test_labels,"IMDB","TextBugger","LR")
    
    victim = LSTM
    run_test(victim,attacker,y_pred_LSTM,test_data["text"],test_labels,"IMDB","TextBugger","LSTM")
    
    attacker = oa.attackers.TextBuggerAttacker()
    victim = GRU
    run_test(victim,attacker,y_pred_GRU,test_data["text"],test_labels,"IMDB","TextBugger","GRU")
    run_test_deepwordbug(victim,"replaceone",y_pred_GRU,test_data["text"],test_labels,"IMDB","DeepWordBug","GRU")
    attacker = oa.attackers.TextFoolerAttacker()
    run_test(victim,attacker,y_pred_GRU,test_data["text"],test_labels,"IMDB","TextFooler","GRU")
    attacker = oa.attackers.PWWSAttacker()
    run_test(victim,attacker,y_pred_GRU,test_data["text"],test_labels,"IMDB","PWWS","GRU")
    deep_word_list = ["replaceone"]
    for i in deep_word_list:
        victim = OpenAttackMNB(RNB,vectorizer)
        run_test_deepwordbug(victim,i,y_pred_RNB,test_data["text"],test_labels,"IMDB","DeepWordBug_"+i,"RNB")
        
        victim = OpenAttackMNB(RNB_50,vectorizer)
        run_test_deepwordbug(victim,i,y_pred_RNB_50,test_data["text"],test_labels,"IMDB","DeepWordBug_"+i,"RNB_50")
        
        victim = OpenAttackMNB(RNB_25,vectorizer)
        run_test_deepwordbug(victim,i,y_pred_RNB_25,test_data["text"],test_labels,"IMDB","DeepWordBug_"+i,"RNB_25")
        
        victim = OpenAttackMNB(RNB_15,vectorizer)
        run_test_deepwordbug(victim,i,y_pred_RNB_15,test_data["text"],test_labels,"IMDB","DeepWordBug_"+i,"RNB_15")
        
        victim = OpenAttackMNB(RNB_5,vectorizer)
        run_test_deepwordbug(victim,i,y_pred_RNB_5,test_data["text"],test_labels,"IMDB","DeepWordBug_"+i,"RNB_5")
        victim = OpenAttackMNB(MNB,vectorizer)
        run_test_deepwordbug(victim,i,y_pred_MNB,test_data["text"],test_labels,"IMDB","DeepWordBug_"+i,"MNB")
        
        victim = OpenAttackMNB(LR,vectorizer)
        run_test_deepwordbug(victim,i,y_pred_LR,test_data["text"],test_labels,"IMDB","DeepWordBug_"+i,"LR")
        
        victim = LSTM
        run_test_deepwordbug(victim,i,y_pred_LSTM,test_data["text"],test_labels,"IMDB","DeepWordBug_"+i,"LSTM")"""
        
    """attacker = oa.attackers.TextBuggerAttacker()
    victim = OpenAttackMNB(RNB_BERT,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT,test_data["text"],test_labels,"IMDB","TextBugger","RNB_BERT")
    
    victim = OpenAttackMNB(RNB_BERT_50,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT_50,test_data["text"],test_labels,"IMDB","TextBugger","RNB_BERT_50")
    
    victim = OpenAttackMNB(RNB_BERT_25,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT_25,test_data["text"],test_labels,"IMDB","TextBugger","RNB_BERT_25")
    
    victim = OpenAttackMNB(RNB_BERT_15,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT_15,test_data["text"],test_labels,"IMDB","TextBugger","RNB_BERT_15")
    
    victim = OpenAttackMNB(RNB_BERT_5,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT_5,test_data["text"],test_labels,"IMDB","TextBugger","RNB_BERT_5")
    
    victim = OpenAttackMNB(RNB_BERT,bert_vectorizer)
    run_test_deepwordbug(victim,"replaceone",y_pred_RNB_BERT,test_data["text"],test_labels,"IMDB","DeepWordBug","RNB_BERT")
    
    victim = OpenAttackMNB(RNB_BERT_50,bert_vectorizer)
    run_test_deepwordbug(victim,"replaceone",y_pred_RNB_BERT_50,test_data["text"],test_labels,"IMDB","DeepWordBug","RNB_BERT_50")
    
    victim = OpenAttackMNB(RNB_BERT_25,bert_vectorizer)
    run_test_deepwordbug(victim,"replaceone",y_pred_RNB_BERT_25,test_data["text"],test_labels,"IMDB","DeepWordBug","RNB_BERT_25")
    
    victim = OpenAttackMNB(RNB_BERT_15,bert_vectorizer)
    run_test_deepwordbug(victim,"replaceone",y_pred_RNB_BERT_15,test_data["text"],test_labels,"IMDB","DeepWordBug","RNB_BERT_15")
    
    victim = OpenAttackMNB(RNB_BERT_5,bert_vectorizer)
    run_test_deepwordbug(victim,"replaceone",y_pred_RNB_BERT_5,test_data["text"],test_labels,"IMDB","DeepWordBug","RNB_BERT_5")
    
    attacker = oa.attackers.TextFoolerAttacker()
    victim = OpenAttackMNB(RNB_BERT,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT,test_data["text"],test_labels,"IMDB","TextFooler","RNB_BERT")
    
    attacker = oa.attackers.TextFoolerAttacker()
    victim = OpenAttackMNB(RNB_BERT_50,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT_50,test_data["text"],test_labels,"IMDB","TextFooler","RNB_BERT_50")
    
    victim = OpenAttackMNB(RNB_BERT_25,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT_25,test_data["text"],test_labels,"IMDB","TextFooler","RNB_BERT_25")
    
    victim = OpenAttackMNB(RNB_BERT_15,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT_15,test_data["text"],test_labels,"IMDB","TextFooler","RNB_BERT_15")
    
    victim = OpenAttackMNB(RNB_BERT_5,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT_5,test_data["text"],test_labels,"IMDB","TextFooler","RNB_BERT_5")
    
    attacker = oa.attackers.PWWSAttacker()
    victim = OpenAttackMNB(RNB_BERT,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT,test_data["text"],test_labels,"IMDB","PWWS","RNB_BERT")
    
    victim = OpenAttackMNB(RNB_BERT_50,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT_50,test_data["text"],test_labels,"IMDB","PWWS","RNB_BERT_50")
    
    victim = OpenAttackMNB(RNB_BERT_25,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT_25,test_data["text"],test_labels,"IMDB","PWWS","RNB_BERT_25")
    
    victim = OpenAttackMNB(RNB_BERT_15,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT_15,test_data["text"],test_labels,"IMDB","PWWS","RNB_BERT_15")
    
    victim = OpenAttackMNB(RNB_BERT_5,bert_vectorizer)
    run_test(victim,attacker,y_pred_RNB_BERT_5,test_data["text"],test_labels,"IMDB","PWWS","RNB_BERT_5")
    
    
    attacker = oa.attackers.TextBuggerAttacker()
    victim = BERT
    run_test(victim,attacker,y_pred_BERT,test_data["text"],test_labels,"IMDB","TextBugger","BERT")
    run_test_deepwordbug(victim,"replaceone",y_pred_BERT,test_data["text"],test_labels,"IMDB","DeepWordBug","BERT")
    attacker = oa.attackers.TextFoolerAttacker()
    run_test(victim,attacker,y_pred_BERT,test_data["text"],test_labels,"IMDB","TextFooler","BERT")
    attacker = oa.attackers.PWWSAttacker()
    run_test(victim,attacker,y_pred_BERT,test_data["text"],test_labels,"IMDB","PWWS","BERT")"""
    
    attacker = oa.attackers.TextBuggerAttacker()
    victim = SMART_BERT
    """run_test(victim,attacker,y_pred_SMART_BERT,test_data["text"],test_labels,"IMDB","TextBugger","SMART_BERT")
    run_test_deepwordbug(victim,"replaceone",y_pred_SMART_BERT,test_data["text"],test_labels,"IMDB","DeepWordBug","SMART_BERT")
    attacker = oa.attackers.TextFoolerAttacker()
    run_test(victim,attacker,y_pred_SMART_BERT,test_data["text"],test_labels,"IMDB","TextFooler","SMART_BERT")"""
    attacker = oa.attackers.PWWSAttacker()
    run_test(victim,attacker,y_pred_SMART_BERT,test_data["text"],test_labels,"IMDB","PWWS","SMART_BERT")
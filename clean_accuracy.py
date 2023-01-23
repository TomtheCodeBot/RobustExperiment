from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from model.robustNB import RobustNaiveBayesClassifierPercentage,RobustNaiveBayesMultiClassifierPercentage
from model.LSTMWrapper import OpenAttackMNB
from model.BERTWrapper import OpenAttackBert
from utils.preprocessing import clean_text_imdb
from utils.dataloader import load_train_test_imdb_data,load_mr_data
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
import datasets
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
    

if __name__ == "__main__":
    train_data , test_data = load_train_test_imdb_data("/home/ubuntu/Robustness_Gym/data/aclImdb")
    vectorizer = CountVectorizer(stop_words="english",
                                preprocessor=clean_text_imdb, 
                                min_df=0)

    acc_dict = {}
    training_features = vectorizer.fit_transform(train_data["text"])
    training_labels = train_data["label"]
    test_features = vectorizer.transform(test_data["text"])
    test_labels = test_data["label"]

    RNB = RobustNaiveBayesClassifierPercentage(100)
    RNB.fit(training_features, training_labels)
    y_pred_RNB = RNB.predict(test_features)
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB)
    acc_dict["RNB"]="{:.2f}%".format(acc*100)
    
    print("RNB Accuracy on the IMDB dataset: {:.2f}".format(acc*100))

    RNB_50 = RobustNaiveBayesClassifierPercentage(50)
    RNB_50.fit(training_features, training_labels)
    y_pred_RNB_50 = RNB_50.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_50)
    acc_dict["RNB_50"]="{:.2f}%".format(acc*100)
    
    print("RNB_50 Accuracy on the IMDB dataset: {:.2f}".format(acc*100))

    RNB_25 = RobustNaiveBayesClassifierPercentage(25)
    RNB_25.fit(training_features, training_labels)
    y_pred_RNB_25 = RNB_25.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_25)
    acc_dict["RNB_25"]="{:.2f}%".format(acc*100)
    
    print("RNB_25 Accuracy on the IMDB dataset: {:.2f}".format(acc*100))

    RNB_15 = RobustNaiveBayesClassifierPercentage(15)
    RNB_15.fit(training_features, training_labels)
    y_pred_RNB_15 = RNB_15.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_15)
    acc_dict["RNB_15"]="{:.2f}%".format(acc*100)
    print("RNB_15 Accuracy on the IMDB dataset: {:.2f}".format(acc*100))
    
    RNB_5 = RobustNaiveBayesClassifierPercentage(5)
    RNB_5.fit(training_features, training_labels)
    y_pred_RNB_5 = RNB_5.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_5)
    acc_dict["RNB_5"]="{:.2f}%".format(acc*100)
    print("RNB_5 Accuracy on the IMDB dataset: {:.2f}".format(acc*100))

    
    
    MNB = MultinomialNB(alpha=1.0)
    MNB.fit(training_features, training_labels)
    y_pred_MNB = MNB.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_MNB)
    acc_dict["MNB"]="{:.2f}%".format(acc*100)
    print("MNB Accuracy on the IMDB dataset: {:.2f}".format(acc*100))

    LR = LogisticRegression(random_state=0)
    LR.fit(training_features, training_labels)
    y_pred_LR = LR.predict(test_features)
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_LR)
    acc_dict["LR"]="{:.2f}%".format(acc*100)
    print("LR Accuracy on the IMDB dataset: {:.2f}".format(acc*100))
    

    BERT = OpenAttackBert("/home/ubuntu/Robustness_Gym/model/weights/BERT/IMDB/ONNX")
    y_pred_BERT = BERT.get_pred(list(test_data["text"]))
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_BERT)
    acc_dict["BERT"]="{:.2f}%".format(acc*100)
    print("BERT Accuracy on the IMDB dataset: {:.2f}".format(acc*100))
    
    from model.LSTMWrapper import loadLSTMModel,RNN
    LSTM = loadLSTMModel("/home/ubuntu/Robustness_Gym/model/weights/LSTM/IMDB_WordTokenizer")
    y_pred_LSTM = LSTM.get_pred(list(test_data["text"]))
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_LSTM)
    acc_dict["LSTM"]="{:.2f}%".format(acc*100)
    print("LSTM Accuracy on the IMDB dataset: {:.2f}".format(acc*100))
    attacker = oa.attackers.TextBuggerAttacker()
    from model.GRUWrapper import loadGRUModel,RNN
    GRU = loadGRUModel("/home/ubuntu/Robustness_Gym/model/weights/GRU/IMDB_WordTokenizer")
    y_pred_GRU = GRU.get_pred(list(test_data["text"]))
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_GRU)
    acc_dict["GRU"]="{:.2f}%".format(acc*100)
    
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
    acc_dict["RNB_BERT"]="{:.2f}%".format(acc*100)
    
    print("RNB_BERT Accuracy on the IMDB dataset: {:.2f}".format(acc*100))

    RNB_BERT_50 = RobustNaiveBayesClassifierPercentage(50)
    RNB_BERT_50.fit(training_features, training_labels)
    y_pred_RNB_BERT_50 = RNB_BERT_50.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT_50)
    acc_dict["RNB_BERT_50"]="{:.2f}%".format(acc*100)
    
    print("RNB_BERT_50 Accuracy on the IMDB dataset: {:.2f}".format(acc*100))

    RNB_BERT_25 = RobustNaiveBayesClassifierPercentage(25)
    RNB_BERT_25.fit(training_features, training_labels)
    y_pred_RNB_BERT_25 = RNB_BERT_25.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT_25)
    acc_dict["RNB_BERT_25"]="{:.2f}%".format(acc*100)
    
    print("RNB_BERT_25 Accuracy on the IMDB dataset: {:.2f}".format(acc*100))

    RNB_BERT_15 = RobustNaiveBayesClassifierPercentage(15)
    RNB_BERT_15.fit(training_features, training_labels)
    y_pred_RNB_BERT_15 = RNB_BERT_15.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT_15)
    acc_dict["RNB_BERT_15"]="{:.2f}%".format(acc*100)
    print("RNB_BERT_15 Accuracy on the IMDB dataset: {:.2f}".format(acc*100))
    
    RNB_BERT_5 = RobustNaiveBayesClassifierPercentage(5)
    RNB_BERT_5.fit(training_features, training_labels)
    y_pred_RNB_BERT_5 = RNB_BERT_5.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT_5)
    acc_dict["RNB_BERT_5"]="{:.2f}%".format(acc*100)
    print("RNB_BERT_5 Accuracy on the IMDB dataset: {:.2f}".format(acc*100))
    del training_features
    del test_features
    
    import json
    with open('/home/ubuntu/Robustness_Gym/result_official/IMDB/result.json', 'w') as f:
        json.dump(acc_dict, f)
        
#############################################################################################
    vectorizer = CountVectorizer(stop_words="english",
                                preprocessor=clean_text_imdb, 
                                min_df=0)
    
    sst2_dataset = datasets.load_dataset("SetFit/sst2")
    train_data = sst2_dataset["train"]
    test_data = sst2_dataset["test"]
    training_features = vectorizer.fit_transform(train_data["text"])
    training_labels = np.array(train_data["label"])
    test_features = vectorizer.transform(test_data["text"])
    test_labels = np.array(test_data["label"])
    acc_dict = {}
    RNB = RobustNaiveBayesClassifierPercentage(100)
    RNB.fit(training_features, training_labels)
    y_pred_RNB = RNB.predict(test_features)
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB)
    acc_dict["RNB"]="{:.2f}%".format(acc*100)
    
    print("RNB Accuracy on the SST2 dataset: {:.2f}".format(acc*100))

    RNB_50 = RobustNaiveBayesClassifierPercentage(50)
    RNB_50.fit(training_features, training_labels)
    y_pred_RNB_50 = RNB_50.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_50)
    acc_dict["RNB_50"]="{:.2f}%".format(acc*100)
    
    print("RNB_50 Accuracy on the SST2 dataset: {:.2f}".format(acc*100))

    RNB_25 = RobustNaiveBayesClassifierPercentage(25)
    RNB_25.fit(training_features, training_labels)
    y_pred_RNB_25 = RNB_25.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_25)
    acc_dict["RNB_25"]="{:.2f}%".format(acc*100)
    
    print("RNB_25 Accuracy on the SST2 dataset: {:.2f}".format(acc*100))

    RNB_15 = RobustNaiveBayesClassifierPercentage(15)
    RNB_15.fit(training_features, training_labels)
    y_pred_RNB_15 = RNB_15.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_15)
    acc_dict["RNB_15"]="{:.2f}%".format(acc*100)
    print("RNB_15 Accuracy on the SST2 dataset: {:.2f}".format(acc*100))
    
    RNB_5 = RobustNaiveBayesClassifierPercentage(5)
    RNB_5.fit(training_features, training_labels)
    y_pred_RNB_5 = RNB_5.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_5)
    acc_dict["RNB_5"]="{:.2f}%".format(acc*100)
    print("RNB_5 Accuracy on the SST2 dataset: {:.2f}".format(acc*100))

    
    
    MNB = MultinomialNB(alpha=1.0)
    MNB.fit(training_features, training_labels)
    y_pred_MNB = MNB.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_MNB)
    acc_dict["MNB"]="{:.2f}%".format(acc*100)
    print("MNB Accuracy on the SST2 dataset: {:.2f}".format(acc*100))

    LR = LogisticRegression(random_state=0)
    LR.fit(training_features, training_labels)
    y_pred_LR = LR.predict(test_features)
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_LR)
    acc_dict["LR"]="{:.2f}%".format(acc*100)
    print("LR Accuracy on the SST2 dataset: {:.2f}".format(acc*100))
    BERT = OpenAttackBert("/home/ubuntu/Robustness_Gym/model/weights/BERT/SST2/ONNX")
    y_pred_BERT = BERT.get_pred(list(test_data["text"]))
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_BERT)
    acc_dict["BERT"]="{:.2f}%".format(acc*100)
    print("BERT Accuracy on the SST2 dataset: {:.2f}".format(acc*100))
    
    from model.LSTMWrapper import loadLSTMModel,RNN

    LSTM = loadLSTMModel("/home/ubuntu/Robustness_Gym/model/weights/LSTM/SST2_WordTokenizer")
    y_pred_LSTM = LSTM.get_pred(list(test_data["text"]))
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_LSTM)
    acc_dict["LSTM"]="{:.2f}%".format(acc*100)
    print("LSTM Accuracy on the SST2 dataset: {:.2f}".format(acc*100))
    attacker = oa.attackers.TextBuggerAttacker()
    from model.GRUWrapper import loadGRUModel,RNN
    GRU = loadGRUModel("/home/ubuntu/Robustness_Gym/model/weights/GRU/SST2_WordTokenizer")
    y_pred_GRU = GRU.get_pred(list(test_data["text"]))
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_GRU)
    acc_dict["GRU"]="{:.2f}%".format(acc*100)
    
    print("GRU Accuracy on the SST2 dataset: {:.2f}".format(acc*100))
    
    bert_vectorizer = BertVectorizer()
    training_features = bert_vectorizer.transform(train_data["text"])
    test_features = bert_vectorizer.transform(test_data["text"])

    RNB_BERT = RobustNaiveBayesClassifierPercentage(100)
    RNB_BERT.fit(training_features, training_labels)
    y_pred_RNB_BERT = RNB_BERT.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT)
    acc_dict["RNB_BERT"]="{:.2f}%".format(acc*100)
    
    print("RNB_BERT Accuracy on the SST2 dataset: {:.2f}".format(acc*100))

    RNB_BERT_50 = RobustNaiveBayesClassifierPercentage(50)
    RNB_BERT_50.fit(training_features, training_labels)
    y_pred_RNB_BERT_50 = RNB_BERT_50.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT_50)
    acc_dict["RNB_BERT_50"]="{:.2f}%".format(acc*100)
    
    print("RNB_BERT_50 Accuracy on the SST2 dataset: {:.2f}".format(acc*100))

    RNB_BERT_25 = RobustNaiveBayesClassifierPercentage(25)
    RNB_BERT_25.fit(training_features, training_labels)
    y_pred_RNB_BERT_25 = RNB_BERT_25.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT_25)
    acc_dict["RNB_BERT_25"]="{:.2f}%".format(acc*100)
    
    print("RNB_BERT_25 Accuracy on the SST2 dataset: {:.2f}".format(acc*100))

    RNB_BERT_15 = RobustNaiveBayesClassifierPercentage(15)
    RNB_BERT_15.fit(training_features, training_labels)
    y_pred_RNB_BERT_15 = RNB_BERT_15.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT_15)
    acc_dict["RNB_BERT_15"]="{:.2f}%".format(acc*100)
    print("RNB_BERT_15 Accuracy on the SST2 dataset: {:.2f}".format(acc*100))
    
    RNB_BERT_5 = RobustNaiveBayesClassifierPercentage(5)
    RNB_BERT_5.fit(training_features, training_labels)
    y_pred_RNB_BERT_5 = RNB_BERT_5.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT_5)
    acc_dict["RNB_BERT_5"]="{:.2f}%".format(acc*100)
    print("RNB_BERT_5 Accuracy on the SST2 dataset: {:.2f}".format(acc*100))
    del training_features
    del test_features
    
    import json
    with open('/home/ubuntu/Robustness_Gym/result_official/SST2/result.json', 'w') as f:
        json.dump(acc_dict, f)

#############################################################################################
    train_data , test_data = load_mr_data()
    vectorizer = CountVectorizer(stop_words="english",
                                preprocessor=clean_text_imdb, 
                                min_df=0)

    training_features = vectorizer.fit_transform(train_data["text"])
    training_labels = train_data["label"]
    test_features = vectorizer.transform(test_data["text"])
    test_labels = test_data["label"]
    acc_dict={}
    RNB = RobustNaiveBayesClassifierPercentage(100)
    RNB.fit(training_features, training_labels)
    y_pred_RNB = RNB.predict(test_features)
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB)
    acc_dict["RNB"]="{:.2f}%".format(acc*100)
    
    print("RNB Accuracy on the MR dataset: {:.2f}".format(acc*100))

    RNB_50 = RobustNaiveBayesClassifierPercentage(50)
    RNB_50.fit(training_features, training_labels)
    y_pred_RNB_50 = RNB_50.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_50)
    acc_dict["RNB_50"]="{:.2f}%".format(acc*100)
    
    print("RNB_50 Accuracy on the MR dataset: {:.2f}".format(acc*100))

    RNB_25 = RobustNaiveBayesClassifierPercentage(25)
    RNB_25.fit(training_features, training_labels)
    y_pred_RNB_25 = RNB_25.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_25)
    acc_dict["RNB_25"]="{:.2f}%".format(acc*100)
    
    print("RNB_25 Accuracy on the MR dataset: {:.2f}".format(acc*100))

    RNB_15 = RobustNaiveBayesClassifierPercentage(15)
    RNB_15.fit(training_features, training_labels)
    y_pred_RNB_15 = RNB_15.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_15)
    acc_dict["RNB_15"]="{:.2f}%".format(acc*100)
    print("RNB_15 Accuracy on the MR dataset: {:.2f}".format(acc*100))
    
    RNB_5 = RobustNaiveBayesClassifierPercentage(5)
    RNB_5.fit(training_features, training_labels)
    y_pred_RNB_5 = RNB_5.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_5)
    acc_dict["RNB_5"]="{:.2f}%".format(acc*100)
    print("RNB_5 Accuracy on the MR dataset: {:.2f}".format(acc*100))

    
    
    MNB = MultinomialNB(alpha=1.0)
    MNB.fit(training_features, training_labels)
    y_pred_MNB = MNB.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_MNB)
    acc_dict["MNB"]="{:.2f}%".format(acc*100)
    print("MNB Accuracy on the MR dataset: {:.2f}".format(acc*100))

    LR = LogisticRegression(random_state=0)
    LR.fit(training_features, training_labels)
    y_pred_LR = LR.predict(test_features)
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_LR)
    acc_dict["LR"]="{:.2f}%".format(acc*100)
    print("LR Accuracy on the MR dataset: {:.2f}".format(acc*100))
    
    BERT = OpenAttackBert("/home/ubuntu/Robustness_Gym/model/weights/BERT/MR/ONNX")
    y_pred_BERT = BERT.get_pred(list(test_data["text"]))
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_BERT)
    acc_dict["BERT"]="{:.2f}%".format(acc*100)
    print("BERT Accuracy on the MR dataset: {:.2f}".format(acc*100))
    from model.LSTMWrapper import loadLSTMModel,RNN

    LSTM = loadLSTMModel("/home/ubuntu/Robustness_Gym/model/weights/LSTM/MR_WordTokenizer")
    y_pred_LSTM = LSTM.get_pred(list(test_data["text"]))
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_LSTM)
    acc_dict["LSTM"]="{:.2f}%".format(acc*100)
    print("LSTM Accuracy on the MR dataset: {:.2f}".format(acc*100))
    attacker = oa.attackers.TextBuggerAttacker()
    from model.GRUWrapper import loadGRUModel,RNN
    GRU = loadGRUModel("/home/ubuntu/Robustness_Gym/model/weights/GRU/MR_WordTokenizer")
    y_pred_GRU = GRU.get_pred(list(test_data["text"]))
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_GRU)
    acc_dict["GRU"]="{:.2f}%".format(acc*100)
    
    print("GRU Accuracy on the MR dataset: {:.2f}".format(acc*100))
    
    bert_vectorizer = BertVectorizer()
    training_features = bert_vectorizer.transform(train_data["text"].to_list())
    test_features = bert_vectorizer.transform(test_data["text"].to_list())

    RNB_BERT = RobustNaiveBayesClassifierPercentage(100)
    RNB_BERT.fit(training_features, training_labels)
    y_pred_RNB_BERT = RNB_BERT.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT)
    acc_dict["RNB_BERT"]="{:.2f}%".format(acc*100)
    
    print("RNB_BERT Accuracy on the MR dataset: {:.2f}".format(acc*100))

    RNB_BERT_50 = RobustNaiveBayesClassifierPercentage(50)
    RNB_BERT_50.fit(training_features, training_labels)
    y_pred_RNB_BERT_50 = RNB_BERT_50.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT_50)
    acc_dict["RNB_BERT_50"]="{:.2f}%".format(acc*100)
    
    print("RNB_BERT_50 Accuracy on the MR dataset: {:.2f}".format(acc*100))

    RNB_BERT_25 = RobustNaiveBayesClassifierPercentage(25)
    RNB_BERT_25.fit(training_features, training_labels)
    y_pred_RNB_BERT_25 = RNB_BERT_25.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT_25)
    acc_dict["RNB_BERT_25"]="{:.2f}%".format(acc*100)
    
    print("RNB_BERT_25 Accuracy on the MR dataset: {:.2f}".format(acc*100))

    RNB_BERT_15 = RobustNaiveBayesClassifierPercentage(15)
    RNB_BERT_15.fit(training_features, training_labels)
    y_pred_RNB_BERT_15 = RNB_BERT_15.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT_15)
    acc_dict["RNB_BERT_15"]="{:.2f}%".format(acc*100)
    print("RNB_BERT_15 Accuracy on the MR dataset: {:.2f}".format(acc*100))
    
    RNB_BERT_5 = RobustNaiveBayesClassifierPercentage(5)
    RNB_BERT_5.fit(training_features, training_labels)
    y_pred_RNB_BERT_5 = RNB_BERT_5.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT_5)
    acc_dict["RNB_BERT_5"]="{:.2f}%".format(acc*100)
    print("RNB_BERT_5 Accuracy on the MR dataset: {:.2f}".format(acc*100))
    del training_features
    del test_features
    
    import json
    with open('/home/ubuntu/Robustness_Gym/result_official/MR/result.json', 'w') as f:
        json.dump(acc_dict, f)

#############################################################################################
    vectorizer = CountVectorizer(stop_words="english",
                                preprocessor=clean_text_imdb, 
                                min_df=0)
    
    sst2_dataset = datasets.load_dataset("ag_news")
    train_data = sst2_dataset["train"]
    test_data = sst2_dataset["test"]
    training_features = vectorizer.fit_transform(train_data["text"])
    training_labels = np.array(train_data["label"])
    test_features = vectorizer.transform(test_data["text"])
    test_labels = np.array(test_data["label"])
    acc_dict={}
    RNB = RobustNaiveBayesMultiClassifierPercentage(100,num_classes=4)
    RNB.fit(training_features, training_labels)
    y_pred_RNB = RNB.predict(test_features)
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB)
    acc_dict["RNB"]="{:.2f}%".format(acc*100)
    
    print("RNB Accuracy on the AGNEWS dataset: {:.2f}".format(acc*100))

    RNB_50 = RobustNaiveBayesMultiClassifierPercentage(50,num_classes=4)
    RNB_50.fit(training_features, training_labels)
    y_pred_RNB_50 = RNB_50.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_50)
    acc_dict["RNB_50"]="{:.2f}%".format(acc*100)
    
    print("RNB_50 Accuracy on the AGNEWS dataset: {:.2f}".format(acc*100))

    RNB_25 = RobustNaiveBayesMultiClassifierPercentage(25,num_classes=4)
    RNB_25.fit(training_features, training_labels)
    y_pred_RNB_25 = RNB_25.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_25)
    acc_dict["RNB_25"]="{:.2f}%".format(acc*100)
    
    print("RNB_25 Accuracy on the AGNEWS dataset: {:.2f}".format(acc*100))

    RNB_15 = RobustNaiveBayesMultiClassifierPercentage(15,num_classes=4)
    RNB_15.fit(training_features, training_labels)
    y_pred_RNB_15 = RNB_15.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_15)
    acc_dict["RNB_15"]="{:.2f}%".format(acc*100)
    print("RNB_15 Accuracy on the AGNEWS dataset: {:.2f}".format(acc*100))
    
    RNB_5 = RobustNaiveBayesMultiClassifierPercentage(5,num_classes=4)
    RNB_5.fit(training_features, training_labels)
    y_pred_RNB_5 = RNB_5.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_5)
    acc_dict["RNB_5"]="{:.2f}%".format(acc*100)
    print("RNB_5 Accuracy on the AGNEWS dataset: {:.2f}".format(acc*100))

    
    
    MNB = MultinomialNB(alpha=1.0)
    MNB.fit(training_features, training_labels)
    y_pred_MNB = MNB.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_MNB)
    acc_dict["MNB"]="{:.2f}%".format(acc*100)
    print("MNB Accuracy on the AGNEWS dataset: {:.2f}".format(acc*100))

    LR = LogisticRegression(random_state=0)
    LR.fit(training_features, training_labels)
    y_pred_LR = LR.predict(test_features)
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_LR)
    acc_dict["LR"]="{:.2f}%".format(acc*100)
    print("LR Accuracy on the AGNEWS dataset: {:.2f}".format(acc*100))
    
        
    BERT = OpenAttackBert("/home/ubuntu/Robustness_Gym/model/weights/BERT/AGNEWS/ONNX")
    y_pred_BERT = BERT.get_pred(list(test_data["text"]))
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_BERT)
    acc_dict["BERT"]="{:.2f}%".format(acc*100)
    print("BERT Accuracy on the AGNEWS dataset: {:.2f}".format(acc*100))
    from model.LSTMWrapper import loadLSTMModel,RNN

    LSTM = loadLSTMModel("/home/ubuntu/Robustness_Gym/model/weights/LSTM/AGNEWS_WordTokenizer",classes=4)
    y_pred_LSTM = LSTM.get_pred(list(test_data["text"]))
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_LSTM)
    acc_dict["LSTM"]="{:.2f}%".format(acc*100)
    print("LSTM Accuracy on the AGNEWS dataset: {:.2f}".format(acc*100))
    attacker = oa.attackers.TextBuggerAttacker()
    from model.GRUWrapper import loadGRUModel,RNN
    GRU = loadGRUModel("/home/ubuntu/Robustness_Gym/model/weights/GRU/AGNEWS_WordTokenizer",classes=4)
    y_pred_GRU = GRU.get_pred(list(test_data["text"]))
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_GRU)
    acc_dict["GRU"]="{:.2f}%".format(acc*100)
    
    print("GRU Accuracy on the AGNEWS dataset: {:.2f}".format(acc*100))
    
    bert_vectorizer = BertVectorizer()
    training_features = bert_vectorizer.transform(train_data["text"])
    test_features = bert_vectorizer.transform(test_data["text"])

    RNB_BERT = RobustNaiveBayesMultiClassifierPercentage(100,num_classes=4)
    RNB_BERT.fit(training_features, training_labels)
    y_pred_RNB_BERT = RNB_BERT.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT)
    acc_dict["RNB_BERT"]="{:.2f}%".format(acc*100)
    
    print("RNB_BERT Accuracy on the AGNEWS dataset: {:.2f}".format(acc*100))

    RNB_BERT_50 = RobustNaiveBayesMultiClassifierPercentage(50,num_classes=4)
    RNB_BERT_50.fit(training_features, training_labels)
    y_pred_RNB_BERT_50 = RNB_BERT_50.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT_50)
    acc_dict["RNB_BERT_50"]="{:.2f}%".format(acc*100)
    
    print("RNB_BERT_50 Accuracy on the AGNEWS dataset: {:.2f}".format(acc*100))

    RNB_BERT_25 = RobustNaiveBayesMultiClassifierPercentage(25,num_classes=4)
    RNB_BERT_25.fit(training_features, training_labels)
    y_pred_RNB_BERT_25 = RNB_BERT_25.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT_25)
    acc_dict["RNB_BERT_25"]="{:.2f}%".format(acc*100)
    
    print("RNB_BERT_25 Accuracy on the AGNEWS dataset: {:.2f}".format(acc*100))

    RNB_BERT_15 = RobustNaiveBayesMultiClassifierPercentage(15,num_classes=4)
    RNB_BERT_15.fit(training_features, training_labels)
    y_pred_RNB_BERT_15 = RNB_BERT_15.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT_15)
    acc_dict["RNB_BERT_15"]="{:.2f}%".format(acc*100)
    print("RNB_BERT_15 Accuracy on the AGNEWS dataset: {:.2f}".format(acc*100))
    
    RNB_BERT_5 = RobustNaiveBayesMultiClassifierPercentage(5,num_classes=4)
    RNB_BERT_5.fit(training_features, training_labels)
    y_pred_RNB_BERT_5 = RNB_BERT_5.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT_5)
    acc_dict["RNB_BERT_5"]="{:.2f}%".format(acc*100)
    print("RNB_BERT_5 Accuracy on the AGNEWS dataset: {:.2f}".format(acc*100))
    del training_features
    del test_features
    
    import json
    with open('/home/ubuntu/Robustness_Gym/result_official/AGNEWS/result.json', 'w') as f:
        json.dump(acc_dict, f)
        
######################################################################################
    vectorizer = CountVectorizer(stop_words="english",
                                preprocessor=clean_text_imdb, 
                                min_df=0)
    
    sst2_dataset = datasets.load_dataset("yelp_review_full")
    train_data = sst2_dataset["train"]
    test_data = sst2_dataset["test"]
    training_features = vectorizer.fit_transform(train_data["text"])
    training_labels = np.array(train_data["label"])
    test_features = vectorizer.transform(test_data["text"])
    test_labels = np.array(test_data["label"])
    acc_dict={}
    RNB = RobustNaiveBayesMultiClassifierPercentage(100,num_classes=5)
    RNB.fit(training_features, training_labels)
    y_pred_RNB = RNB.predict(test_features)
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB)
    acc_dict["RNB"]="{:.2f}%".format(acc*100)
    
    print("RNB Accuracy on the YELP dataset: {:.2f}".format(acc*100))

    RNB_50 = RobustNaiveBayesMultiClassifierPercentage(50,num_classes=5)
    RNB_50.fit(training_features, training_labels)
    y_pred_RNB_50 = RNB_50.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_50)
    acc_dict["RNB_50"]="{:.2f}%".format(acc*100)
    
    print("RNB_50 Accuracy on the YELP dataset: {:.2f}".format(acc*100))

    RNB_25 = RobustNaiveBayesMultiClassifierPercentage(25,num_classes=5)
    RNB_25.fit(training_features, training_labels)
    y_pred_RNB_25 = RNB_25.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_25)
    acc_dict["RNB_25"]="{:.2f}%".format(acc*100)
    
    print("RNB_25 Accuracy on the YELP dataset: {:.2f}".format(acc*100))

    RNB_15 = RobustNaiveBayesMultiClassifierPercentage(15,num_classes=5)
    RNB_15.fit(training_features, training_labels)
    y_pred_RNB_15 = RNB_15.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_15)
    acc_dict["RNB_15"]="{:.2f}%".format(acc*100)
    print("RNB_15 Accuracy on the YELP dataset: {:.2f}".format(acc*100))
    
    RNB_5 = RobustNaiveBayesMultiClassifierPercentage(5,num_classes=5)
    RNB_5.fit(training_features, training_labels)
    y_pred_RNB_5 = RNB_5.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_5)
    acc_dict["RNB_5"]="{:.2f}%".format(acc*100)
    print("RNB_5 Accuracy on the YELP dataset: {:.2f}".format(acc*100))

    
    
    MNB = MultinomialNB(alpha=1.0)
    MNB.fit(training_features, training_labels)
    y_pred_MNB = MNB.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_MNB)
    acc_dict["MNB"]="{:.2f}%".format(acc*100)
    print("MNB Accuracy on the YELP dataset: {:.2f}".format(acc*100))

    LR = LogisticRegression(random_state=0)
    LR.fit(training_features, training_labels)
    y_pred_LR = LR.predict(test_features)
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_LR)
    acc_dict["LR"]="{:.2f}%".format(acc*100)
    print("LR Accuracy on the YELP dataset: {:.2f}".format(acc*100))
    
            
    BERT = OpenAttackBert("/home/ubuntu/Robustness_Gym/model/weights/BERT/YELP/ONNX")
    y_pred_BERT = BERT.get_pred(list(test_data["text"]))
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_BERT)
    acc_dict["BERT"]="{:.2f}%".format(acc*100)
    print("BERT Accuracy on the YELP dataset: {:.2f}".format(acc*100))
    from model.LSTMWrapper import loadLSTMModel,RNN

    LSTM = loadLSTMModel("/home/ubuntu/Robustness_Gym/model/weights/LSTM/YELP_WordTokenizer",classes=5)
    y_pred_LSTM = LSTM.get_pred(list(test_data["text"]))
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_LSTM)
    acc_dict["LSTM"]="{:.2f}%".format(acc*100)
    print("LSTM Accuracy on the YELP dataset: {:.2f}".format(acc*100))
    attacker = oa.attackers.TextBuggerAttacker()
    from model.GRUWrapper import loadGRUModel,RNN
    GRU = loadGRUModel("/home/ubuntu/Robustness_Gym/model/weights/GRU/YELP_WordTokenizer",classes=5)
    y_pred_GRU = GRU.get_pred(list(test_data["text"]))
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_GRU)
    acc_dict["GRU"]="{:.2f}%".format(acc*100)
    
    print("GRU Accuracy on the YELP dataset: {:.2f}".format(acc*100))
    
    bert_vectorizer = BertVectorizer()
    training_features = bert_vectorizer.transform(train_data["text"])
    test_features = bert_vectorizer.transform(test_data["text"])

    RNB_BERT = RobustNaiveBayesMultiClassifierPercentage(100,num_classes=5)
    RNB_BERT.fit(training_features, training_labels)
    y_pred_RNB_BERT = RNB_BERT.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT)
    acc_dict["RNB_BERT"]="{:.2f}%".format(acc*100)
    
    print("RNB_BERT Accuracy on the YELP dataset: {:.2f}".format(acc*100))

    RNB_BERT_50 = RobustNaiveBayesMultiClassifierPercentage(50,num_classes=5)
    RNB_BERT_50.fit(training_features, training_labels)
    y_pred_RNB_BERT_50 = RNB_BERT_50.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT_50)
    acc_dict["RNB_BERT_50"]="{:.2f}%".format(acc*100)
    
    print("RNB_BERT_50 Accuracy on the YELP dataset: {:.2f}".format(acc*100))

    RNB_BERT_25 = RobustNaiveBayesMultiClassifierPercentage(25,num_classes=5)
    RNB_BERT_25.fit(training_features, training_labels)
    y_pred_RNB_BERT_25 = RNB_BERT_25.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT_25)
    acc_dict["RNB_BERT_25"]="{:.2f}%".format(acc*100)
    
    print("RNB_BERT_25 Accuracy on the YELP dataset: {:.2f}".format(acc*100))

    RNB_BERT_15 = RobustNaiveBayesMultiClassifierPercentage(15,num_classes=5)
    RNB_BERT_15.fit(training_features, training_labels)
    y_pred_RNB_BERT_15 = RNB_BERT_15.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT_15)
    acc_dict["RNB_BERT_15"]="{:.2f}%".format(acc*100)
    print("RNB_BERT_15 Accuracy on the YELP dataset: {:.2f}".format(acc*100))
    
    RNB_BERT_5 = RobustNaiveBayesMultiClassifierPercentage(5,num_classes=5)
    RNB_BERT_5.fit(training_features, training_labels)
    y_pred_RNB_BERT_5 = RNB_BERT_5.predict(test_features)

    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB_BERT_5)
    acc_dict["RNB_BERT_5"]="{:.2f}%".format(acc*100)
    print("RNB_BERT_5 Accuracy on the YELP dataset: {:.2f}".format(acc*100))
    del training_features
    del test_features
    
    import json
    with open('/home/ubuntu/Robustness_Gym/result_official/YELP/result.json', 'w') as f:
        json.dump(acc_dict, f)
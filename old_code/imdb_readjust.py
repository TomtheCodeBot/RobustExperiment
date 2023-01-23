from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from model.robustNB import RobustNaiveBayesClassifierPercentage
from model.LSTMWrapper import OpenAttackMNB
from model.LSTMWrapper import loadLSTMModel
from utils.preprocessing import clean_text_imdb
from utils.dataloader import load_train_test_imdb_data
from utils.res_posprocess import readjust_result
import OpenAttack as oa
import ssl
import numpy as np
from model.BERTWrapper import RNN

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

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
    training_labels = train_data["sentiment"]
    test_features = vectorizer.transform(test_data["text"])
    test_labels = test_data["sentiment"]
    result = {}
    RNB = RobustNaiveBayesClassifierPercentage(100)
    RNB.fit(training_features, training_labels)
    y_pred_RNB = RNB.predict(test_features)
    result["RNB"] = (np.squeeze(np.asarray(y_pred_RNB)) == np.asarray(test_labels)).sum()
    
    MNB = MultinomialNB(alpha=1.0)
    MNB.fit(training_features, training_labels)
    y_pred_MNB = MNB.predict(test_features)

    # Evaluation
    result["MNB"] = (np.squeeze(np.asarray(y_pred_MNB)) == np.asarray(test_labels)).sum()
    
    LR = LogisticRegression(random_state=0)
    LR.fit(training_features, training_labels)
    y_pred_LR = LR.predict(test_features)
    result["LR"] = (np.squeeze(np.asarray(y_pred_LR)) == np.asarray(test_labels)).sum()
    
    LSTM = loadLSTMModel("/home/ubuntu/Robustness_Gym/model/weights/LSTM/IMDB")
    y_pred_LSTM = LSTM.get_pred(list(test_data["text"]))
    
    # Evaluation
    result["LSTM"] = (np.squeeze(np.asarray(y_pred_LSTM)) == np.asarray(test_labels)).sum()
    
    LSTM = loadLSTMModel("/home/ubuntu/Robustness_Gym/model/weights/LSTM/IMDB_WordTokenizer")
    y_pred_LSTM = LSTM.get_pred(list(test_data["text"]))
    
    # Evaluation
    result["LSTM_WORD"] = (np.squeeze(np.asarray(y_pred_LSTM)) == np.asarray(test_labels)).sum()
    readjust_result("/home/ubuntu/Robustness_Gym/results/imdb/TextBugger",result)
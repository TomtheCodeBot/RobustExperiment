from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from model.robustNB import RobustNaiveBayesClassifierPercentage,RobustNaiveBayesMultiClassifierPercentage
from model.LSTMWrapper import OpenAttackMNB
from model.LSTMWrapper import loadLSTMModel
from utils.preprocessing import clean_text_imdb
from utils.dataloader import load_train_test_imdb_data
from utils.attacking_platform import attack_platform
from model.BERTWrapper import RNN
import OpenAttack as oa
import ssl

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

    RNB = RobustNaiveBayesMultiClassifierPercentage(100)
    RNB.fit(training_features, training_labels)
    y_pred_RNB = RNB.predict(test_features)
    print(y_pred_RNB)
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_RNB)

    print("RNB Accuracy on the IMDB dataset: {:.2f}".format(acc*100))
    y_pred_RNB = RNB.predict_proba(test_features)
    print(y_pred_RNB)

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

    print("LR Accuracy on the IMDB dataset: {:.2f}".format(acc*100))
    
    LSTM = loadLSTMModel("/home/ubuntu/Robustness_Gym/model/weights/LSTM/IMDB_WordTokenizer")
    y_pred_LSTM = LSTM.get_pred(list(test_data["text"]))
    
    # Evaluation
    acc = accuracy_score(test_labels, y_pred_LSTM)

    print("LSTM Accuracy on the IMDB dataset: {:.2f}".format(acc*100))
    # load some examples of SST-2 for evaluation
    # choose the costomized classifier as the victim model
    victim = OpenAttackMNB(RNB,vectorizer)
    
    attacker = oa.attackers.TextBuggerAttacker()
    attack_platform("imdb","test",victim,"RNB",attacker,"TextBugger",dataset_mapping)
    # choose PWWS as the attacker and initialize it with default parameters
    
    # launch attacks and print attack results 
    victim = OpenAttackMNB(MNB,vectorizer)
    attack_platform("imdb","test",victim,"MNB",attacker,"TextBugger",dataset_mapping)
    
    victim = LSTM
    attack_platform("imdb","test",victim,"LSTM_WORD",attacker,"TextBugger",dataset_mapping)
    
    victim = OpenAttackMNB(LR,vectorizer)
    attack_platform("imdb","test",victim,"LR",attacker,"TextBugger",dataset_mapping)
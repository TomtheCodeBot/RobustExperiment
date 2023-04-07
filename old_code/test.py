import scipy
import matplotlib.pyplot as plt
import re
import os
import numpy as np
import pandas as pd
from scipy.special import xlogy
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from model.robustNB import RobustNaiveBayesClassifierPercentage,RobustNaiveBayesMultiClassifierPercentage
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import datasets


def clean_text(text):
    """
    Applies some pre-processing on the given text.

    Steps :
    - Removing HTML tags
    - Removing punctuation
    - Lowering text
    """
    
    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)    
    text = re.sub(r"\'", "", text)    
    text = re.sub(r"\"", "", text)    
    
    # convert text to lowercase
    text = text.strip().lower()
    
    # replace punctuation characters with spaces
    filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text

def load_train_test_imdb_data(data_dir):
    """Loads the IMDB train/test datasets from a folder path.
    Input:
    data_dir: path to the "aclImdb" folder.
    
    Returns:
    train/test datasets as pandas dataframes.
    """

    data = {}
    for split in ["train", "test"]:
        data[split] = []
        for sentiment in ["neg", "pos"]:
            score = 1 if sentiment == "pos" else 0

            path = os.path.join(data_dir, split, sentiment)
            file_names = os.listdir(path)
            for f_name in file_names:
                with open(os.path.join(path, f_name), "r",encoding="utf8") as f:
                    review = f.read()
                    data[split].append([review, score])

    np.random.shuffle(data["train"])        
    data["train"] = pd.DataFrame(data["train"],
                                 columns=['text', 'sentiment'])

    np.random.shuffle(data["test"])
    data["test"] = pd.DataFrame(data["test"],
                                columns=['text', 'sentiment'])

    return data["train"], data["test"]

def calculate_delta(mnb,x,Y,t,positive=1,scale=1000):
    if positive==1:
        theta_min=np.argmin(mnb.theta_pos)
    else:
        theta_min=np.argmin(mnb.theta_neg)
    iter = x.shape[0]//scale
    for i in range(iter):
        chunk = x[(i*scale):((i+1)*scale)]
        chunk_y = Y[(i*scale):((i+1)*scale)]
        delta_neg = np.zeros((len(np.where(chunk_y == positive)[0]), chunk.shape[1]),dtype=np.int64)
        training_neg_wc = np.sum(chunk[np.where(chunk_y == positive)], axis=1)
        training_neg_wc = np.squeeze(training_neg_wc)
        delta_neg[:, np.argmin(theta_min)] = t/scale * training_neg_wc
        delta_neg.astype(np.int64)
        
        x[(i*scale):((i+1)*scale)][np.where(chunk_y == 0)[0]] += delta_neg
    return x
if __name__ == "__main__":
    """IMDB_DATASET = 'data/aclImdb/'
    train_data, test_data = load_train_test_imdb_data(IMDB_DATASET)"""
    sst2_dataset = datasets.load_dataset("SetFit/sst2")
    train_data = sst2_dataset["train"]
    test_data = sst2_dataset["test"]
    
    

    # Transform each text into a vector of word counts

    # the min_df filters out words that have occurrence less than a certain threshold. this is used to reduce memory
    # usage
    vectorizer = CountVectorizer(stop_words="english",
                                preprocessor=clean_text, 
                                min_df=0)

    # vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.2,
    #                              stop_words='english')

    """x_train = vectorizer.fit_transform(train_data["text"])
    x_train = x_train.toarray()
    y_train = train_data["sentiment"]
    x_test = vectorizer.transform(test_data["text"])
    x_test = x_test.toarray()
    y_test = test_data["sentiment"]"""
    x_train = vectorizer.fit_transform(train_data["text"])
    y_train = np.array(train_data["label"])
    x_train = x_train.toarray()
    x_test = vectorizer.transform(test_data["text"])
    x_test = x_test.toarray()
    y_test = np.array(test_data["label"])
    
    
    percentage_errors = [0, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    adv_training_accuracies_regular = []
    adv_test_accuracies_regular = []

    for t in percentage_errors:
        print("Now on: " + str(t))
        
        rnb = MultinomialNB(alpha=1.0)
        rnb.fit(x_train, y_train)
        predictions_test = rnb.predict(x_test)
        predictions_train = rnb.predict(x_train)
        
        training_features_adv = np.copy(x_train)
        k_pos = np.sum(x_train[np.where(y_train == 1)[0]])
        k_neg = np.sum(x_train[np.where(y_train == 0)[0]])
        training_features_adv = training_features_adv.astype(np.uint8)

        delta_pos = np.zeros((len(np.where(y_train == 1)[0]), x_train.shape[1]))
        training_pos_wc = np.sum(training_features_adv[np.where(y_train == 1)], axis=1)
        delta_pos[:, np.argmin(rnb.feature_log_prob_[1,:])] = t/100 * training_pos_wc
        delta_pos = delta_pos.astype(np.uint8)
        training_features_adv[np.where(y_train == 1)] += delta_pos
        
        delta_neg = np.zeros((len(np.where(y_train == 0)[0]), x_train.shape[1]))
        training_neg_wc = np.sum(training_features_adv[np.where(y_train == 0)], axis=1)
        delta_neg[:, np.argmin(rnb.feature_log_prob_[0,:])] = t/100 * training_neg_wc
        delta_neg= delta_neg.astype(np.uint8)
        training_features_adv[np.where(y_train == 0)] += delta_neg
        
        predictions_train_adv = rnb.predict(training_features_adv)
        acc_train = accuracy_score(y_train, predictions_train_adv)
        
        test_features_adv = np.copy(x_test)
        test_features_adv = test_features_adv.astype(np.uint8)
        k_pos = np.sum(x_test[np.where(y_test == 1)[0]])
        k_neg = np.sum(x_test[np.where(y_test == 0)[0]])
        
        delta_pos = np.zeros((len(np.where(y_test == 1)[0]), x_train.shape[1]))
        test_pos_wc = np.sum(test_features_adv[np.where(y_test == 1)[0]], axis=1)
        delta_pos[:, np.argmin(rnb.feature_log_prob_[1,:])] = t/100 * test_pos_wc
        delta_pos = delta_pos.astype(np.uint8)
        test_features_adv[np.where(y_test == 1)[0]] += delta_pos
        
        delta_neg = np.zeros((len(np.where(y_test == 0)[0]), x_train.shape[1]))
        test_neg_wc = np.sum(test_features_adv[np.where(y_test == 0)[0]], axis=1)
        delta_neg[:, np.argmin(rnb.feature_log_prob_[0,:])] = t/100 * test_neg_wc
        delta_neg= delta_neg.astype(np.uint8)
        test_features_adv[np.where(y_test == 0)[0]] += delta_neg
    
        
        predictions_test_adv = rnb.predict(test_features_adv)
        acc_test = accuracy_score(y_test, predictions_test_adv)
        
        print("Test Accuracy on the IMDB dataset: {:.2f}".format(acc_test*100))
        print("Train Accuracy on the IMDB dataset: {:.2f}".format(acc_train*100))
        print("------------------")
        
        adv_training_accuracies_regular.append(acc_train)
        adv_test_accuracies_regular.append(acc_test)
    adv_training_accuracies_robust = []
    adv_test_accuracies_robust = []


    for t in percentage_errors:
        
        print("Now on: " + str(t))
        
        rnb = RobustNaiveBayesClassifierPercentage(t)
        rnb.fit(x_train, y_train)
        predictions_test = rnb.predict(x_test)
        predictions_train = rnb.predict(x_train)
        
        training_features_adv = np.copy(x_train)
        training_features_adv = training_features_adv.astype(np.uint8)
        k_pos = np.sum(x_train[np.where(y_train == 1)[0]])
        k_neg = np.sum(x_train[np.where(y_train == 0)[0]])
        
        delta_pos = np.zeros((len(np.where(predictions_train == 1)[0]), x_train.shape[1]))
        training_pos_wc = np.sum(training_features_adv[np.where(predictions_train == 1)], axis=1)
        delta_pos[:, np.argmin(rnb.theta_pos)] = t/100 * training_pos_wc
        delta_pos = delta_pos.astype(np.uint8)
        training_features_adv[np.where(predictions_train == 1)] += delta_pos
        
        
        delta_neg = np.zeros((len(np.where(predictions_train == 0)[0]), x_train.shape[1]))
        training_neg_wc = np.sum(training_features_adv[np.where(predictions_train == 0)], axis=1)
        delta_neg[:, np.argmin(rnb.theta_neg)] = t/100 * training_neg_wc
        delta_neg = delta_neg.astype(np.uint8)
        training_features_adv[np.where(predictions_train == 0)] += delta_neg
        
        predictions_train_adv = rnb.predict(training_features_adv)
        acc_train = accuracy_score(y_train, predictions_train_adv)
        
        
        test_features_adv = np.copy(x_test)
        k_pos = np.sum(x_test[np.where(y_test == 1)[0]])
        k_neg = np.sum(x_test[np.where(y_test == 0)[0]])
        
        delta_pos = np.zeros((len(np.where(y_test == 1)[0]), x_train.shape[1]))
        test_pos_wc = np.sum(test_features_adv[np.where(y_test == 1)[0]], axis=1)
        delta_pos[:, np.argmin(rnb.theta_pos)] = t/100 * test_pos_wc
        delta_pos = delta_pos.astype(np.uint8)
        test_features_adv[np.where(y_test == 1)[0]] += delta_pos
        
        delta_neg = np.zeros((len(np.where(y_test == 0)[0]), x_train.shape[1]))
        test_neg_wc = np.sum(test_features_adv[np.where(y_test == 0)[0]], axis=1)
        delta_neg[:, np.argmin(rnb.theta_neg)] = t/100 * test_neg_wc
        delta_neg = delta_neg.astype(np.uint8)
        test_features_adv[np.where(y_test == 0)[0]] += delta_neg
    
        
        predictions_test_adv = rnb.predict(test_features_adv)
        acc_test = accuracy_score(y_test, predictions_test_adv)
        
        print("Test Accuracy adv on the IMDB dataset: {:.2f}".format(acc_test*100))
        print("Train Accuracy adv on the IMDB dataset: {:.2f}".format(acc_train*100))
        print("------------------")
        
        adv_training_accuracies_robust.append(acc_train)
        adv_test_accuracies_robust.append(acc_test)
        
        
    
    print(adv_training_accuracies_robust)
    print(adv_test_accuracies_robust)
    
    print(adv_training_accuracies_regular)
    print(adv_test_accuracies_regular)
    
    #IMDB adv train:[0.91288, 0.90264, 0.89692, 0.89004, 0.88352, 0.87672, 0.87252, 0.86972, 0.86812, 0.86524, 0.86392, 0.86292, 0.86168, 0.86032]
    #IMDB adv test:[0.81896, 0.8218, 0.82404, 0.82532, 0.82656, 0.82888, 0.82996, 0.8304, 0.83276, 0.83332, 0.83348, 0.83356, 0.83228, 0.83324]
    #IMDB normal train:[0.91288, 0.9112, 0.9018, 0.87468, 0.8184, 0.66412, 0.47804, 0.29344, 0.15744, 0.08328, 0.05168, 0.04228, 0.04224, 0.04404]
    #IMDB normal test:[0.81896, 0.81496, 0.80056, 0.75472, 0.66396, 0.4618, 0.28584, 0.1546, 0.07812, 0.04168, 0.02744, 0.02304, 0.02144, 0.02304]
    #SST2 adv train:[0.942485549132948, 0.9395953757225434, 0.9358381502890173, 0.9226878612716763, 0.8952312138728323, 0.8793352601156069, 0.8635838150289017, 0.8521676300578035, 0.8414739884393063, 0.8336705202312139, 0.8255780346820809, 0.8189306358381503, 0.8122832369942197, 0.8069364161849711]
    #SST2 adv test:[0.8094453596924767, 0.8099945085118067, 0.8077979132344866, 0.8077979132344866, 0.8034047226798462, 0.8034047226798462, 0.7984623833058759, 0.7962657880285557, 0.7918725974739155, 0.7797913234486545, 0.7753981328940143, 0.7677100494233937, 0.7655134541460736, 0.7627677100494233]
    #SST2 normal train:[0.942485549132948, 0.942485549132948, 0.942485549132948, 0.9423410404624277, 0.9333815028901734, 0.8956647398843931, 0.8569364161849711, 0.7871387283236995, 0.6939306358381503, 0.6248554913294798, 0.5408959537572254, 0.4492774566473988, 0.3763005780346821, 0.255635838150289]
    #SST2 normal test:[0.8094453596924767, 0.8094453596924767, 0.8094453596924767, 0.8088962108731467, 0.7885777045579352, 0.7204832509610104, 0.6375617792421746, 0.5623283909939594, 0.46293245469522243, 0.40801757276221856, 0.33772652388797364, 0.26468973091707854, 0.22185612300933552, 0.1372872048325096]
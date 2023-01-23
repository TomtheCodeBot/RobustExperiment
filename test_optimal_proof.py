from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from model.robustNB import RobustNaiveBayesClassifierPercentage
from utils.preprocessing import clean_text_imdb
from utils.dataloader import load_train_test_imdb_data
import ssl
import numpy as np
from sklearn.preprocessing import normalize
from scipy.stats import pearsonr   
import matplotlib.pyplot as plt
import pathlib
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
def plot(name,theta,log=True,bins=60000):
    
    if log:
        theta = np.log(theta)
    print(theta)
        
    theta = np.squeeze(np.asarray(theta))
    theta = theta.tolist()
    min_y_lim = min(theta) -1
    plt.ylim(bottom=min_y_lim)
    bin_combine = len(theta)//bins
    plot_data = []
    for i in range(bins):
        plot_data.append((sum(theta[i*bin_combine:(i+1)*bin_combine])/bin_combine)-min_y_lim)
    max_y_lim = max(plot_data)+min_y_lim+0.5
    plt.ylim(top=max_y_lim)
    pathlib.Path(f'/home/ubuntu/Robustness_Gym/plot/{name}').mkdir(parents=True, exist_ok=True) 

    plt.bar(range(0,bins), plot_data,bottom= min_y_lim)
    plt.savefig(f'/home/ubuntu/Robustness_Gym/plot/{name}/'+"RNB.jpg")
    plt.close()
    
    return
if __name__ == "__main__":
    train_data , test_data = load_train_test_imdb_data("/home/ubuntu/Robustness_Gym/data/aclImdb")
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
    
    
    print("RNB Accuracy on the IMDB dataset: {:.2f}".format(acc*100))


    positive_over_negative = np.array(np.divide(RNB.theta_pos,RNB.theta_neg))[0]
    negative_over_positive = np.array(np.divide(RNB.theta_neg,RNB.theta_pos))[0]
    
    theta_pos = np.array(RNB.theta_pos)[0]
    theta_neg = np.array(RNB.theta_neg)[0]
    
    theta_pos = positive_over_negative/np.sum(theta_pos)
    theta_neg = positive_over_negative/np.sum(theta_neg)

    positive_over_negative = positive_over_negative/np.sum(positive_over_negative)
    negative_over_positive = positive_over_negative/np.sum(negative_over_positive)
    
    print(pearsonr(positive_over_negative,theta_pos))
    print(pearsonr(negative_over_positive,theta_neg))
    
    positive_over_negative = np.array(np.divide(RNB.theta_pos,RNB.theta_neg))[0]
    negative_over_positive = np.array(np.divide(RNB.theta_neg,RNB.theta_pos))[0]
    
    theta_pos = np.array(RNB.theta_pos)[0]
    theta_neg = np.array(RNB.theta_neg)[0]
    
    indices_pos = np.argsort(positive_over_negative)[::-1]
    indices_neg = np.argsort(negative_over_positive)[::-1]
    
    theta_pos = theta_pos[indices_pos]
    theta_neg = theta_neg[indices_neg]
    
    positive_over_negative = positive_over_negative[indices_pos]
    negative_over_positive = negative_over_positive[indices_neg]
    
    plot("posoverneg",positive_over_negative,False,len(negative_over_positive))
    plot("thetapos",theta_pos,bins=len(negative_over_positive))
    plot("negoverpos",negative_over_positive,False,bins=len(negative_over_positive))
    plot("thetaneg",theta_neg,bins=len(negative_over_positive))
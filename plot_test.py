
from sklearn.feature_extraction.text import CountVectorizer
from utils.preprocessing import clean_text_imdb
from utils import plot_vocab_count
from utils.dataloader import *
import numpy as np
import datasets
import matplotlib.pyplot as plt

if __name__ == "__main__":
    vectorizer = CountVectorizer(stop_words="english",
                                preprocessor=clean_text_imdb, 
                                min_df=0)

    dataset = datasets.load_dataset("ag_news")
    train_data = dataset["train"]
    test_data = dataset["test"]
    training_features = vectorizer.fit_transform(train_data["text"])
    plot_vocab_count(vectorizer,train_data["text"],"agnews.jpg")
    
    dataset = datasets.load_dataset("yelp_review_full")
    train_data = dataset["train"]
    test_data = dataset["test"]
    training_features = vectorizer.fit_transform(train_data["text"])
    plot_vocab_count(vectorizer,train_data["text"],"yelp.jpg")
    
    dataset = datasets.load_dataset("SetFit/sst2")
    train_data = dataset["train"]
    test_data = dataset["test"]
    training_features = vectorizer.fit_transform(train_data["text"])
    plot_vocab_count(vectorizer,train_data["text"],"sst2.jpg")
    
    
    train_data , test_data = load_train_test_imdb_data("/home/ubuntu/Robustness_Gym/data/aclImdb")
    training_features = vectorizer.fit_transform(train_data["text"])
    test_features = vectorizer.transform(test_data["text"])
    plot_vocab_count(vectorizer,test_data["text"],"imdb.jpg")
    
    train_data , test_data = load_mr_data()
    training_features = vectorizer.fit_transform(train_data["text"])
    test_features = vectorizer.transform(test_data["text"])
    plot_vocab_count(vectorizer,test_data["text"],"mr.jpg")
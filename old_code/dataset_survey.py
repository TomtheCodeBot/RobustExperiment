from sklearn.feature_extraction.text import CountVectorizer
from utils.preprocessing import clean_text_imdb
from utils.dataloader import *
from utils import survey_huggingface_dataset
import ssl
import numpy as np
from tqdm import tqdm
import pathlib
import json
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

    training_features = vectorizer.fit_transform(train_data["text"])
    test_features = vectorizer.transform(test_data["text"])
    num_element = 0
    num_element+= training_features.sum()
    num_element+= test_features.sum()
    average_length = num_element/(training_features.shape[0]+test_features.shape[0])
    print(f"IMDB - train amount = {training_features.shape[0]}, test amount = {test_features.shape[0]}, avg. token length = {average_length}")
    
    train_data , test_data = load_mr_data()
    vectorizer = CountVectorizer(stop_words="english",
                                preprocessor=clean_text_imdb, 
                                min_df=0)

    training_features = vectorizer.fit_transform(train_data["text"])
    test_features = vectorizer.transform(test_data["text"])
    num_element = 0
    num_element+= training_features.sum()
    num_element+= test_features.sum()
    average_length = num_element/(training_features.shape[0]+test_features.shape[0])
    print(f"MR - train amount = {training_features.shape[0]}, test amount = {test_features.shape[0]}, avg. token length = {average_length}")

    huggingface_dataset = ["ag_news","yelp_review_full","SetFit/sst2"]
    for i in huggingface_dataset:
        train_amount,test_amount,avg = survey_huggingface_dataset(i)
        print(f"{i} - train amount = {train_amount}, test amount = {test_amount}, avg. token length = {avg}")
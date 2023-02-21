
from sklearn.feature_extraction.text import CountVectorizer
from utils.preprocessing import clean_text_imdb
from utils import plot_theta_RNB,plot_theta_MNB
from utils.dataloader import *
import numpy as np
import datasets
import matplotlib.pyplot as plt

if __name__ == "__main__":
    error_rate=[5,15,25,50,100]
    for i in error_rate:
        plot_theta_RNB("SetFit/sst2",index=None,num_classes=2,error_rate=i,bins=-1)
        #plot_theta_MNB("SetFit/sst2",index=None,num_classes=2)
        
        plot_theta_RNB("imdb",index=None,num_classes=2,error_rate=i,bins=-1)
        #plot_theta_MNB("imdb",index=None,num_classes=2)
        
        plot_theta_RNB("yelp_review_full",index=None,num_classes=5,error_rate=i,bins=-1)
        #plot_theta_MNB("yelp_review_full",index=None,num_classes=5)

        #plot_theta_RNB("mr",index=None,num_classes=2,error_rate=i)
        #plot_theta_MNB("mr",index=None,num_classes=2)
        
        plot_theta_RNB("ag_news",index=None,num_classes=4,error_rate=i,bins=-1)
        #plot_theta_MNB("ag_news",index=None,num_classes=4)
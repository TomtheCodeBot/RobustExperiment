from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from utils.preprocessing import clean_text_imdb
from utils.dataloader import load_train_test_imdb_data,load_mr_data
import numpy as np
import datasets
from model.robustNB  import RobustNaiveBayesMultiClassifierPercentage
import matplotlib.pyplot as plt
import math
import pathlib
import glob
import xlsxwriter
import json
vectorizer = CountVectorizer(stop_words="english",
                                preprocessor=clean_text_imdb, 
                                min_df=0)

def survey_huggingface_dataset(name:str,input_name:str="text"):
    dataset = datasets.load_dataset(name)
    train_data = dataset["train"]
    test_data = dataset["test"]
    training_features = vectorizer.fit_transform(train_data[input_name])
    test_features = vectorizer.transform(test_data[input_name])
    num_element = 0
    num_element+= training_features.sum()
    num_element+= test_features.sum()
    average_length = num_element/(training_features.shape[0]+test_features.shape[0])
    
    return training_features.shape[0],test_features.shape[0],average_length

def plot_theta_RNB(name:str="ag_news",input_name:str="text",label_name:str="label", error_rate:int=100,num_classes:int=4,index:int=1,bins:int=100):
    if name == "imdb":
        train_data , _ = load_train_test_imdb_data("/home/ubuntu/RobustExperiment/data/aclImdb")
    if name == "mr":
        train_data , _ = load_mr_data()
    else:
        dataset = datasets.load_dataset(name)
        train_data = dataset["train"]
    training_features = vectorizer.fit_transform(train_data[input_name])
    training_labels = np.array(train_data[label_name])
    RNB = RobustNaiveBayesMultiClassifierPercentage(error_rate,num_classes)
    RNB.fit(training_features, training_labels)
    theta = RNB.theta
    
    if index is None:
        sum_theta =  theta[0].copy()
        for i in range(1,len(theta)):
            sum_theta+=theta[i]
        theta = sum_theta/num_classes
    else:
        theta = theta[index]
    theta = np.log(theta)
    
    theta = np.squeeze(np.asarray(theta))
    theta = theta.tolist()
    min_y_lim = min(theta) -1
    plt.ylim(bottom=min_y_lim)
    theta = sorted(theta,reverse=True)
    if bins == -1:
        bins = len(theta)-1
    bin_combine = len(theta)//bins
    plot_data = []
    for i in range(bins):
        plot_data.append((sum(theta[i*bin_combine:(i+1)*bin_combine])/bin_combine)-min_y_lim)
    max_y_lim = max(plot_data)+min_y_lim+0.5
    plt.ylim(top=max_y_lim)
    pathlib.Path(f'/home/ubuntu/RobustExperiment/plot/{name}').mkdir(parents=True, exist_ok=True) 

    plt.bar(range(0,bins), plot_data,bottom= min_y_lim)
    plt.savefig(f'/home/ubuntu/RobustExperiment/plot/{name}/RNB_{str(error_rate)}.jpg')
    plt.close()
    
def plot_theta_MNB(name:str="ag_news",input_name:str="text",label_name:str="label", error_rate:int=100,num_classes:int=4,index:int=1,bins:int=100):
    if name == "imdb":
        train_data , _ = load_train_test_imdb_data("/home/ubuntu/Robustness_Gym/data/aclImdb")
    if name == "mr":
        train_data , _ = load_mr_data()
    else:
        dataset = datasets.load_dataset(name)
        train_data = dataset["train"]
    training_features = vectorizer.fit_transform(train_data[input_name])
    training_labels = np.array(train_data[label_name])
    RNB = MultinomialNB(alpha=1.0)
    RNB.fit(training_features, training_labels)
    theta = RNB.feature_log_prob_
    theta = np.power(10,theta)
    if index is None:
        sum_theta =  theta[0].copy()
        for i in range(1,len(theta)):
            sum_theta+=theta[i]
        theta = sum_theta/num_classes
    else:
        theta = theta[index]
    theta = np.log(theta)
    
    theta = np.squeeze(np.asarray(theta))
    theta = theta.tolist()
    min_y_lim = min(theta) -1
    plt.ylim(bottom=min_y_lim)
    theta = sorted(theta,reverse=True)
    bin_combine = len(theta)//bins
    plot_data = []
    for i in range(bins):
        plot_data.append((sum(theta[i*bin_combine:(i+1)*bin_combine])/bin_combine)-min_y_lim)
    max_y_lim = max(plot_data)+min_y_lim+0.5
    plt.ylim(top=max_y_lim)
    pathlib.Path(f'/home/ubuntu/Robustness_Gym/plot/{name}').mkdir(parents=True, exist_ok=True) 

    plt.bar(range(0,bins), plot_data,bottom= min_y_lim)
    plt.savefig(f'/home/ubuntu/Robustness_Gym/plot/{name}/'+"MNB.jpg")
    plt.close()
    
def plot_vocab_count(vectorizer,input_texts,name:str="sample.jpg",bins:int=100):
    
    plot_data = []
    vocab_count = [0 for i in range(len(vectorizer.vocabulary_))]
    indices = vectorizer.transform(input_texts)
    indices = indices.sum(axis=0)
    indices = np.squeeze(np.asarray(indices))
    vocab_count = indices.tolist()
    vocab_count = sorted(vocab_count,reverse=True)
    vocab_count = vocab_count[1:]
    sum_vocab = sum(vocab_count)
    print(sum_vocab)
    #vocab_count = [i/(sum_vocab) for i in vocab_count]
    bin_combine = len(vocab_count)//bins
    
    for i in range(bins):
        plot_data.append(((sum(vocab_count[i*bin_combine:(i+1)*bin_combine])/bin_combine)+1))
    plt.bar(range(0,bins), plot_data)
    plt.savefig("/home/ubuntu/Robustness_Gym/plot/"+name)
    plt.close()
    

def draw_excel(path_to_results):
    workbook = xlsxwriter.Workbook(path_to_results+"/results.xlsx")
    worksheet = workbook.add_worksheet()
    cellformat = workbook.add_format()
    cellformat.set_align('center')
    cellformat.set_align('vcenter')
    
    f = open(path_to_results+'/result.json')
    clean_acc = json.load(f)
    f.close()
    lst_attack = list(clean_acc.keys())
    first_collumn = 65
    worksheet.merge_range(f'{chr(first_collumn)}1:{chr(first_collumn)}2',"Models",cell_format=cellformat)
    worksheet.merge_range(f'{chr(first_collumn+1)}1:{chr(first_collumn+1)}2',"Clean Accuracy (%)",cell_format=cellformat)
    for i in range(0,len(lst_attack)):
        # Merge 3 cells.
        worksheet.merge_range(f'{chr(first_collumn+(i*2)+2)}1:{chr(first_collumn+(i*2)+3)}1',lst_attack[i],cell_format=cellformat)
        worksheet.write(f'{chr(first_collumn+(i*2)+2)}2', 'AuA(%) (ASR(%)↓)',cellformat)
        worksheet.write(f'{chr(first_collumn+(i*2)+3)}2', 'Avg. Query↓',cellformat)
        results = list(clean_acc[lst_attack[i]].keys())
        results = sorted(results, key=lambda item: (int(item.split("_")[-1])+len(item.split("_"))*(100)
                               if item.split("_")[-1].isdigit() else float(0), item))
        for k in range(len(results)):

            worksheet.write(f'{chr(first_collumn)}{str(k+3)}', results[k],cellformat)
            percentage  = clean_acc[lst_attack[i]][results[k]]["Attack success rate"]
            AuA = clean_acc[lst_attack[i]][results[k]]["Accuracy under attack"]
            worksheet.write(f'{chr(first_collumn+(i*2)+2)}{str(k+3)}', f"{AuA} ({percentage})",cellformat)
            querries  = clean_acc[lst_attack[i]][results[k]]["Avg num queries"]
            worksheet.write(f'{chr(first_collumn+(i*2)+3)}{str(k+3)}', f"{querries}",cellformat)
            worksheet.write(f'{chr(first_collumn+1)}{str(k+3)}', clean_acc[lst_attack[i]][results[k]]["Original accuracy"],cellformat)

    worksheet.autofit()
    workbook.close()
    pass

if __name__ == "__main__":
    draw_excel("/home/ubuntu/RobustExperiment/text_attack_result/SST2")
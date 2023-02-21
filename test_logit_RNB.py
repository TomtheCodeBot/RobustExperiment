from textattack.datasets import HuggingFaceDataset
from typing import List, Dict
import random
from textattack.attack_recipes import TextFoolerJin2019,HotFlipEbrahimi2017,DeepWordBugGao2018,TextBuggerLi2018
from textattack.transformations import WordSwapEmbedding
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.goal_functions import UntargetedClassification
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from utils.dataloader import load_train_test_imdb_data
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from utils.preprocessing import clean_text_imdb
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from utils.bert_vectorizer import BertVectorizer
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack import Attack
from textattack.models.wrappers import ModelWrapper,PyTorchModelWrapper,SklearnModelWrapper,HuggingFaceModelWrapper
from textattack.models.helpers import LSTMForClassification
import torch
import argparse
from textattack import Attacker,AttackArgs
from textattack.datasets import Dataset
import datasets
import numpy as np
import os 
from sklearn.base import BaseEstimator
from scipy.special import logit

class RobustNaiveBayesClassifierPercentage(BaseEstimator):

    def __init__(self, percentage_noise=0.0, debug=False):

        # assert 0 <= percentage_noise <= 100, "Please enter a valid percentage between 0 and 100 inclusive"

        self.percentage_noise = percentage_noise

        self.kappa = None

        self.pos_prior_probability = None
        self.neg_prior_probability = None
        self.theta_pos = None
        self.theta_neg = None
        self.f_pos = None
        self.f_neg = None
        self.pos_indices = None
        self.neg_indices = None

        self._has_fit = False

        self.debug = debug

    def _solve_subproblem(self, f, kappa):
        '''
        Solves the subproblem:
        max_{v > 0} F(v) = kappa * log v + \sum f_i log max(f_i, v) - max(f_i, v) 

        where kappa is self.percentage_noise * sum of word counts in text corpus training
        Returns the optimal value of the dual variable theta*
        '''
        if self.debug:
            print("Internal kappa percentage: " + str(kappa))

        if (len(f.shape) == 2):
            n = f.shape[1]
        else:
            n = f.shape[0]

        f_l1 = np.sum(f)

        f_int = (-np.sort(-f))
        f_int = np.insert(f_int, 0, 9999999)
        f_new = np.insert(f_int, n+1, 0.)

        if (len(f_new.shape) < 2):
            f_new = np.reshape(f_new, (1, f_new.shape[0]))

        rho_curr = kappa + f_l1
        h_curr = 0
        v_curr = max(f_new[0, 1], rho_curr/n)
        F_curr = rho_curr * np.log(v_curr) - n * v_curr

        v_star = v_curr
        F_star = F_curr
        k_star = 0

        for k in range(1, n + 1):
            if k == n:
                h_curr = h_curr + f_new[0, k] * \
                    np.log(f_new[0, k]) - f_new[0, k]
                F_curr = h_curr + kappa * np.log(f_new[0, k])
                v_curr = f_new[0, k]
            else:
                rho_curr = rho_curr - f_new[0, k]
                h_curr = h_curr + f_new[0, k] * \
                    np.log(f_new[0, k]) - f_new[0, k]
                v_curr = min(f_new[0, k], max(
                    f_new[0, k + 1], rho_curr/(n - k)))
                F_curr = h_curr + rho_curr * np.log(v_curr) - (n - k) * v_curr

            if F_curr > F_star:
                F_star = F_curr
                v_star = v_curr
                k_star = k

        theta = (1./(kappa + f_l1)) * np.maximum(f, v_star * np.ones(f.shape))

        return theta

    def fit(self, X, y):
        self.pos_indices = np.where(y == 1)[0]
        self.neg_indices = np.where(y == 0)[0]
        self.f_pos = np.sum(X[self.pos_indices], axis=0)
        self.f_neg = np.sum(X[self.neg_indices], axis=0)

        self.f_pos = 1 + self.f_pos
        self.f_neg = 1 + self.f_neg

        self.kappa_pos = (self.percentage_noise/100) * np.sum(self.f_pos)
        self.kappa_neg = (self.percentage_noise/100) * np.sum(self.f_neg)

        self.theta_pos = self._solve_subproblem(self.f_pos, self.kappa_pos)
        self.theta_neg = self._solve_subproblem(self.f_neg, self.kappa_neg)

        if (len(self.theta_pos.shape) < 2):
            self.theta_pos = np.reshape(
                self.theta_pos, (1, self.theta_pos.shape[0]))

        if (len(self.theta_neg.shape) < 2):
            self.theta_neg = np.reshape(
                self.theta_neg, (1, self.theta_neg.shape[0]))

        self.pos_prior_probability = len(self.pos_indices)/X.shape[0]
        self.neg_prior_probability = len(self.neg_indices)/X.shape[0]

        self._has_fit = True

    def predict(self, X):
        if not self._has_fit:
            print("Please call fit() before you start predicting")
            return None

        if self.theta_pos.shape[1] != X.shape[1]:
            print("Shape mismatch. Please train with proper dimensions")
            return None
        pos_prob = np.log(self.pos_prior_probability) + \
            np.sum(X@np.log(self.theta_pos).T, axis=-1)
        neg_prob = np.log(self.neg_prior_probability) + \
            np.sum(X@np.log(self.theta_neg).T, axis=-1)

        predictions = (pos_prob >= neg_prob).astype(int)

        return predictions

    def predict_proba(self, X):
        if not self._has_fit:
            print("Please call fit() before you start predicting")
            return None

        if self.theta_pos.shape[1] != X.shape[1]:
            print("Shape mismatch. Please train with proper dimensions")
            return None

        pos_prob = np.log(self.pos_prior_probability) + \
            np.sum(X@np.log(self.theta_pos).T, axis=-1)
        neg_prob = np.log(self.neg_prior_probability) + \
            np.sum(X@np.log(self.theta_neg).T, axis=-1)

        exp_pos = np.expand_dims(pos_prob, axis=-1)
        exp_neg = np.expand_dims(neg_prob, axis=-1)

        output = np.concatenate((exp_neg, exp_pos), axis=1)
        if len(output.shape)>2:
            output = np.squeeze(np.array(output),axis=-1)
        return output


def build_attacker(model,args):
    if (args['attack_method'] == 'textfooler'):
        attacker=TextFoolerJin2019.build(model)
    elif (args['attack_method'] == 'textbugger'):
        attacker=TextBuggerLi2018.build(model)
    elif (args['attack_method'] == 'deepwordbug'):
        attacker=DeepWordBugGao2018.build(model)
    else:
        attacker=TextFoolerJin2019.build(model)
    if(args['modify_ratio']!=0):
        attacker.constraints.append(MaxWordsPerturbed(max_percent=args['modify_ratio']))
    return attacker

def str2bool(strIn):
    if strIn.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif strIn.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        print(strIn)
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
def set_seed(seed=42):
    if seed is not None:
        print(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True
        
def attack(args,wrapper,name,dataset):
    attackMethod=args.attack_method
    attack_args_dict = {'attack_method': attackMethod, 'attack_times': 1,'attack_examples':int(args.attack_examples),'modify_ratio':float(args.modify_ratio),
            'log_path': '{}/{}/{}_{}-{}.txt'.format(args.load_path, attackMethod,args.attack_dataset,args.modify_ratio,name)}
    attack = build_attacker(wrapper, attack_args_dict)
    attack_args = AttackArgs(num_examples=attack_args_dict['attack_examples'], log_to_txt=attack_args_dict['log_path'], csv_coloring_style="file")
    attacker=Attacker(attack,dataset,attack_args)
    attacker.attack_dataset()
    
def gen_dataset(instances):
    test_instances=instances
    test_dataset=[]
    for instance in range(len(test_instances)):
        test_dataset.append((test_instances["text"][instance],int(test_instances["label"][instance])))
    dataset=Dataset(test_dataset,shuffle=True)
    return dataset

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="parameter")
    parser.add_argument('-hd','--hidden_size',default=500)
    parser.add_argument('-be','--beta',default='0.1')
    parser.add_argument('-b','--batch_size',default=64)
    parser.add_argument('-s','--seed',default=0)
    parser.add_argument('-d','--dropout',default=0.1)
    parser.add_argument('-g','--gpu_num',default=2)
    parser.add_argument('-a','--attack',type=str2bool,nargs='?',const=False)
    parser.add_argument('-l','--load',type=str2bool,nargs='?',const=False)
    parser.add_argument('-t', '--test', type=str2bool, nargs='?', const=False)
    parser.add_argument('-lp', '--load_path', default=None)
    parser.add_argument('-am', '--attack_method', default=None)
    parser.add_argument('-ai', '--adv_init_mag', default=0.08)
    parser.add_argument('-as', '--adv_steps', default=3)
    parser.add_argument('-al', '--adv_lr', default=0.04)
    parser.add_argument('-amn','--adv_max_norm',default=None)
    parser.add_argument('-ad','--attack_dataset',default='test')#attack dataset & accuracy dataset
    parser.add_argument('-ae','--attack_examples',default=200)
    parser.add_argument('-mr','--modify_ratio',default=0.15)
    parser.add_argument('-e', '--epoch', default=10)
    parser.add_argument('-se','--save_epoch',type=str2bool,nargs='?',const=False)
    parser.add_argument('-c','--class_num',default=2)
    parser.add_argument('-dp','--dataset_path',default='dataset/sst2')
    parser.add_argument('-sp','--save_path',default='')
    args = parser.parse_args()
    
    vectorizer = CountVectorizer(stop_words="english",
                            preprocessor=clean_text_imdb, 
                            min_df=0)

    train_data , test_data = load_train_test_imdb_data("/home/ubuntu/RobustExperiment/data/aclImdb")
    vectorizer = CountVectorizer(stop_words="english",
                                preprocessor=clean_text_imdb, 
                                min_df=0)
    training_features = vectorizer.fit_transform(train_data["text"])
    training_labels = train_data["label"]
    test_features = vectorizer.transform(test_data["text"])
    test_labels = test_data["label"]
    set_seed(1)
    RNB = RobustNaiveBayesClassifierPercentage(100)
    RNB.fit(training_features, training_labels)
    
    RNB_50 = RobustNaiveBayesClassifierPercentage(50)
    RNB_50.fit(training_features, training_labels)
    
    
    RNB_25 = RobustNaiveBayesClassifierPercentage(25)
    RNB_25.fit(training_features, training_labels)
    
    RNB_15 = RobustNaiveBayesClassifierPercentage(15)
    RNB_15.fit(training_features, training_labels)
    
    RNB_5 = RobustNaiveBayesClassifierPercentage(5)
    RNB_5.fit(training_features, training_labels)
    
    
    MNB = MultinomialNB(alpha=1.0)
    MNB.fit(training_features, training_labels)

    LR = LogisticRegression(random_state=0)
    LR.fit(training_features, training_labels)
    
    bert_vectorizer = BertVectorizer()
    training_features = bert_vectorizer.transform(train_data["text"])
    test_features = bert_vectorizer.transform(test_data["text"])
    RNB_BERT = RobustNaiveBayesClassifierPercentage(100)
    RNB_BERT.fit(training_features, training_labels)
    
    RNB_BERT_50 = RobustNaiveBayesClassifierPercentage(50)
    RNB_BERT_50.fit(training_features, training_labels)
    
    RNB_BERT_25 = RobustNaiveBayesClassifierPercentage(25)
    RNB_BERT_25.fit(training_features, training_labels)
    
    RNB_BERT_15 = RobustNaiveBayesClassifierPercentage(15)
    RNB_BERT_15.fit(training_features, training_labels)
    
    
    RNB_BERT_5 = RobustNaiveBayesClassifierPercentage(5)
    RNB_BERT_5.fit(training_features, training_labels)
    
    
    
    del training_features
    del test_features
    for i in range(1,3):
        set_seed(i)
        dataset = gen_dataset(test_data)
        args.load_path=f"/home/ubuntu/RobustExperiment/text_attack_result/test_RNB_with_logit/IMDB/{i}/"
        args.attack_method="deepwordbug"
        
        wrapper = SklearnModelWrapper(MNB,vectorizer)
        attack(args,wrapper,"MNB",dataset)
        
        wrapper = SklearnModelWrapper(LR,vectorizer)
        attack(args,wrapper,"LR",dataset)
        
        wrapper = SklearnModelWrapper(RNB,vectorizer)
        attack(args,wrapper,"RNB_100",dataset)
        
        wrapper = SklearnModelWrapper(RNB_50,vectorizer)
        attack(args,wrapper,"RNB_50",dataset)
        
        wrapper = SklearnModelWrapper(RNB_25,vectorizer)
        attack(args,wrapper,"RNB_25",dataset)
        
        wrapper = SklearnModelWrapper(RNB_15,vectorizer)
        attack(args,wrapper,"RNB_15",dataset)
        
        wrapper = SklearnModelWrapper(RNB_5,vectorizer)
        attack(args,wrapper,"RNB_5",dataset)
        
        """wrapper = SklearnModelWrapper(RNB_BERT,bert_vectorizer)
        attack(args,wrapper,"RNB_BERT_100",dataset)
        
        wrapper = SklearnModelWrapper(RNB_BERT_50,bert_vectorizer)
        attack(args,wrapper,"RNB_BERT_50",dataset)
        
        wrapper = SklearnModelWrapper(RNB_BERT_25,bert_vectorizer)
        attack(args,wrapper,"RNB_BERT_25",dataset)
        
        wrapper = SklearnModelWrapper(RNB_BERT_15,bert_vectorizer)
        attack(args,wrapper,"RNB_BERT_15",dataset)
        
        wrapper = SklearnModelWrapper(RNB_BERT_5,bert_vectorizer)
        attack(args,wrapper,"RNB_BERT_5",dataset)
        
        
        args.attack_method="textbugger"
        
        wrapper = SklearnModelWrapper(MNB,vectorizer)
        attack(args,wrapper,"MNB",dataset)
        
        wrapper = SklearnModelWrapper(LR,vectorizer)
        attack(args,wrapper,"LR",dataset)
        
        wrapper = SklearnModelWrapper(RNB,vectorizer)
        attack(args,wrapper,"RNB_100",dataset)
        
        wrapper = SklearnModelWrapper(RNB_50,vectorizer)
        attack(args,wrapper,"RNB_50",dataset)
        
        wrapper = SklearnModelWrapper(RNB_25,vectorizer)
        attack(args,wrapper,"RNB_25",dataset)
        
        wrapper = SklearnModelWrapper(RNB_15,vectorizer)
        attack(args,wrapper,"RNB_15",dataset)
        
        wrapper = SklearnModelWrapper(RNB_5,vectorizer)
        attack(args,wrapper,"RNB_5",dataset)
        
        wrapper = SklearnModelWrapper(RNB_BERT,bert_vectorizer)
        attack(args,wrapper,"RNB_BERT_100",dataset)
        
        wrapper = SklearnModelWrapper(RNB_BERT_50,bert_vectorizer)
        attack(args,wrapper,"RNB_BERT_50",dataset)
        
        wrapper = SklearnModelWrapper(RNB_BERT_25,bert_vectorizer)
        attack(args,wrapper,"RNB_BERT_25",dataset)
        
        wrapper = SklearnModelWrapper(RNB_BERT_15,bert_vectorizer)
        attack(args,wrapper,"RNB_BERT_15",dataset)
        
        wrapper = SklearnModelWrapper(RNB_BERT_5,bert_vectorizer)
        attack(args,wrapper,"RNB_BERT_5",dataset)
            
        args.attack_method="textfooler"

        wrapper = SklearnModelWrapper(MNB,vectorizer)
        attack(args,wrapper,"MNB",dataset)
        
        wrapper = SklearnModelWrapper(LR,vectorizer)
        attack(args,wrapper,"LR",dataset)
        
        wrapper = SklearnModelWrapper(RNB,vectorizer)
        attack(args,wrapper,"RNB_100",dataset)
        
        wrapper = SklearnModelWrapper(RNB_50,vectorizer)
        attack(args,wrapper,"RNB_50",dataset)
        
        wrapper = SklearnModelWrapper(RNB_25,vectorizer)
        attack(args,wrapper,"RNB_25",dataset)
        
        wrapper = SklearnModelWrapper(RNB_15,vectorizer)
        attack(args,wrapper,"RNB_15",dataset)
        
        wrapper = SklearnModelWrapper(RNB_5,vectorizer)
        attack(args,wrapper,"RNB_5",dataset)
        
        wrapper = SklearnModelWrapper(RNB_BERT,bert_vectorizer)
        attack(args,wrapper,"RNB_BERT_100",dataset)
        
        wrapper = SklearnModelWrapper(RNB_BERT_50,bert_vectorizer)
        attack(args,wrapper,"RNB_BERT_50",dataset)
        
        wrapper = SklearnModelWrapper(RNB_BERT_25,bert_vectorizer)
        attack(args,wrapper,"RNB_BERT_25",dataset)
        
        wrapper = SklearnModelWrapper(RNB_BERT_15,bert_vectorizer)
        attack(args,wrapper,"RNB_BERT_15",dataset)
        
        wrapper = SklearnModelWrapper(RNB_BERT_5,bert_vectorizer)
        attack(args,wrapper,"RNB_BERT_5",dataset)""" 
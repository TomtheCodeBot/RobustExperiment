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
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from utils.preprocessing import clean_text_imdb
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from model.robustNB import RobustNaiveBayesClassifierPercentage
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
class CustomModelWrapper(PyTorchModelWrapper):
    def __init__(self,model,tokenizer):
        super(CustomModelWrapper,self).__init__(model,tokenizer)

    def __call__(self,text_input_list):
        inputs_dict = self.tokenizer(
            text_input_list,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        inputs_dict.to(model_device)

        with torch.no_grad():
            outputs = self.model(**inputs_dict)

        if isinstance(outputs,tuple):
            return outputs[-1]#model-h,model-bh

        if isinstance(outputs,torch.Tensor):
            return outputs#baseline

        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence models return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            return outputs
        else:
            # HuggingFace classification models return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.
            return outputs.logits

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
    parser.add_argument('-ae','--attack_examples',default=1000)
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

    sst2_dataset = datasets.load_dataset("SetFit/sst2")
    train_data = sst2_dataset["train"]
    test_data = sst2_dataset["test"]
    training_features = vectorizer.fit_transform(train_data["text"])
    training_labels = np.array(train_data["label"])
    test_features = vectorizer.transform(test_data["text"])
    test_labels = np.array(test_data["label"])
    
    
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
    
    tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2",use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2")
    BERT = HuggingFaceModelWrapper(model,tokenizer)
    
    model = LSTMForClassification.from_pretrained("lstm-imdb")
    LSTM = PyTorchModelWrapper(
                        model, model.tokenizer
                    )
    
    del training_features
    del test_features
    for i in range(0,3):
        set_seed(i)
        dataset = gen_dataset(test_data)
        args.load_path=f"/home/ubuntu/RobustExperiment/text_attack_result/SST2/{i}/"
        args.attack_method="deepwordbug"
        
        attack(args,LSTM,"LSTM",dataset)
        
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
        
        attack(args,BERT,"BERT",dataset)
        
        args.attack_method="textbugger"
        
        attack(args,LSTM,"LSTM",dataset)
        
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
        
        attack(args,BERT,"BERT",dataset)
        
        args.attack_method="textfooler"
        
        attack(args,LSTM,"LSTM",dataset)
        
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
        
        attack(args,BERT,"BERT",dataset)
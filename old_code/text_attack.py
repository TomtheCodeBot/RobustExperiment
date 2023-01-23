from textattack.datasets import HuggingFaceDataset
from typing import List, Dict
import random
import argparse
from textattack import Attacker,AttackArgs
from textattack.datasets import Dataset
from utils.dataloader import load_train_test_imdb_data
import datasets
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
from simpletransformers.classification import ClassificationModel

from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack import Attack
from textattack.models.wrappers import ModelWrapper,PyTorchModelWrapper
import torch
import os
import numpy as np

class BERTModelWrapper(PyTorchModelWrapper):
    def __init__(self,bert_model_path, batch=32, num_classes=2):
        self.model = ClassificationModel(
            'bert',
            bert_model_path,
            use_cuda=True,

            args={"silent": True}
        )
        self.softmax = lambda x: np.exp(x)/np.sum(np.exp(x), axis=-1)[:,None]
        self.batch_size = batch
        self.num_classes = num_classes
    def __call__(self,text_input_list):
        iter_range = len(text_input_list)//self.batch_size
        if len(text_input_list) % self.batch_size != 0:
            iter_range += 1
        result = torch.empty((len(text_input_list), self.num_classes), dtype=torch.float16)
        for i in range(iter_range):
            result[i*self.batch_size:(i+1)*self.batch_size] = torch.FloatTensor(self.softmax(
                self.model.predict(text_input_list[i*self.batch_size:(i+1)*self.batch_size])[1]))
        return result.detach().numpy()
    
    
def str2bool(strIn):
    if strIn.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif strIn.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        print(strIn)
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
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
def attack(args,instances,model="/home/ubuntu/Robustness_Gym/model/weights/BERT/SST2"):
    wrapper = BERTModelWrapper(model)
    attack_arg_dict = {'attack_method': args.attack_method, 'attack_times': 1,'attack_examples':int(args.attack_examples),'modify_ratio':float(args.modify_ratio),
                'log_path': '{}_{}_{}_{}.txt'.format(args.load_path, args.attack_method,args.attack_dataset,args.modify_ratio)}
    attack = build_attacker(wrapper, attack_arg_dict)
    attack_args = AttackArgs(num_examples=attack_arg_dict['attack_examples'], log_to_txt=attack_arg_dict['log_path'], csv_coloring_style="file")
    test_instances=instances
    for i in range(attack_arg_dict['attack_times']):
        print("Attack time {}".format(i))
        test_dataset=[]
        for instance in range(len(test_instances)):
            test_dataset.append((test_instances["text"][instance],int(test_instances["label"][instance])))
        dataset=Dataset(test_dataset,shuffle=True)
        attacker=Attacker(attack,dataset,attack_args)
        attacker.attack_dataset()
        
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
        
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="parameter")
    parser.add_argument('-hd','--hidden_size',default=500)
    parser.add_argument('-be','--beta',default='0.1')
    parser.add_argument('-b','--batch_size',default=64)
    parser.add_argument('-s','--seed',default=0)
    parser.add_argument('-d','--dropout',default=0.1)
    parser.add_argument('-g','--gpu_num',default=1)
    parser.add_argument('-a','--attack',type=str2bool,nargs='?',const=False)
    parser.add_argument('-l','--load',type=str2bool,nargs='?',const=False)
    parser.add_argument('-t', '--test', type=str2bool, nargs='?', const=False)
    parser.add_argument('-lp', '--load_path', default="/home/ubuntu/Robustness_Gym/TEXT_ATTACK_RESULT")
    parser.add_argument('-am', '--attack_method', default="deepwordbug")
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
    seed=int(args.seed)
    set_seed(seed)
    sst2_dataset = datasets.load_dataset("SetFit/sst2")
    train_data = sst2_dataset["train"]
    test_data = sst2_dataset["test"]

    attack(args,test_data)

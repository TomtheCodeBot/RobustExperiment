import datasets
from utils.dataloader import load_mr_data,load_train_test_imdb_data

if __name__ == "__main__":
   
    sst2_dataset = datasets.load_dataset("ag_news")
    sst2_dataset["train"].to_csv("/home/ubuntu/Robustness_Gym/tsv_data/AGNEWS/train.tsv",index=False,sep='\u0249',header=['sentence',"label"])
    sst2_dataset["test"].to_csv("/home/ubuntu/Robustness_Gym/tsv_data/AGNEWS/dev.tsv",index=False,sep='\u0249',header=['sentence',"label"])
    
    sst2_dataset = datasets.load_dataset("SetFit/sst2")
    sst2_dataset["train"].to_csv("/home/ubuntu/Robustness_Gym/tsv_data/SST2/train.tsv",columns =['text', 'label'],index=False,sep='\u0249',header=['sentence',"label"])
    sst2_dataset["test"].to_csv("/home/ubuntu/Robustness_Gym/tsv_data/SST2/dev.tsv",columns =['text', 'label'],index=False,sep='\u0249',header=['sentence',"label"])
    
    sst2_dataset = datasets.load_dataset("yelp_review_full")
    sst2_dataset["train"].to_csv("/home/ubuntu/Robustness_Gym/tsv_data/YELP/train.tsv",columns =['text', 'label'],index=False,sep='\u0249',header=['sentence',"label"])
    sst2_dataset["test"].to_csv("/home/ubuntu/Robustness_Gym/tsv_data/YELP/dev.tsv",columns =['text', 'label'],index=False,sep='\u0249',header=['sentence',"label"])
    
    train_data , test_data = load_mr_data()
    train_data.to_csv("/home/ubuntu/Robustness_Gym/tsv_data/MR/train.tsv",index=False,sep='\u0249',header=['sentence',"label"])
    test_data.to_csv("/home/ubuntu/Robustness_Gym/tsv_data/MR/dev.tsv",index=False,sep='\u0249',header=['sentence',"label"])
    
    train_data , test_data = load_train_test_imdb_data("/home/ubuntu/Robustness_Gym/data/aclImdb")
    train_data.to_csv("/home/ubuntu/Robustness_Gym/tsv_data/IMDB/train.tsv",index=False,sep='\u0249',header=['sentence',"label"])
    test_data.to_csv("/home/ubuntu/Robustness_Gym/tsv_data/IMDB/dev.tsv",index=False,sep='\u0249',header=['sentence',"label"])
    
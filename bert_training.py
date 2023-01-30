from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import datasets
import sklearn
from utils.dataloader import load_train_test_imdb_data

#train_data , test_data = load_train_test_imdb_data("/home/ubuntu/RobustExperiment/data/aclImdb")
sst2_dataset = datasets.load_dataset("sst2")
train_data = pd.DataFrame(sst2_dataset["train"])
test_data = pd.DataFrame(sst2_dataset["validation"])
print(train_data)
train_data = train_data.rename({'label': 'labels'}, axis=1).drop(columns=["idx"])
test_data = test_data.rename({'label': 'labels'}, axis=1).drop(columns=["idx"])
# Optional model configuration
model_args = ClassificationArgs(num_train_epochs=10,
                                output_dir="/home/ubuntu/RobustExperiment/model/weights/BERT/SST2",
                                train_batch_size=32)


# Create a ClassificationModel
model = ClassificationModel(
    'bert',
    'bert-base-uncased',
    use_cuda=True,
    args=model_args
) 

# Train the model
model.train_model(train_data,eval_df =test_data ,eval_during_training=True)


# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(test_data,acc = sklearn.metrics.accuracy_score)
print(result)
print(len(wrong_predictions))
print(len(test_data))
"""print(model.predict(["I'd been postponing purchasing this one ever since its DVD release \x96 for one thing, because I'd been somewhat underwhelmed by this director's two other horror titles (SQUIRM [1976] and BLUE SUNSHINE [1977]), but also the fact that the film itself is said to have been slightly trimmed for gore on the Media Blasters/Shriek Show 2-Disc Set! I now chanced upon it as a rental and am glad I did \x96 because, not only is it superior to the earlier efforts (at least, on this preliminary assessment), but I also found the film to be one of the better imitations of THE Texas CHAIN SAW MASSACRE (1974). This factor, however, only helped remind me that I've yet to check out another such example \x96 Wes Craven's classic THE HILLS HAVE EYES (1977), whose 2-Disc R1 edition from Anchor Bay I purchased some time ago but, after all, Halloween-time is fast approaching... <br /><br />Anyway, the film manages an effortlessly unsettling backwoods atmosphere (it was shot in the forest and mountain regions of Oregon) \x96 with plenty of effective frissons throughout but, thankfully, not too much violence (even if the last of the villains is dispatched in quite an outrageous fashion!). The principal young cast here (one of them played by Jack Lemmon's son, Chris) isn't quite as obnoxious as those we usually encounter in this type of genre offering \x96 despite freely indulging in the shenanigans one associates with teen-oriented flicks and which, by and large, persist to this day! George Kennedy appears as a sympathetic Ranger; though he doesn't have a lot to do, his characterization is decidedly enhanced by making him a lover of plant and animal life. Also notable among the locals is familiar character actor Mike Kellin in a nice role as the drunkard who first comes into contact with the murderous duo of the narrative \x96 his warning to the teenagers, naturally, goes unheeded but he's later able to lead Kennedy to them.<br /><br />The hermetic family the teens come across in the woods, then, is eventually revealed to be hiding a skeleton in their closet. While one of the girls displays genuine curiosity at the intruders' presence, the rest are openly hostile to them \x96 and, in the case of the burly and uncouth twins, appropriately creepy (one of them is even prone to maniacal laughter during his rampages); at a certain point in the narrative, the Ranger even offers an interesting explanation as to the nature of their aggressive and generally uncivilized behavior."]))"""
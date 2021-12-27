import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from Tweet import Tweet

# we make use of a GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print('Loading BERT Model ...')
# We load our finetuned Bert model hosted on the huggingface platform
tokenizer = AutoTokenizer.from_pretrained("adam-chell/tweet-sentiment-analyzer")
model = AutoModelForSequenceClassification.from_pretrained("adam-chell/tweet-sentiment-analyzer")

# pushing the model to the device
model.to(device)

# we import the already preprocessed data
df_submission = pd.read_csv('./data/data_submission_preprocessed.csv')

# we add for a fake label value of 1 (neutral for this model), just in order to make the data BERT friendly
# if will have no influence in the predictions
df_submission['positive'] = 1

sub_texts = df_submission.tweet.values.tolist()
sub_labels = df_submission.positive.values

# we tokenize and fit the data into a model friendly type of input
sub_encodings = tokenizer(sub_texts, truncation=True, padding=True)
sub_dataset = Tweet(sub_encodings, sub_labels)
sub_loader = DataLoader(sub_dataset, batch_size=16, shuffle=False)

print("Starting to compute the predictions")
print("It might take long if you're not equiped whith a GPU ...")
predictions = []
for batch in sub_loader:
    # we first get the three inputs for the model
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    # the probabilities of prediction are contained in outputs[1]
    logits = outputs[1]
    # so we take the class with the maximal probability
    batch_predictions = torch.argmax(logits, dim=1).to(torch.device("cpu")).numpy().tolist()
    # add we append the predictions in order to create the full Series
    predictions = predictions + batch_predictions


df_submission['Prediction'] = predictions
# reminder : class '0' is negative, and '2' is positive, so we re translate in our rules
df_submission['Prediction'] = df_submission['Prediction'].apply(lambda x: 1 if x==2 else -1)
# renaming column to be accepted in the submission Platform 
df_submission = df_submission.rename(columns={'tweet_idx' : 'Id'})
# then we're ready to write out the file
df_submission[['Id', 'Prediction']].to_csv('submission.csv', index=False)
print("You are done !")






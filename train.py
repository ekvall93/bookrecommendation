import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import torch
from transformers import Trainer, TrainingArguments
import pickle
from transformers import BertForRanking
from rankingDataset import RankingDataset

train_dataset = pickle.load(open("./book_data/rankingDataset/train_dataset.pkl", "rb"))
val_dataset = pickle.load(open("./book_data/rankingDataset/val_dataset.pkl", "rb"))
#test_dataset = pickle.load(open("./book_data/rankingDataset/test_dataset.pkl", "rb"))


training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=30,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

model = BertForRanking.from_pretrained("bert-base-uncased")

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()

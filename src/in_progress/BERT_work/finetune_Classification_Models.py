# Perform my imports
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, AutoModelForSequenceClassification
import heapq
import json
import time
import random
import os
import re
import math
import pandas as pd
import sys
import numpy as np
from typing import Optional, List
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig, TrainingArguments, AutoModel, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, DataCollatorWithPadding
import json, csv
from sklearn.model_selection import train_test_split
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from peft import LoraConfig
import matplotlib.pyplot as plt
import evaluate

# Import from Other Locations
from python_functions.modify_datasets import *

# Specify the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pd.set_option('future.no_silent_downcasting', True)

# Initialize Data
# features to evaluate
evaluation_features = ['asthma', 'smoking', 'pneu', 'common_cold', 'pain', 'fever', 'antibiotics']

# Load in Validation Data 
df_train = pd.read_csv("clinical_data/train.csv")
df_train.reset_index(drop=True, inplace=True)

# Split Train Data again into train and eval (for testing pre-training)
df_train, df_train_eval = train_test_split(df_train, test_size = .125, random_state=0)
df_train.reset_index(drop=True, inplace=True)
df_train_eval.reset_index(drop=True, inplace=True)

# Specify Training Arguments
training_args = TrainingArguments(
    # output_dir="BERT_models/ClinBERT_Classification_checkpoints_tmp",
    output_dir="BERT_models/BERT_Classification_checkpoints_tmp",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

# Specify pre-trained model path
#model_path = "BERT_models/ClinBERT_pretrained"
model_path = "BERT_models/BERT_pretrained"

def tokenize_function(examples):
    # 'examples' is the dictionary passed by the 'map' function when batched=True
    # You must explicitly pass the list of texts from the 'advanced_text' column
    return tokenizer(examples['advanced_text'])

# Load the metric
acc_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    # The Trainer passes a tuple: (predictions, labels)
    logits, labels = eval_pred
    # Convert logits (raw scores) to class predictions (0 or 1)
    predictions = np.argmax(logits, axis=-1)
    # Compute the metrics
    acc = acc_metric.compute(predictions=predictions, references=labels)
    return {"accuracy": acc["accuracy"]}

# Iterate Through Each of Our Features 
for feature in evaluation_features:
    print(feature)
    
    # Get Temporary Datasets
    tmp_df_train = df_train[[feature,'advanced_text']]
    tmp_df_train_eval = df_train_eval[[feature,'advanced_text']]
    
    # Encode Our Data
    if feature != 'fever':
        # Perform Formatting of Feature Data
        tmp_df_train.loc[:, feature] = tmp_df_train[feature].replace({"no": 0, "yes": 1})
        tmp_df_train_eval.loc[:, feature] = tmp_df_train_eval[feature].replace({"no": 0, "yes": 1})
        
        # load Model Information
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path,device_map="auto", num_labels=2)
    else:
        # Perform Formatting of Feature Data
        tmp_df_train.loc[:, feature] = tmp_df_train[feature].replace({"no": 0, "low": 1, 'high':2})
        tmp_df_train_eval.loc[:, feature] = tmp_df_train_eval[feature].replace({"no": 0, "low": 1, 'high':2})
        
        # load Model Information
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path,device_map="auto", num_labels=3)
      

    # Tokenize Data
    tmp_df_train.rename(columns = {feature:'label'}, inplace = True)
    tmp_df_train_eval.rename(columns = {feature:'label'}, inplace = True)
    tmp_df_train_raw = Dataset.from_pandas(tmp_df_train[['advanced_text', 'label']])
    tmp_df_train_raw_tokenized = tmp_df_train_raw.map(tokenize_function,batched=True)
    tmp_df_train_eval_raw = Dataset.from_pandas(tmp_df_train_eval[['advanced_text', 'label']])
    tmp_df_train_eval_raw_tokenized = tmp_df_train_eval_raw.map(tokenize_function,batched=True)
    
    # Specify Info For Training
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tmp_df_train_raw_tokenized,
        eval_dataset=tmp_df_train_eval_raw_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Train Model
    trainer.train()
    
    # Save Model
    #final_model_path = f"BERT_models/ClinBERT_Classification_Models/{feature}"
    final_model_path = f"BERT_models/BERT_Classification_Models/{feature}"
    os.makedirs(final_model_path, exist_ok=True)
    trainer.save_model(final_model_path)
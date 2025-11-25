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

# Load in the Base ClinicalBERT Model
# model_path = "BERT_models/Clinical-BERT-Base"
model_path = "BERT_models/BERT-Base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path,device_map="auto")

# Prepare our Dataset for Pre-training
def tokenize_function(examples):
    # 'examples' is the dictionary passed by the 'map' function when batched=True
    # You must explicitly pass the list of texts from the 'advanced_text' column
    return tokenizer(examples['advanced_text'])
# Train Tokenize
df_train_raw = Dataset.from_pandas(df_train[['advanced_text']])
df_train_raw_tokenized = df_train_raw.map(tokenize_function,batched=True)
# Eval Tokenize
df_train_eval_raw = Dataset.from_pandas(df_train_eval[['advanced_text']])
df_train_eval_raw_tokenized = df_train_eval_raw.map(tokenize_function,batched=True)

# Set the training Arguments
training_args = TrainingArguments(
    # output_dir="BERT_models/Clinical_BERT_pretrained_checkpoints",
    output_dir="BERT_models/BERT_pretrained_checkpoints",
    num_train_epochs=10,                     
    per_device_train_batch_size=2,         
    per_device_eval_batch_size=8,
    learning_rate=2e-5,                     
    weight_decay=0.01,
    warmup_ratio=0.05,                     
    eval_strategy="epoch",            
    save_strategy="epoch",
    load_best_model_at_end=True,            
    fp16=True,                              # Enable mixed precision for speed
    logging_steps=100,
)

# Set the Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Set the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=df_train_raw_tokenized,
    eval_dataset=df_train_eval_raw_tokenized,
    data_collator=data_collator,
    tokenizer=tokenizer, # Pass the tokenizer for the Trainer to save
)

# Actually Train Model
trainer.train()

# Save the Model 
# final_model_path = "BERT_models/ClinBERT_pretrained"
final_model_path = "BERT_models/BERT_pretrained"
trainer.save_model(final_model_path)
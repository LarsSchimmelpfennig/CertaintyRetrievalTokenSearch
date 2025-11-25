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

# Load in Test Data 
df_test = pd.read_csv("clinical_data/test.csv")
df_test.reset_index(drop=True, inplace=True)

# Make Full Predictions
# Iterate through every feature 
for feature in evaluation_features:
    print(feature)
    # Load in Model
    # filepath = f"BERT_models/ClinBERT_Classification_Models/{feature}"
    filepath = f"BERT_models/BERT_Classification_Models/{feature}"
    tokenizer = AutoTokenizer.from_pretrained(filepath)
    model = AutoModelForSequenceClassification.from_pretrained(filepath)
    
    # Initialize Prediction Data
    prediction_df = df_test.copy()
    
    
    # Iterate Through Each Sample
    predicted_class_list = []
    probability_list = []
    for i in range(len(prediction_df)):
        if i % 50 == 0:
            print(i)
        # Generate Predictions
        inputs = tokenizer(prediction_df.loc[i,'advanced_text'], return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        
        # Get predicted Class and Probability
        predicted_class = logits.argmax().item()
        probabilities = F.softmax(logits, dim=1)
        top2_values, top2_indices = torch.topk(probabilities, k=2, dim=1)
        top2_diff = top2_values[:, 0] - top2_values[:, 1]

        # Add to Lists
        predicted_class_list.append(predicted_class)
        probability_list.append(top2_diff.cpu().float().item())
        
    # Map integers to the Class 
    if feature == 'fever':
        mapping = {0: 'no', 1: 'low', 2: 'high'}
    else:
        mapping = {0: 'no', 1: 'yes'}
    predicted_class_list = [mapping[x] for x in predicted_class_list]
        
    # Add to the dataframe
    df_test[f'LLM_{feature}'] = predicted_class_list
    df_test[f'{feature}_conf'] = probability_list
    
# Save the predictions
# df_test.to_csv("1k_ClinBERT_SimSum_Predictions.csv", index=False)
df_test.to_csv("1k_BERT_SimSum_Predictions.csv", index=False)
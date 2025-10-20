# Perform Imports
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import heapq
import json
import time
import random
import os
import re
import gc
import math
import pandas as pd
import sys
import numpy as np
from typing import Optional, List
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig, TrainingArguments
import json, csv
from sklearn.model_selection import train_test_split
from trl import SFTTrainer
from datasets import Dataset
from peft import LoraConfig

# Add the certs information
sys.path.append("..")
from CeRTS_beam_multi import *
from CeRTS_utils import *
sys.path.remove("..")

# Add some fine-tuning specific functions
from functions.generation_functions import *
from functions.dataset_creation_functions import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########### Initialize Important Model Information
# Specify models that we will test for (SFT models)
model_eval_list = ['models/llama-3.2-3b-instruct-SFT-10_16-merged', 'models/Qwen2.5-1.5B-Instruct-SFT-10_16-merged']
# Specify base information needed for some model functions
variables = ['asthma', 'smoking', 'pneu', 'common_cold', 'pain', 'fever', 'antibiotics']
json_template = {
        "asthma": "yes|no",
        "smoking": "yes|no",
        "pneu": "yes|no",            
        "common_cold": "yes|no",
        "pain": "yes|no",
        "fever": "high|low|no",
        "antibiotics": "yes|no"}
features = json_template.keys()

############ Load In Model Information
test_df = pd.read_csv("clinical_data/test.csv")
test_df.reset_index(drop=True, inplace = True)

############ Iterate through each model
# Iterate Through each model
for model_path in model_eval_list:
    
    # Load tokenizer and model 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    
    # Specify Output location and see if it already exists 
    OUT_PATH = f'results_data/CeRTS_SimSUM_1k_SFT_{model_path.split("/")[1]}.csv'
    
    # check if dataframe exists and create if not
    if os.path.exists(OUT_PATH):
        out_df = pd.read_csv(OUT_PATH)
    else:
        out_columns = ["compact_note", "asthma", "smoking", "pneu", "common_cold", "pain", "fever", "antibiotics", "LLM_asthma", "asthma_conf", "LLM_smoking", "smoking_conf", "LLM_pneu", "pneu_conf", "LLM_common_cold", "common_cold_conf", "LLM_pain", "pain_conf", "LLM_fever", "fever_conf", "LLM_antibiotics", "antibiotics_conf"]
        out_df = pd.DataFrame(columns = out_columns)
        out_df.to_csv(OUT_PATH)
    
    # Time Model 
    t1 = time.time()
    x = 0
    print('using file', OUT_PATH)
    
    # Open and save rows
    with open(OUT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for _, row in test_df.iterrows():
            #print(row)
            #print(row['text'])
            #print('advanced')
            #print(row['advanced_text'])

            x+=1
            if x % 25 == 0:
                print(x, round((time.time()-t1)/60, 2), 'mins')

            
            compact_note,asthma,smoking,pneu,common_cold,pain,fever,antibiotics = row[['advanced_text','asthma','smoking','pneu','common_cold','pain','fever','antibiotics']]

            messages = gen_prompt(compact_note, features, json_template)
            #print(messages)
            answer_distributions = CeRTS_output_dist(messages, features, model, tokenizer, device, beam_width=5, max_steps=100)
            d_feature_conf = {}
            row_extract_conf = []
            for feature, answer_dist in zip(features, answer_distributions):
                # print(feature, answer_dist)
                response = answer_dist[0][0]
                # print(answer_dist[0])
                confidence = top_2_delta([answer_score_tuple[1] for answer_score_tuple in answer_dist])
                # print('Feature:', feature, 'Response:', response, 'Confidence:', confidence)
                d_feature_conf[feature] = {'top_answer':response, 'confidence':confidence, 'dist':answer_dist}
                row_extract_conf.append(response)
                row_extract_conf.append(confidence)
                # print("-------------------")

            combined_row = [x] + [compact_note,asthma,smoking,pneu,common_cold,pain,fever,antibiotics] + row_extract_conf
            writer.writerow(combined_row)
    
    # Remove Data from torch
    del model
    del tokenizer
    
    torch.cuda.empty_cache()   # clears unused memory from the GPU cache
    torch.cuda.ipc_collect()   # optional: cleans up interprocess memory handles

    # Force garbage collection on CPU objects
    gc.collect()
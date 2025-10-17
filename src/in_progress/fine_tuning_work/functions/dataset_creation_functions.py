import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
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
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig
import json, csv
from sklearn.model_selection import train_test_split

sys.path.append("..")
from CeRTS_beam_multi import *
from CeRTS_utils import *
sys.path.remove("..")

from functions.generation_functions import *

def create_gold_standard_extractions(df, json_temp):
    '''
    This function will add a column of the gold standard extractions in the original format to the dataset.
    '''
    # Get list of correct dictionaries 
    correct_output_list = []
    for indx in df.index:
        tmp_dict = {}
        for key in json_temp.keys():
            tmp_dict[key] = df.loc[indx,key]
        correct_output_list.append(str(tmp_dict))
    
    # add our correct output to dataframe 
    df['correct_output'] = correct_output_list
    return df

def format_for_standard_SFT(df, template, zero_shot = False):
    "Given a dataframe of training data with columns 'advanced_text' and 'correct_output' this function will create a list of dicts to function as the training dataset."
    
    # Throw an error if we don't have the correct cols
    required_cols = set(["advanced_text","correct_output"])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Input DataFrame must contain columns {list(required_cols)}, "
            f"but is missing {list(missing)}")
        
    # Iterate through our df columns to create our training labels
    training_data =[]
    for i in range(len(df)):
        if zero_shot:
            tmp_train_prompt = gen_prompt_no_shot(df.loc[i, 'advanced_text'], template.keys(), template)
            tmp_train_prompt.append({'role':"assistant", 'content': df.loc[i, 'correct_output']})
            training_data.append(tmp_train_prompt)
        else:
            tmp_messages_prompt = gen_prompt(df.loc[i, 'advanced_text'], template.keys(), template)
            final_tmp_prompt = {'prompt':tmp_messages_prompt, 'completion':[{'role':"assistant", 'content': df.loc[i, 'correct_output']}]}
            training_data.append(final_tmp_prompt)
    return training_data
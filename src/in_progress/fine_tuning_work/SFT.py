# Perform my imports
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import heapq
import json
import time
import random
import os
import re
import math
import gc
import pandas as pd
import sys
import numpy as np
from typing import Optional, List
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig, TrainingArguments
import json, csv
from sklearn.model_selection import train_test_split
from trl import SFTTrainer, SFTConfig
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

######## LOAD IN DATA
df_train = pd.read_csv("clinical_data/train.csv")

######## Initialize Important Information
json_template = {
    "asthma": "yes|no",
    "smoking": "yes|no",
    "pneu": "yes|no",            
    "common_cold": "yes|no",
    "pain": "yes|no",
    "fever": "high|low|no",
    "antibiotics": "yes|no"
}
features = json_template.keys()
model_list = ["models/llama-3.2-3b-instruct", "models/Qwen2.5-1.5B-Instruct"]

# Format Dataset for fine_tuning
df_train = create_gold_standard_extractions(df_train, json_template)
train_data = format_for_standard_SFT(df_train, json_template)
# train_data = [
#     {"messages": conv} for conv in train_data   
# ]
train_data = Dataset.from_list(train_data)

####### Iterate through our model list 
for model_path in model_list:

    # load tokenizer/model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto" 
    )
    
    # Define Training Arguments
    peft_config = LoraConfig(
    r=8,                        # low-rank dimension
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
    )

    # ---- Step 4: Training arguments ----
    training_args = SFTConfig(
        output_dir=f'{model_path}-SFT-10_16',
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        completion_only_loss = True
    )

    # ---- Step 5: Trainer with PEFT/LoRA ----
    # updated_data = [
    #     {"messages": conv} for conv in train_data   # conv is your list of role/content dicts
    # ]
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        processing_class=tokenizer,
        args=training_args,
        peft_config=peft_config   # <<< LoRA here
    )
    
    #### TRAIN AND SAVE MODEL
    trainer.train()

    # Then Save
    trainer.model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"✅ LoRA adapter saved to {training_args.output_dir}")

    # Save merged Model (so don't have to marge and unload when using later)
    merged_model = trainer.model.merge_and_unload()
    final_model_dir = f"{training_args.output_dir}-merged"
    merged_model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    # Remove IData from Torch
    del trainer
    del model
    del merged_model
    del tokenizer

    torch.cuda.empty_cache()   # clears unused memory from the GPU cache
    torch.cuda.ipc_collect()   # optional: cleans up interprocess memory handles

    # Force garbage collection on CPU objects
    gc.collect()
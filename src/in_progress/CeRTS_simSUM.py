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
import numpy as np
from typing import Optional, List
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig
import json, csv

from CeRTS_beam_multi import *
from CeRTS_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# bnb_config = BitsAndBytesConfig(
#     load_in_32bit=True,
# )

#model_id = 'tiiuae/Falcon3-3B-Instruct'
#model_id = 'meta-llama/Llama-3.2-3B-Instruct'
#model_id = 'Qwen/Qwen2.5-7B-Instruct'
model_id = 'Qwen/Qwen2.5-3B-Instruct'
#model_id = 'meta-llama/Llama-3.1-8B-Instruct'
print(model_id)

model_name = model_id.split('/')[-1]

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_id, token = os.getenv("HF_TOKEN"))
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    #quantization_config = bnb_config,
    token = os.getenv("HF_TOKEN")
)

df = pd.read_csv('data/SimSUM.csv', sep=';')
#print(df.head(2))
#print(df.keys())

out_df = pd.read_csv(f'data/CeRTS_SimSUM_{model_name}.csv')
df = df[~df['advanced_text'].isin(out_df['compact_note'])]
print(len(df))
print(type(df))

variables = ['asthma', 'smoking', 'pneu', 'common_cold', 'pain', 'fever', 'antibiotics', 'days_at_home']

json_template = {
  "asthma": "yes|no",
  "smoking": "yes|no",
  "pneu": "yes|no",            # pneumonia
  "common_cold": "yes|no",
  "pain": "yes|no",
  "fever": "high|low|none",
  "antibiotics": "yes|no",
  "days_at_home": "INSERT INTEGER"   # e.g., 0, 1, 2, ...
}

# LLM_asthma,asthma_conf,LLM_smoking,smoking_conf,LLM_pneu,pneu_conf,LLM_common_cold,
#                          common_cold_conf,LLM_pain,pain_conf,LLM_fever,fever_conf,LLM_antibiotics,antibiotics_conf,
#                          LLM_days_at_home,days_at_home_conf

features = json_template.keys()


# messages = [
#         {
#             "role": "system",
#             "content": (
#                 "You are an information extraction function. "
#                 "Return EXACTLY one JSON object and nothing else (no prose, no code fences). "
#                 "Use these keys and ONLY these values:\n"
#                 "- asthma: yes|no\n"
#                 "- smoking: yes|no\n"
#                 "- pneu: yes|no\n"
#                 "- common_cold: yes|no\n"
#                 "- pain: yes|no\n"
#                 "- fever: high|low|none\n"
#                 "- antibiotics: yes|no\n"
#                 "- days_at_home: non-negative integer (0,1,2,...) WITHOUT quotes\n"
#                 "Do not add, remove, or rename keys. Use lowercase literals exactly as shown."
#             )
#         },
#         {
#             "role": "user",
#             "content": (
#                 f"Extract the following fields: {', '.join(features)}. "
#                 f"Output ONLY a single JSON object that matches this exact schema: {json.dumps(json_template)}. "
#                 f"Text: {text}"
#             )
#         }
#     ]


def gen_prompt(text, features, json_template):

    messages = [
        {
            "role": "system",
            "content": (
                "You are an information extraction function.\n"
                "Return EXACTLY one JSON object and nothing else (no prose, no code fences).\n"
                "All boolean-like fields MUST be STRING LITERALS 'yes' or 'no' (never 1/0, never true/false).\n"
                "Use lowercase literals exactly as shown. Do not add, remove, or rename keys.\n"
                "Schema (types and enums are STRICT):\n"
                '{'
                '"type":"object","additionalProperties":false,"properties":{'
                    '"asthma":{"type":"string","enum":["yes","no"]},'
                    '"smoking":{"type":"string","enum":["yes","no"]},'
                    '"pneu":{"type":"string","enum":["yes","no"]},'
                    '"common_cold":{"type":"string","enum":["yes","no"]},'
                    '"pain":{"type":"string","enum":["yes","no"]},'
                    '"fever":{"type":"string","enum":["high","low","none"]},'
                    '"antibiotics":{"type":"string","enum":["yes","no"]},'
                    '"days_at_home":{"type":"integer","minimum":0}'
                '}}'
            )
        },
        # Tiny few-shot to anchor 'yes'/'no' (numeric phrasing → strings)
        {
            "role": "user",
            "content": (
                "Extract fields from: 'Pneumonia (1), antibiotics started; fever low; stayed home 2 days.'"
            )
        },
        {
            "role": "assistant",
            "content": '{"asthma":"no","smoking":"no","pneu":"yes","common_cold":"no","pain":"no","fever":"low","antibiotics":"yes","days_at_home":2}'
        },
        {
            "role": "user",
            "content": (
                f"Now extract the following fields: {', '.join(features)}.\n"
                f"Output ONLY a single JSON object that matches this exact schema: "
                f"{json.dumps(json_template)}\n"
                "Replace placeholders with the allowed enum values above; for days_at_home output a non-negative integer (no quotes).\n"
                "Never use 1/0 or true/false for boolean-like fields—always 'yes' or 'no'.\n\n"
                f"Text: {text}"
            )
        }
    ]
    

    return messages 
                #"Do not include any text before or after the JSON.\n\n"
                # "Replace each placeholder with one of the allowed values above; for days_at_home replace "
                # "'INSERT INTEGER' with an integer >= 0 (no quotes). "

t1 = time.time()
x = 0
with open(f'data/CeRTS_SimSUM_{model_name}.csv', "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for _, row in df.iterrows():
        #print(row)
        #print(row['text'])
        #print('advanced')
        #print(row['advanced_text'])

        x+=1
        if x % 50:
            print(x, (time.time()-t1)/60, 'mins')

        compact_note,asthma,smoking,pneu,common_cold,pain,fever,antibiotics,days_at_home = row[['advanced_text','asthma','smoking','pneu','common_cold','pain','fever','antibiotics','days_at_home']]

        messages = gen_prompt(compact_note, features, json_template)
        #print(messages)
        answer_distributions = CeRTS_output_dist(messages, features, model, tokenizer, device, beam_width=5, max_steps=100)
        d_feature_conf = {}
        row_extract_conf = []
        for feature, answer_dist in zip(features, answer_distributions):
            print(feature, answer_dist)
            response = answer_dist[0][0]
            #print(answer_dist[0])
            confidence = top_2_delta([answer_score_tuple[1] for answer_score_tuple in answer_dist])
            print('Feature:', feature, 'Response:', response, 'Confidence:', confidence)
            d_feature_conf[feature] = {'top_answer':response, 'confidence':confidence, 'dist':answer_dist}
            row_extract_conf.append(response)
            row_extract_conf.append(confidence)

        combined_row = [compact_note,asthma,smoking,pneu,common_cold,pain,fever,antibiotics,days_at_home] + row_extract_conf

        writer.writerow(combined_row)
        
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
from sklearn.model_selection import train_test_split

from CeRTS_beam_multi import *
from CeRTS_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

print('updated test')

# bnb_config = BitsAndBytesConfig(
#     load_in_32bit=True,
# )

#model_id = 'Qwen/Qwen2.5-7B-Instruct'
#model_id = 'meta-llama/Llama-3.1-8B-Instruct'

#model_id = 'tiiuae/Falcon3-3B-Instruct'
#model_id = 'meta-llama/Llama-3.2-1B-Instruct'
#model_id = 'tiiuae/Falcon3-1B-Instruct'
#model_id = 'Qwen/Qwen2.5-3B-Instruct'

#model_id = 'Qwen/Qwen2.5-1.5B-Instruct'
#model_id = 'meta-llama/Llama-3.2-3B-Instruct'

for model_id in ['meta-llama/Llama-3.2-3B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct']:

    model_name = model_id.split('/')[-1]

    print(model_id)

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token = os.getenv("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        #quantization_config = bnb_config,
        token = os.getenv("HF_TOKEN")
    )

    df = pd.read_csv('data/SimSUM_updated.csv')

    train_df, temp_df = train_test_split(df,test_size=0.20,shuffle=True,random_state=1)

    val_df, test_df = train_test_split(temp_df,test_size=0.50,shuffle=True,random_state=1)

    val_prompt_100_df, val_final_df = train_test_split(
        val_df,
        train_size=100,
        shuffle=True,
        random_state=1,
    )

    print(len(train_df), len(test_df), len(val_final_df), len(val_prompt_100_df))
    #print(df.head(2))
    #print(df.keys())

    print('running 1k test', len(test_df))

    OUT_PATH = f'data/CeRTS_SimSUM_1k_test_split_{model_name}.csv'

    out_df = pd.read_csv(OUT_PATH)
    print(len(out_df))
    #df = df[~df['advanced_text'].isin(out_df['compact_note'])]
    #print(len(df))

    #df = df[~df['advanced_text'].isin(prompt_tuning_set['compact_note'])]

    variables = ['asthma', 'smoking', 'pneu', 'common_cold', 'pain', 'fever', 'antibiotics']

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


    def gen_prompt(text, features, json_template):

        example = """Pt reports frequent unexplained nausea and occasional vomiting x3 weeks. No fever, cough, dyspnea, blocked nose, or chest pain. No travel, diet, or med changes. No sick contacts. No wt loss; slight dehydration noted despite ↑fluid intake. Pt indicates recent ↑stress.

    **Physical Examination**
    Pt alert, well-hydrated; mucous membranes slightly dry. BP 115/75, HR 68 regular. Cardiac exam, lungs CTA. Abd soft, slightly tender in epigastric region, no rebound/guarding/masses/organomegaly. BS normal. Mild epigastric discomfort on palpation."""

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
                        '"fever":{"type":"string","enum":["high","low","no"]},'
                        '"antibiotics":{"type":"string","enum":["yes","no"]},'
                    '}}'
                )
            },
            # Tiny few-shot to anchor 'yes'/'no' (numeric phrasing → strings)
            {
                "role": "user",
                "content": (
                    f"Text: {example}"
                )
            },
            {
                "role": "assistant",
                "content": '{"asthma":"no","smoking":"no","pneu":"no","common_cold":"no","pain":"no","fever":"no","antibiotics":"yes"}'
            },
            {
                "role": "user",
                "content": (
                    f"Now extract the following fields: {', '.join(features)}.\n"
                    f"Output ONLY a single JSON object that matches this exact schema: "
                    f"{json.dumps(json_template)}\n"
                    "Replace placeholders with the allowed enum values above.\n"
                    "Never use 1/0 or true/false for boolean-like fields—always 'yes' or 'no'.\n"
                    "If fever is not mentioned, set fever to 'no'.\n\n"
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
    print('using file', OUT_PATH)
    with open(OUT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for _, row in test_df.iterrows():
            #print(row)
            #print(row['text'])
            #print('advanced')
            #print(row['advanced_text'])

            x+=1
            if x % 50 == 0:
                print(x, round((time.time()-t1)/60, 2), 'mins')

            
            compact_note,asthma,smoking,pneu,common_cold,pain,fever,antibiotics = row[['advanced_text','asthma','smoking','pneu','common_cold','pain','fever','antibiotics']]

            messages = gen_prompt(compact_note, features, json_template)
            #print(messages)
            answer_distributions = CeRTS_output_dist(messages, features, model, tokenizer, device, beam_width=5, max_steps=100)
            d_feature_conf = {}
            row_extract_conf = []
            for feature, answer_dist in zip(features, answer_distributions):
                #print(feature, answer_dist)
                response = answer_dist[0][0]
                #print(answer_dist[0])
                confidence = top_2_delta([answer_score_tuple[1] for answer_score_tuple in answer_dist])
                #print('Feature:', feature, 'Response:', response, 'Confidence:', confidence)
                d_feature_conf[feature] = {'top_answer':response, 'confidence':confidence, 'dist':answer_dist}
                row_extract_conf.append(response)
                row_extract_conf.append(confidence)

            combined_row = [compact_note,asthma,smoking,pneu,common_cold,pain,fever,antibiotics] + row_extract_conf

            writer.writerow(combined_row)
        
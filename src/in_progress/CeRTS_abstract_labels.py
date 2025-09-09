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

def gen_prompt(abstract, title, MP, disease, json_template, features):
    messages = [
        {
            "role": "system",
            "content": (
                f"You are an NLP tool. Extract the following information from PubMed abstracts: {','.join(features)}."
                
                "**Definition of CIViC Evidence Types**\n"
                "• Predictive – The variant indicates likely response or resistance to a specific therapy.\n"
                "• Diagnostic – Evidence pertains to a variant’s impact on patient diagnosis (cancer subtype).\n"
                "• Prognostic – Evidence pertains to a variant’s impact on disease progression, severity, or patient survival regardless of therapy.\n"
                "• Predisposing – Evidence pertains to a germline molecular profile’s role in conferring susceptibility to disease (including pathogenicity evaluations).\n"
                "• Oncogenic – Evidence pertains to a somatic variant’s involvement in tumor pathogenesis as described by the Hallmarks of Cancer.\n"
                "• Functional – Evidence pertains to a variant that alters biological function from the reference state.\n\n"

                "**Definition of CIViC Evidence Levels**\n"
                "• A – Validated association – Proven/consensus link in human medicine; established in practice or major trials.\n"
                "• B – Clinical evidence – Human clinical data (trials/observational cohorts) in multiple patients supports the link.\n"
                "• C – Case study – Single or few patient reports support the link.\n"
                "• D – Preclinical evidence – In vitro/in vivo model data supports the link (cells, animals, assays).\n"
                "• E – Inferential association – Indirect/associative evidence suggests a link (one step removed from a direct clinical association).\n"
            )
        },
        {
            "role": "user",
            "content": (
                f"Extract the following CIViC information from the text: {','.join(features)}. "
                f"""If the information is not available, output "NA" for that field. """
                f"Output rules:\n"
                f"""• Evidence Type must be exactly one of "Predictive","Diagnostic","Prognostic","Predisposing","Oncogenic","Functional","NA".\n"""
                f"""• Evidence Level must be exactly one of "A","B","C","D","E","NA" (uppercase string).\n"""
                f"• Provide the result **only** as a key-value pair with no extra text, using this JSON schema: {json.dumps(json_template)}.\n"
                f"""• Strings must be surrounded by "".\n"""
                f"Gene Variant: {MP}, Cancer Type: {disease}\n\n"
                f"Title: {title}\n"
                f"PubMed Abstract: {abstract}"
            )
        }
    ]

    return messages


FIELDNAMES = [
    "index", "pmid","PMCID","status","evidenceType","evidenceLevel",
    "molecularProfile_name","disease_name","title","abstract","pred_level","level_confidence","level_dist","pred_type","type_confidence","type_dist"
]


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    bnb_config = BitsAndBytesConfig(
        load_in_32bit=True,
    )

    #model_id = 'Qwen/Qwen2.5-7B-Instruct'
    model_id = 'meta-llama/Llama-3.1-8B-Instruct'
    print(model_id)

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token = os.getenv("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config = bnb_config,
        token = os.getenv("HF_TOKEN")
    )

    json_template = {"Evidence Type": "INSERT EVIDENCE TYPE", "Evidence Level": "INSERT EVIDENCE LEVEL"}
    features = json_template.keys()

    df = pd.read_csv('data/pubmed_title_abstract.csv').fillna("")
    print(len(df))

    df["title"] = df["title"].str.strip()
    df["abstract"] = df["abstract"].str.strip()
    df = df[(df["title"] != "") & (df["abstract"] != "")]

    out_df = pd.read_csv('data/CeRTS_type_level.csv')
    df = df[~df['index'].isin(out_df['index'])]
    df = df[df['status']=='ACCEPTED']
    
    print(len(df))

    sample_df = df.sample(n=10, random_state=1)

    with open('data/CeRTS_type_level.csv', "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        for _, row in sample_df.iterrows():

            MP, disease, title, abstract = row[['molecularProfile_name','disease_name','title','abstract']].values
            print(MP)

            messages = gen_prompt(abstract, title, MP, disease, json_template, features)
            #print(messages)
            answer_distributions = CeRTS_output_dist(messages, features, model, tokenizer, device, beam_width=5, max_steps=100)
            d_feature_conf = {}
            for feature, answer_dist in zip(features, answer_distributions):
                print(feature, answer_dist)
                response = answer_dist[0][0]
                #print(answer_dist[0])
                confidence = top_2_delta([answer_score_tuple[1] for answer_score_tuple in answer_dist])
                print('Feature:', feature, 'Response:', response, 'Confidence:', confidence)
                d_feature_conf[feature] = {'top_answer':response, 'confidence':confidence, 'dist':answer_dist}
            
            #print(row)            
            row_dict = row.to_dict()
            row_dict.update({
                "pred_level": d_feature_conf["Evidence Level"]["top_answer"],
                "level_confidence": d_feature_conf["Evidence Level"]["confidence"],
                "level_dist": d_feature_conf["Evidence Level"]["dist"],
                "pred_type": d_feature_conf["Evidence Type"]["top_answer"],
                "type_confidence": d_feature_conf["Evidence Type"]["confidence"],
                "type_dist": d_feature_conf["Evidence Type"]["dist"],
            })

            writer.writerow(row_dict)

    
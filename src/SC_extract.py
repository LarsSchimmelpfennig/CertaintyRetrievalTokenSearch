import torch
import numpy as np
from typing import Optional, List
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig, AutoModelForCausalLM
import csv
import heapq
import pandas as pd
import json
import time
import random
import os
import re
import math
import gc

#torch.backends.cudnn.deterministic = False
#torch.use_deterministic_algorithms(False)
# torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print('device count', torch.cuda.device_count())

def extract_first_json(text):
    # Regex pattern to capture the first JSON block
    json_pattern = re.compile(r"\{.*?\}", re.DOTALL)
    
    match = json_pattern.search(text)  # Find first match
    if match:
        try:
            return json.loads(match.group().strip())  # Parse and return valid JSON
        except json.JSONDecodeError:
            return None  # Handle invalid JSON
    return None  # No JSON found

def generate_completion(text, model, tokenizer, temperature=1, max_tokens=10000, truncate=False):
    inputs = tokenizer.encode(
            text,
            return_tensors='pt',
            add_special_tokens=True
    ).to(model.device)

    outputs = model.generate(
        inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True, output_scores=True
    )
    
    # Cut off the first tokens (the input text).
    if truncate:
        outputs['sequences'] = outputs['sequences'][:, inputs.shape[-1]:]
    
    # Decode the generated text.
    completion_text = tokenizer.decode(outputs['sequences'][0], skip_special_tokens=True, clean_up_tokenization_space=True)
    #print(f'Generated completion: {completion_text}')

    return completion_text


def gen_prompt(text, prompt_idx, feature, feature_json_str, feature_json_str_missing):
    prompts = [
    f"""
    ### System
    As an NLP tool, extract the following information from the provided text: {feature}.

    ### User
    Extract the following information from the following text: {feature}.  
    If the information is not available, only output {feature_json_str_missing} and nothing else.  
    Provide output **only as key:value pairs**, and do not include any additional text.  

    Provide output in the following JSON format:  
    {feature_json_str}  

    **Text:** {text}

    ### Assistant
    """, 

    #No missing
    f"""
    ### System
    As an NLP tool, extract the following information from the provided text: {feature}.

    ### User
    Extract the following information from the following text: {feature}. 
    Provide output **only as key:value pairs**, and do not include any additional text.

    Provide output in the following JSON format:
    {feature_json_str}

    **Text:** {text}

    ### Assistant
    """,

    f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    As an NLP tool, extract the following information from the following text: {feature}<|im_end|>
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Extract the following information from the following text: {feature} If the information is not available, only output {feature_json_str_missing} and not anything else. Provide output only in a key:value pair, and do not include any additional text. Provide output in the following JSON format: {feature_json_str}. Text:{text} <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """, 

    #No missing
    f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    As an NLP tool, extract the following information from the following text: {feature}<|im_end|>
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Extract the following information from the following text: {feature}. Provide output only in a key:value pair, and do not include any additional text. Provide output in the following JSON format: {feature_json_str}. Text:{text} <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
    ]

    return prompts[prompt_idx]


def uncertainty_prompt(feature, answers, prompt_idx):
    prompts = [f"""
    ### System
    You are an expert physician, asked to rate the uncertainty of an answer.

    ### Prompt
    You have extracted the following information from a clinical note: {feature}

    ### User
    Identify the most common response from the answers and count how many times it occured. Return this information with this format: {{"Response": "INSERT RESPONSE", "Count": "INSERT COUNT"}}
    Provide output **only as key:value pairs**, and do not include any additional text.
    
    Example for extracting blood pressure:
        Example Responses: ["120/80", "110/90", "120/80"]
        Example Output: {{"Response": "120/80", "Count": "2"}}

    Provide output only in a key:value pair, and do not include any additional text. Provide output in the following JSON format: {{"Response": "INSERT RESPONSE", "Count": "INSERT COUNT"}}

    Responses: {json.dumps(answers)}
    ### Assistant
    """,

    f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an expert physician, asked to rate the uncertainty of an answer.

    <|eot_id|><|start_header_id|>user<|end_header_id|>
    ### Prompt
    You have extracted the following information from a clinical note: {feature}

    Identify the most common response from the answers and count how many times it occured. If multiple responses occur equally, return the first one. Return this information with this format: {{"Response": "INSERT RESPONSE", "Count": "INSERT COUNT"}}
    Provide output **only as key:value pairs**, and do not include any additional text.
    
    Example for extracting blood pressure:
        Example Responses: ["120/80", "110/90", "120/80"]
        Example Output: {{"Response": "120/80", "Count": "2"}}

    Provide output only in a key:value pair, and do not include any additional text. Provide output in the following JSON format: {{"Response": "INSERT RESPONSE", "Count": "INSERT COUNT"}}

    Responses: {json.dumps(answers)}
    <|start_header_id|>assistant<|end_header_id|>
    """
    ]

    return prompts[prompt_idx]



df = pd.read_csv('data/final_notes_for_annotation.csv')

#Does making the search more exhaustive improve the calibration of the output distirbution
#Llama needs the extra special tokens in the input prompt

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    #llm_int8_enable_fp32_cpu_offload=True
)

replicates = 15
#'Qwen/Qwen2.5-7B-Instruct-1M'
#'NousResearch/Hermes-3-Llama-3.1-8B'
# 'google/gemma-2-9b-it'
# 'meta-llama/Llama-3.1-8B-Instruct'
#'mistralai/Mistral-Small-24B-Instruct-2501'

#Still need to run
#'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'microsoft/Phi-4-mini-instruct'
#'mistralai/Mixtral-8x7B-Instruct-v0.1', 'microsoft/Phi-4-mini-instruct'
print('running updated')
for model_id in ['Qwen/Qwen3-8B']:

    tokenizer = None
    model = None
    print("Loading model", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token = os.getenv("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        #device_map="auto",
        device_map="balanced",
        #device_map="balanced_low_0",
        quantization_config = bnb_config,
        token = os.getenv("HF_TOKEN"),
        trust_remote_code=True
    )

    print('eos token', tokenizer.eos_token_id)

    g = torch.Generator('cuda')
    terminators = [tokenizer.eos_token_id]
    temperature=1

    feature_space = [
        ('Mention of lung cancer/carcinoma', """{"Mention of lung cancer/carcinoma": "YES" or "NO"}""", None, """{"Mention of lung cancer/carcinoma": "YES"}"""),
        ("Number of discharge medications", """{"Number of discharge medications": "INSERT NUMBER OF DISCHARGE MEDICATIONS"}""", """{"Number of discharge medications": "0"}""", """{"Number of discharge medications": "0"}"""),
        ('Age', """{"Age": "INSERT AGE NUMBER"}""", """{"Age": "Not Available"}""", """{"Age": "56"}"""),
        ('First treatment date', """{"First treatment date": "INSERT FIRST TREATMENT DATE as M/D/YYYY"}""", """{"First treatment date": "Not Available"}""", """{"First treatment date": "INSERT FIRST TREATMENT DATE as 02/28/2002"}"""),
        #('Prescribed medications', """{"Prescribed medications": "YES" or "NO"}""", None, """{"Prescribed medications": "YES"}"""),
        ('Blood pressure value at discharge', """{"Blood pressure value at discharge": "INSERT BLOOD PRESSURE VALUE AT DISCHARGE as SYSTOLIC PRESSURE/DIASTOLIC PRESSURE"}""", """{"Blood pressure value at discharge": "Not Available"}""", """{"Blood pressure value at discharge": "120/80"}"""),
        ('Treated with immunotherapy', """{"Treated with immunotherapy": "YES" or "NO"}""", None, """{"Treated with immunotherapy": "YES"}""")
    ]

    for feature, feature_json_str, feature_json_str_missing, example_json in feature_space:

        if ('Mixtral' in model_id and feature not in ['Treated with immunotherapy']) or ('Phi' in model_id and feature not in ["Number of discharge medications"]):
            continue
        
        if 'Llama' in model_id or 'Qwen' in model_id or 'Phi' in model_id:
            if feature_json_str_missing is None:
                prompt_idx = 3
            else:
                prompt_idx = 2
        else:
            if feature_json_str_missing is None:
                prompt_idx = 1
            else:
                prompt_idx = 0

        if 'Llama' in model_id or 'Qwen' in model_id or 'Phi' in model_id:
            uncertainty_prompt_idx = 1
        else:
            uncertainty_prompt_idx = 0
        
        filename = f"data/other_confidence_methods/extract_{feature.replace('/', '_')}_SC_LLM_Annot_{model_id.split('/')[1]}.csv"
        t1 = time.time()

        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['patient_id', 'prompt', 'temp', 'output', 'confidence'])
            for idx, (patient_id, note) in enumerate(df[['EPIC_MRN', 'NOTE_TEXT']].values):
                responses = []
                for _ in range(replicates):
                    text = gen_prompt(note, prompt_idx, feature, feature_json_str, feature_json_str_missing)
                    torch.cuda.empty_cache()
                    gc.collect()
                    output = generate_completion(text, model, tokenizer, truncate=True)
                    json_response = extract_first_json(output)
                    if json_response is None:
                        answer = 'missing'
                    else:
                        answer = json_response.get(feature)
                    responses.append(answer)

                #print(responses)
                uncertainty_text = uncertainty_prompt(feature, responses, uncertainty_prompt_idx)
                #print(uncertainty_text)
                uncertainty_output = generate_completion(uncertainty_text, model, tokenizer, truncate=True)
                #print(uncertainty_output)
                json_uncertainty_response = extract_first_json(uncertainty_output)
                #print(json_uncertainty_response)
                if json_uncertainty_response is None:
                    main_response = responses[0]
                    confidence = 'missing'
                else:
                    main_response = json_uncertainty_response.get("Response")
                    count = json_uncertainty_response.get("Count")
                    if count is None or (type(count)!=int and not count.isnumeric()):
                        confidence = 'missing'
                    else:
                        confidence = int(count) / replicates
                    print(main_response, count, confidence)
                
                csvwriter.writerow([patient_id, prompt_idx, temperature, main_response, confidence])
                print(model_id, feature, idx, (time.time() - t1) / 60, 'mins')
        print('done in', (time.time() - t1) / 60, 'mins')

    del model
    torch.cuda.empty_cache()
    gc.collect()
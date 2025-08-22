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
from collections import defaultdict

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

from transformers import logging           # must be done *before* the first HF import
logging.set_verbosity_error()   

print('current')
#torch.backends.cudnn.deterministic = False
#torch.use_deterministic_algorithms(False)
# torch.backends.cudnn.benchmark = True

#NOTE top_p = 1 is used for all files.

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

def structured_exploration(model, tokenizer, net_inputs, prob_threshold=0.05, max_steps=100):
    """
    Systematically explores multiple token paths using top-k filtering and a probability threshold.
    Returns all completed sequences with their probabilities after the ': ' tokens.
    """
    model.config.use_cache = False
    log_prob_threshold = math.log(prob_threshold)
    
    #todo consider using logliklihood instead of probability for heap
    beams = [(0, random.random(), net_inputs, 1)]  # Priority queue of (score, prob, sequence, min_prob)
    #beams = torch.tensor([[0.0, *inputs.tolist()]], device=model.device)
    results = []  # Stores completed sequences with probabilities
    i = 0
    
    while len(beams) > 0 and i < max_steps:
        log_prob, _, seq, min_prob = heapq.heappop(beams)
        log_prob = -log_prob #convert back to negative
        #seq['sequences'] = seq['sequences'].to(model.device)
        #seq_tensor = torch.tensor(seq['sequences'], device=model.device) if isinstance(seq['sequences'], list) else seq['sequences']
        with torch.no_grad():
            outputs = model.generate(
                seq['sequences'],
                #past_key_values=None,#seq.get("past_key_values", None),
                max_new_tokens=1,
                pad_token_id=tokenizer.eos_token_id,
                top_p=1,
                #top_k=None,
                #attention_mask=seq_tensor.ne(tokenizer.pad_token_id),
                temperature=temperature,
                return_dict_in_generate=True, output_scores=True,
                do_sample=False,  # Controls randomness
                #use_cache=False
            )
        
        logits = outputs['scores'][0]
        logits.div_(temperature)

        # logits[sorted_indices[sorted_indices_to_remove]] = -float("Inf")  # Mask unlikely tokens
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs)

        # Compute new log probabilities in a vectorized manner
        new_log_probs = log_prob + log_probs[0]

        # Filter valid tokens based on the threshold
        valid_mask = new_log_probs >= log_prob_threshold
        valid_tokens = list(zip(new_log_probs[valid_mask].tolist(), torch.nonzero(valid_mask, as_tuple=True)[0].tolist()))

        # valid_tokens = []

        # for token_id, token_log_prob in enumerate(log_probs[0]):
        #     new_log_prob = log_prob + token_log_prob.item()
        #     if new_log_prob >= log_prob_threshold:
        #         valid_tokens.append((new_log_prob, token_id))

        for new_log_prob, token in valid_tokens:

            token_prob = math.exp(new_log_prob - log_prob)   # p_i = exp(ΔlogP)
            new_min_prob = min(min_prob, token_prob)
            new_seq = torch.cat([seq['sequences'], torch.tensor([[token]], device=seq['sequences'].device)], dim=-1)
            #new_seq = torch.cat(
            #    [seq['sequences'], torch.tensor([[token]], device=seq['sequences'].device)],
            #    dim=-1,
            #).cpu()
            
            #print(math.exp(new_log_prob), new_min_prob, tokenizer.decode(new_seq[0]).split('assistant')[1])
            # If EOS token is generated, store the result
            if token == tokenizer.eos_token_id:
                results.append((new_log_prob, new_min_prob, {'sequences': new_seq}))
            else:
                heapq.heappush(beams, (-new_log_prob, random.random(), {'sequences': new_seq}, new_min_prob)) #min heap, more negative log probs are less likley

            #if len(results) == 5:
            #    return i+1, sorted(results, key=lambda x: -x[0])
        
        i += 1
    
    print('num steps', i)
    return i, sorted(results, key=lambda x: x[0])

# def gen_prompt(text, prompt_idx, feature, feature_json_str, feature_json_str_missing):
#     system_message = {
#         "role": "system",
#         "content": (
#             f"You are an NLP tool. Extract the following information from text: {feature}."
#         ),
#     }

#     # Build the user message depending on whether we include the missing-value rule
#     if prompt_idx == 0:
#         user_content = (
#             f"Extract the following information from the text: {feature}. "
#             f"If the information is not available, output {feature_json_str_missing} and nothing else. "
#             f"Provide the result **only** as key-value pairs with no extra text, "
#             f"using this JSON schema: {feature_json_str}. "
#             f"Text: {text}"
#         )
#     else:  # prompt_idx == 1
#         user_content = (
#             f"Extract the following information from the text: {feature}. "
#             f"Provide the result **only** as key-value pairs with no extra text, "
#             f"using this JSON schema: {feature_json_str}. "
#             f"Text: {text}"
#         )

#     user_message = {"role": "user", "content": user_content}

#     # Minimal assistant stub; most models will overwrite this when generating
#     assistant_message = {"role": "assistant", "content": ""}

#     return [system_message, user_message, assistant_message]

# def deepseek_prompt(text, prompt_idx, feature, feature_json_str, feature_json_str_missing):
#     prompts = [
#         f"""
#         <|begin_of_text|><|start_header_id|>system<|end_header_id|>
#         As an NLP tool, extract the following information from the following text: {feature}<|im_end|>
#         <|eot_id|><|start_header_id|>user<|end_header_id|>
#         Extract the following information from the following text: {feature} If the information is not available, only output {feature_json_str_missing} and not anything else. Provide output only in a key:value pair, and do not include any additional text. Provide output in the following JSON format: {feature_json_str}. Text:{text} <|eot_id|>
#         <|start_header_id|>assistant<|end_header_id|>
#         """, 

#         #No missing
#         f"""
#         <|begin_of_text|><|start_header_id|>system<|end_header_id|>
#         As an NLP tool, extract the following information from the following text: {feature}<|im_end|>
#         <|eot_id|><|start_header_id|>user<|end_header_id|>
#         Extract the following information from the following text: {feature}. Provide output only in a key:value pair, and do not include any additional text. Provide output in the following JSON format: {feature_json_str}. Text:{text} <|eot_id|>
#         <|start_header_id|>assistant<|end_header_id|>
#         """
#     ]

#     return prompts[prompt_idx]

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


#'google/flan-t5-large'

df = pd.read_csv('data/final_notes_for_annotation.csv')

#Does making the search more exhaustive improve the calibration of the output distirbution
#Llama needs the extra special tokens in the input prompt

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

# def build_equal_map(m, n_gpus: int = 4):
#     """Return a dict that assigns ~equal numbers of layers to each GPU."""
#     n_layers = m.config.num_hidden_layers
#     per_gpu   = math.ceil(n_layers / n_gpus)

#     dmap = {"model.embed_tokens": 0}   # embeddings on GPU-0
#     layer_idx, gpu_idx = 0, 0
#     while layer_idx < n_layers:
#         dmap[f"model.layers.{layer_idx}"] = gpu_idx
#         layer_idx += 1
#         # bump the GPU when we've filled `per_gpu` layers
#         if layer_idx % per_gpu == 0 and gpu_idx < n_gpus - 1:
#             gpu_idx += 1

#     # final layers (norm & head) go on the last GPU we used
#     dmap["model.norm"]   = gpu_idx
#     dmap["lm_head"]      = gpu_idx
#     return dmap


#'google/flan-t5-large', 'mosaicml/mpt-7b-instruct' 'google/gemma-3-27b-it'
model_list = ['Qwen/Qwen2.5-7B-Instruct-1M', 'google/gemma-2-9b-it', 'meta-llama/Llama-3.1-8B-Instruct',
                  'microsoft/Phi-4-mini-instruct', 'mistralai/Mistral-Small-24B-Instruct-2501',
                    'NousResearch/Hermes-3-Llama-3.1-8B', 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'Qwen/Qwen3-8B', 'mistralai/Mixtral-8x7B-Instruct-v0.1']

#'mistralai/Mixtral-8x7B-Instruct-v0.1'

#from transformers import AutoProcessor, Gemma3ForConditionalGeneration

for model_id in model_list[-1:]:

    tokenizer = None
    model = None

    print("Loading model", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token = os.getenv("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="balanced",
        quantization_config = bnb_config,
        token = os.getenv("HF_TOKEN"),
        trust_remote_code=True,
        #max_memory={i: "39GiB" for i in range(4)}
    )

    # summary = defaultdict(list)
    # for name, device in model.hf_device_map.items():
    #     summary[device].append(name)

    # for gpu in sorted(summary):
    #     modules = summary[gpu]
    #     print(f"{gpu}: {len(modules):3d} sub-modules")
    #     # show the first few module names so you can spot-check
    #     for m in modules[:5]:
    #         print(f"    • {m}")
    #     if len(modules) > 5:
    #         print("    …")

    #print('eos token', tokenizer.eos_token_id)

    g = torch.Generator('cuda')
    terminators = [tokenizer.eos_token_id]
    temperature=1
    #top_k=None
    #top_p=1

    prob_threshold=0.05
    
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
        #if model_id in ['Qwen/Qwen2.5-7B-Instruct-1M', 'google/gemma-2-9b-it'] and feature != 'Treated with immunotherapy':
        #    continue

        if 'Llama' in model_id or 'Qwen2.5' in model_id:
            if feature_json_str_missing is None:
                prompt_idx = 3
            else:
                prompt_idx = 2
        else:
            if feature_json_str_missing is None:
                prompt_idx = 1
            else:
                prompt_idx = 0
            
        #print('prompt_idx', prompt_idx)
        
        filename = f"data/min_token_dist/{feature.replace('/', '_')}_{model_id.split('/')[1]}.csv"
        t1 = time.time()

        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['patient_id', 'top_outputs', 'top_probs', 'answer_min_token_prob', 'num_equivalent_runs'])
            for idx, (patient_id, note) in enumerate(df[['EPIC_MRN', 'NOTE_TEXT']].values):
                
                # messages = gen_prompt(note, prompt_idx, feature, feature_json_str, feature_json_str_missing)
                # if 'gemma' in model_id or 'deepseek' in model_id:
                #     #print('skipping system prompt')
                #     #remove the system prompt from gemma and deepseek
                #     messages = messages[1:]

                # input_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                # #print(input_prompt)

                # if 'deepseek' in model_id:
                #     input_prompt = deepseek_prompt(note, prompt_idx, feature, feature_json_str, feature_json_str_missing)
                # inputs = tokenizer.encode(input_prompt, return_tensors='pt', add_special_tokens=True).to(model.device)
                
                inputs = tokenizer.encode(gen_prompt(note, prompt_idx, feature, feature_json_str, feature_json_str_missing), return_tensors='pt', add_special_tokens=True).to(model.device)
                #decoded_prompt = tokenizer.decode(inputs[0], skip_special_tokens=False)
                #print(decoded_prompt)
                net_inputs = {'sequences': inputs, 'past_key_values': None}
                num_steps, results = structured_exploration(model, tokenizer, net_inputs, prob_threshold=prob_threshold, max_steps=100)
                #print('len results', len(results))
                d_val_probs = {}
                token_lengths = []
                if len(results) == 0:
                    first_min_token_prob = 0
                else:
                    first_min_token_prob = results[0][1]
                #print('first min token', first_min_token_prob)

                for log_prob, min_token_prob, tokens in results:
                    tokens['sequences'] = tokens['sequences'][:, inputs.shape[-1]:]
                    token_length = len(tokens['sequences'][0])
                    token_lengths.append(token_length)
                    json_str = tokenizer.decode(tokens['sequences'][0], skip_special_tokens=True, clean_up_tokenization_space=True).strip()
                    #print(math.exp(log_prob), min_token_prob, json_str)
                    data = extract_first_json(json_str)
                    if data is None:
                        #print('val is none')
                        val = "missing"
                    else:
                        val = str(data.get(feature))
                        if val in d_val_probs:
                            d_val_probs[val] += math.exp(log_prob)
                        else:
                            d_val_probs[val] = math.exp(log_prob)
                
                if len(token_lengths) == 0:
                    token_lengths = [len(tokenizer.encode(example_json, return_tensors='pt')[0])]
                
                avg_token_length = np.mean(token_lengths)
                equivalent_runs = num_steps/avg_token_length
                d_val_keys = list(d_val_probs.keys())
                d_val_values = [f'{value:.4f}' for value in d_val_probs.values()]
                print(model_id, prob_threshold, set(d_val_probs.keys()), 'Distribution sum', np.sum(list(d_val_probs.values())))
                #print('num steps', num_steps, 'avg token len', avg_token_length, 'equiv runs', equivalent_runs)
                csvwriter.writerow([patient_id, ','.join(d_val_keys), ','.join(d_val_values), first_min_token_prob, equivalent_runs])
                print(idx, feature, model_id, (time.time() - t1) / 60, 'mins')
        print('done in', (time.time() - t1) / 60, 'mins')

    del model
    torch.cuda.empty_cache()
    gc.collect()
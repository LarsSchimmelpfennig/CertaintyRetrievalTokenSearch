import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import heapq
import json
import time
import random
import os
import re
import math

def extract_first_json(text):
    """
    Extracts the first valid JSON object found in the given text.
    
    Args:
        text (str): The input text from which to extract the JSON object.
    
    Returns:
        dict or None: The extracted JSON object as a dictionary if valid, otherwise None.
    """
    json_pattern = re.compile(r"\{.*?\}", re.DOTALL)
    match = json_pattern.search(text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None

def structured_exploration(model, tokenizer, net_inputs, prob_threshold=0.05, max_steps=100, top_p=1):
    """
    Systematically explores multiple token paths using top-p filtering and a probability threshold.
    
    Args:
        model: The language model used for sequence generation.
        tokenizer: The tokenizer corresponding to the model for handling token encoding and decoding.
        net_inputs (dict): A dictionary containing the initial input sequence and optional past key values.
        prob_threshold (float, optional): The minimum probability threshold for considering a token. Default is 0.05.
        max_steps (int, optional): The maximum number of steps allowed for sequence exploration. Default is 100.
        top_p (float, optional): The nucleus sampling parameter that controls the cumulative probability of considered tokens. Default is 1.
    
    Returns:
        list of tuples: A sorted list of completed sequences and their log probabilities, where each tuple contains:
            - log probability of the sequence (float)
            - dictionary with key 'sequences' containing the generated token sequence (tensor)
    """
    log_prob_threshold = math.log(prob_threshold)
    
    
    beams = [(0, random.random(), net_inputs)] #random.random() Used as a tie-breaker so that net_inputs is never used for sorting.
    results = [] 
    i = 0
    
    while len(beams) > 0 and i < max_steps:
        log_prob, r, seq = heapq.heappop(beams)
        outputs = model.generate(
            seq['sequences'],
            past_key_values=seq.get("past_key_values"),
            max_new_tokens=1,
            pad_token_id=tokenizer.eos_token_id,
            temperature=temperature,
            top_k=top_k, 
            top_p=top_p,
            return_dict_in_generate=True, 
            output_scores=True,
            do_sample=False,
            return_legacy_cache=True
        )
        
        logits = outputs['scores'][0]
        logits.div_(temperature)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs)
        valid_tokens = []

        for token_id, token_log_prob in enumerate(log_probs[0]):
            new_log_prob = log_prob + token_log_prob.item()
            if new_log_prob >= log_prob_threshold:
                valid_tokens.append((new_log_prob, token_id))

        for new_log_prob, token in valid_tokens:
            new_seq = torch.cat([seq['sequences'], torch.tensor([[token]], device=seq['sequences'].device)], dim=-1)
            if token == tokenizer.eos_token_id:
                results.append((new_log_prob, {'sequences': new_seq}))
            elif new_log_prob >= log_prob_threshold:
                heapq.heappush(beams, (new_log_prob, random.random(), {'sequences': new_seq}))
        
        i += 1
    
    print('num steps', i)
    return sorted(results, key=lambda x: x[0])

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    model_id = 'meta-llama/Llama-3.1-8B-Instruct'

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token = os.getenv("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config = bnb_config,
        token = os.getenv("HF_TOKEN")
    )

    prob_threshold=0.05
    temperature=1
    top_k=None
    top_p=0.9
    t1 = time.time()

    feature = "Age"
    feature_json_str = """{"Age": "INSERT AGE NUMBER"}"""
    feature_json_str_missing = """{"Age": "Not Available"}"""

    text='The age of Rob is 7.'

    input_prompt = f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    As an NLP tool, extract the following information from the following text: {feature}<|im_end|>
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Extract the following information from the following text: {feature} If the information is not available, only output {feature_json_str_missing} and not anything else. Provide output only in a key:value pair, and do not include any additional text. Provide output in the following JSON format: {feature_json_str}. Text:{text} <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """

    inputs = tokenizer.encode(input_prompt, return_tensors='pt').to(model.device)
    net_inputs = {'sequences': inputs}
    results = structured_exploration(model, tokenizer, net_inputs, prob_threshold=prob_threshold)
    print('len results', len(results))
    d_val_probs = {} #Combines the probabilities of JSON's with the same value
    for log_prob, tokens in results:
        tokens['sequences'] = tokens['sequences'][:, inputs.shape[-1]:]
        json_str = tokenizer.decode(tokens['sequences'][0], skip_special_tokens=True, clean_up_tokenization_space=True).strip()
        data = extract_first_json(json_str)
        print(data)
        val = data.get(feature)
        if val is None:
            print('val is none')
            val = "missing"
        if val in d_val_probs:
            d_val_probs[val] += math.exp(log_prob)
        else:
            d_val_probs[val] = math.exp(log_prob)

    d_val_keys = list(d_val_probs.keys())
    d_val_values = [f'{value:.4f}' for value in d_val_probs.values()]
    print('done in', (time.time() - t1) / 60, 'mins')

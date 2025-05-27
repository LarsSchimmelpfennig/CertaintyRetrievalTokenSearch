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


def top_2_delta(output_probs):
    """
    Expectes array of sorted output probs.

    Returns confidence of the LLM by the difference in the probabilities of the top-2 output sequences.
    """
    if len(output_probs) > 1:
        return output_probs[0] - output_probs[1]
    return output_probs[0]


def structured_token_search(model, tokenizer, net_inputs, prob_threshold=0.05, max_steps=100):
    """
    Systematically explores multiple token paths using a minumum probability threshold and max inference steps.

    Args:
        model: The language model used for sequence generation.
        tokenizer: The tokenizer corresponding to the model for handling token encoding and decoding.
        net_inputs (dict): A dictionary containing the initial input sequence and optional past key values.
        prob_threshold (float, optional): The minimum probability threshold for considering a token. Default is 0.05.
        max_steps (int, optional): The maximum number of steps allowed for sequence exploration. Default is 100.

    Returns:
        list of tuples: A sorted list of completed sequences and their log probabilities, where each tuple contains:
            - log probability of the sequence (float)
            - dictionary with key 'sequences' containing the generated token sequence (tensor)
    """
    log_prob_threshold = math.log(prob_threshold)
    beams = [(0, random.random(), net_inputs)]  # Priority queue of (score, prob, sequence)
    results = []  # Stores completed sequences with probabilities
    i = 0
    
    while len(beams) > 0 and i < max_steps:
        log_prob, r, seq = heapq.heappop(beams)
        log_prob = -log_prob #convert back to negative
        outputs = model.generate(
            seq['sequences'],
            max_new_tokens=1,
            pad_token_id=tokenizer.eos_token_id,
            temperature=temperature,
            return_dict_in_generate=True, output_scores=True,
            do_sample=False, 
        )
        
        logits = outputs['scores'][0]
        logits.div_(temperature)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs)
        new_log_probs = log_prob + log_probs[0]
        valid_mask = new_log_probs >= log_prob_threshold
        valid_tokens = list(zip(new_log_probs[valid_mask].tolist(), torch.nonzero(valid_mask, as_tuple=True)[0].tolist()))

        for new_log_prob, token in valid_tokens:
            new_seq = torch.cat([seq['sequences'], torch.tensor([[token]], device=seq['sequences'].device)], dim=-1)
            if token == tokenizer.eos_token_id:
                results.append((new_log_prob, {'sequences': new_seq}))
            else:
                heapq.heappush(beams, (-new_log_prob, random.random(), {'sequences': new_seq})) #min heap, more negative log probs are less likley
        
        i += 1
    
    print('num steps', i)
    return i, sorted(results, key=lambda x: x[0])

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
    t1 = time.time()

    feature = "Age"
    feature_json_str = """{"Age": "INSERT AGE NUMBER"}"""
    feature_json_str_missing = """{"Age": "Not Available"}"""

    text='The age of Rob is 7.'

    messages = [
        {
            "role": "system",
            "content": (
                f"You are an NLP tool. Extract the following information from text: {feature}."
            )
        },
        {
            "role": "user",
            "content": (
                f"Extract the following information from the text: {feature}. "
                f"If the information is not available, output {feature_json_str_missing} and nothing else. "
                f"Provide the result **only** as a key-value pair with no extra text, using this JSON schema: {feature_json_str}. "
                f"Text: {text}"
            )
        }
    ]

    input_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(input_prompt, return_tensors='pt', add_special_tokens=True).to(model.device)
    net_inputs = {'sequences': inputs}
    results = structured_token_search(model, tokenizer, net_inputs, prob_threshold=prob_threshold)
    print('len results', len(results))
    d_val_probs = {} #Combines the probabilities of JSON's with the same value
    for log_prob, tokens in results:
        tokens['sequences'] = tokens['sequences'][:, inputs.shape[-1]:]
        json_str = tokenizer.decode(tokens['sequences'][0], skip_special_tokens=True, clean_up_tokenization_space=True).strip()
        data = extract_first_json(json_str)
        print(data)
        data = extract_first_json(json_str)
        if data is None:
            print('val is none')
            val = "missing"
        else:
            val = str(data.get(feature))
            if val in d_val_probs:
                d_val_probs[val] += math.exp(log_prob)
            else:
                d_val_probs[val] = math.exp(log_prob)


    print('done in', (time.time() - t1) / 60, 'mins')

    sorted_d_val_keys = [key for _, key in sorted(zip(d_val_probs.values(), d_val_probs.keys()), reverse=True)]
    response = sorted_d_val_keys[0]
    confidence = top_2_delta(sorted(d_val_probs.values()))

    print('Response:',response,'Confidence:',confidence)

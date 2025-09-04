import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import json
import time
import os
import math
import sys

from CeRTS_utils import *

#Keep track of the distirbution of the variable JSON region.
#When a comma is generated, provide the LLM with the start of the next JSON and then record again.

@torch.no_grad()
def multi_beam_search(model, tokenizer, net_inputs, beam_width=5, max_steps=100):
    """
    Beam search using PKV caching.
    - Accepts/returns PKV; never calls model._reorder_cache directly.
    - On step 0 (no cache), feeds full sequences; thereafter only last tokens + reordered cache.
    Returns: List[(log_prob, {'sequences': LongTensor[1, T+L], 'pkv': Cache|tuple|None})], best-first.
    """
    log_prob_threshold = math.log(0.001) # beam min prob threshold
    input_ids = net_inputs["sequences"]      # [1, T]

    base_len = input_ids.shape[-1]

    past = net_inputs.get("pkv", None)       # Cache|tuple|None
    device = input_ids.device
    eos_id = tokenizer.eos_token_id

    # Live beam state
    live_seqs = [input_ids]                               # list of [1, t_i]
    live_scores = torch.tensor([0.0], device=device)      # [B]
    finished = []                                         # (score, {'sequences':..., 'pkv':...})
    steps = 0


    while steps < max_steps and len(live_seqs) > 0:
        steps += 1
        #print('step', steps)

        # Forward
        if past is None:
            # Step 0: encode full prefixes for all live beams
            batch_ids = torch.cat(live_seqs, dim=0)       # [B, T]
            outputs = model(input_ids=batch_ids, use_cache=True)
        else:
            # Subsequent steps: feed only last token per beam with (reordered) cache
            last_tokens = torch.cat([seq[:, -1:] for seq in live_seqs], dim=0)  # [B, 1]
            outputs = model(input_ids=last_tokens, use_cache=True, past_key_values=past)

        logits = outputs.logits[:, -1, :]  # [B, V]
        log_probs = torch.log_softmax(logits, dim=-1)     # [B, V]
        current_past = outputs.past_key_values            # Cache|tuple

        valid_mask = log_probs >= log_prob_threshold    # EOS is also excluded here
        valid_counts = valid_mask.sum(dim=-1)      # [B]
        log_probs = torch.where(valid_mask, log_probs, torch.full_like(log_probs, float("-inf")))

        # Build candidates
        B, V = log_probs.shape
        #print('shape B, V', log_probs.shape)
        k_per_beam = min(beam_width, V)
        candidates = []  # (new_score, tok_id, parent_idx)

        for b in range(B):
            nv = int(valid_counts[b].item())
            if nv == 0:
                continue  # prune this beam entirely

            vals, ids = torch.topk(log_probs[b], k=min(k_per_beam, nv))
            base = float(live_scores[b])
            for add_lp, tok_id in zip(vals.tolist(), ids.tolist()):
                # extra safety (should already hold by construction)
                if add_lp < log_prob_threshold:
                    continue
                candidates.append((base + add_lp, tok_id, b))
                # Optional debug prints:
                #print(math.exp(base + add_lp), tokenizer.decode(tok_id))

        # Global prune to top (beam_width *some factor*); we’ll keep exactly beam_width live beams
        candidates.sort(key=lambda x: x[0], reverse=True)
        #print('selecting next live beams')
        # Select next live beams and collect finished
        next_live_seqs, next_live_scores, next_parent_idx = [], [], []
        for score, tok_id, parent in candidates:
            tok = torch.tensor([[tok_id]], device=device, dtype=input_ids.dtype)
            new_seq = torch.cat([live_seqs[parent], tok], dim=-1)
            #print(math.exp(score), tokenizer.decode(new_seq[:, net_inputs['sequences'].shape[-1]:][0], skip_special_tokens=True, clean_up_tokenization_space=True))

            if ',' in tokenizer.decode(tok_id) or tok_id == eos_id: 
                #the last JSON element wont have a comma
                #The answer may be emitted with a comma, so include current new_seq
                #print('added to fin')

                #Get the generated text
                gen_ids = new_seq[:, base_len:]  # <-- only what the model just wrote for this field
                gen_text = tokenizer.decode(
                    gen_ids[0],
                    skip_special_tokens=True,                # keep punctuation as-is
                    clean_up_tokenization_space=True
                )

                finished.append((score, 
                                 {"sequences": new_seq, "pkv": None,
                                   "gen_ids": gen_ids, "gen_text": gen_text}))

            elif score >= log_prob_threshold:
                next_live_seqs.append(new_seq)
                next_live_scores.append(score)
                next_parent_idx.append(parent)

            if len(next_live_seqs) == beam_width:
                break

        if len(next_live_seqs) == 0:
            break  # nothing left to expand

        # Reorder the cache to the surviving parents (NO model._reorder_cache call)
        parent_idx_tensor = torch.tensor(next_parent_idx, device=device, dtype=torch.long)
        past = reorder_past(current_past, parent_idx_tensor, model=model)

        live_seqs = next_live_seqs
        live_scores = torch.tensor(next_live_scores, device=device)

    finished.sort(key=lambda x: x[0], reverse=True)
    return finished


def append_assistant_prompt(messages, assistant_prompt, tokenizer):
    """
    Encode `messages` using the chat template with an *open* assistant turn,
    then append the raw tokens of `assistant_prompt` (no extra special tokens).
    Returns a single LongTensor [1, T] you can pass to your decoder/beam code.
    """
    # 1) Prefix with assistant header (no EOT)
    base_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )  # -> Tensor [1, T_base]

    if not isinstance(base_ids, torch.Tensor):
        base_ids = torch.as_tensor(base_ids, dtype=torch.long)
        if base_ids.dim() == 1:
            base_ids = base_ids.unsqueeze(0)

    # 2) Raw content tokens for the assistant prompt (always list[int] -> tensor)
    prompt_ids_list = tokenizer.encode(assistant_prompt, add_special_tokens=False)
    if len(prompt_ids_list) == 0:
        #print('empty prompt')
        prompt_ids = base_ids.new_zeros((1, 0))  # empty prompt: 1x0 tensor
    else:
        prompt_ids = torch.tensor(prompt_ids_list, dtype=torch.long).unsqueeze(0)

    # 3) Concatenate
    input_ids = torch.cat([base_ids, prompt_ids], dim=1).to(device)

    return input_ids  # pass as net_inputs["sequences"]


def CeRTS_output_dist(messages, features, model, tokenizer, beam_width=5, max_steps=100):
    answer_distributions = [] # [(answer, score)]
    assistant_prompt = ''
    
    #fill answer_distributions for each feature
    for feature in features:

        if not answer_distributions:
            assistant_prompt += '{"' + feature + '": '
        else:
            prev_answer = answer_distributions[-1][0][0] #from the last answer, get the answer with the highest score
            assistant_prompt += prev_answer
            assistant_prompt += ',"' + feature + '": '

        #print(assistant_prompt)
        inputs = append_assistant_prompt(messages, assistant_prompt, tokenizer)
        #measures the output distribution and merges identical answers
        net_inputs = {'sequences': inputs}  # or include "pkv": precomputed_pkv if you have it
        results = multi_beam_search(model, tokenizer, net_inputs, beam_width=beam_width, max_steps=max_steps)
        d_val_probs = {} #Combines the probabilities of JSON's with the same value
        for log_prob, tokens in results:
            gen_text = tokens['gen_text'].strip() #only includes the text from the variable region

            #This allows the LLM to generate only a response for one feature, if it closes the JSON early there is no penalty.
            if ',' in gen_text:
                gen_text = gen_text.split(',')[0]
                json_str = '{"' + feature + '": ' + f'{str(gen_text)}' + '}'

            elif '}' in gen_text:
                #print('end bracket')
                gen_text = gen_text.split('}')[0]
                json_str = '{"' + feature + '": ' + f'{str(gen_text)}' + '}'

            data = extract_first_json(json_str)

            #Now the same numeric answers are combined before selecting what the top answer is.
            if data is None:
                #print('val is none')
                val = "missing"
            else:
                raw_val = data.get(feature)

                val = canonical_numeric_key(raw_val)
                if val is None:                    # not a clean number, keep as trimmed string
                    val = str(raw_val).strip()

                if val in d_val_probs:
                    d_val_probs[val] += math.exp(log_prob)
                else:
                    d_val_probs[val] = math.exp(log_prob)

        answer_distributions.append(sorted(d_val_probs.items(), 
                                           key=lambda item: item[1], reverse=True)) #[(answer, score)]
        
    print(answer_distributions)
        
    #return the top anser and the confidence for each feature
    for feature, answer_dist in zip(features, answer_distributions):
        #print(feature)
        #print(answer_dist)
        response = answer_dist[0][0]
        #print(answer_dist[0])
        confidence = top_2_delta([answer_score_tuple[1] for answer_score_tuple in answer_dist])
        print('Feature:', feature, 'Response:', response, 'Confidence:', confidence)
        

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

    temperature=1
    t1 = time.time()

    json_template = {"Name": "INSERT NAME", "Age": "INSERT AGE NUMBER"}
    features = json_template.keys()

    #text= 'The age of Rob is 7.'
    text= '2 years ago Rob turned 18. Today is his birthday and he is having fun.'

    # iterate to modify the assistant prompt
    messages = [
        {
            "role": "system",
            "content": (
                f"You are an NLP tool. Extract the following information from text: {','.join(features)}."
            )
        },
        {
            "role": "user",
            "content": (
                f"Extract the following information from the text: {','.join(features)}. "
                f"""If the information is not available, output "Not Available" for that field. """
                f"Provide the result **only** as a key-value pair with no extra text, using this JSON schema: {json.dumps(json_template)}. "
                f"Text: {text}"
            )
        }
    ]

    t1 = time.time()
    CeRTS_output_dist(messages, features, model, tokenizer, beam_width=5, max_steps=100)
    print(time.time() - t1, 'sec')


# Imports 
#### NONE REQUIRED YET

def tokenize_function(examples, tokenizer):
    # 'examples' is the dictionary passed by the 'map' function when batched=True
    # You must explicitly pass the list of texts from the 'advanced_text' column
    return tokenizer(examples['advanced_text'], truncation=True)
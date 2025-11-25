import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm  # Progress bar

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
# This section allows you to specify all "important things" in one place.
CONFIG = {
    "model_name": "emilyalsentzer/Bio_ClinicalBERT", 
    "max_length": 128,       # Max sequence length (shortened for speed in demo)
    "batch_size": 16,        # Reduce if you get OutOfMemory errors
    "epochs": 4,             # Number of training passes
    "learning_rate": 2e-5,   # Standard BERT learning rate (2e-5 to 5e-5)
    "freeze_bert": True,     # Set to True to train ONLY the heads first
    "seed": 42,              # Random seed for reproducibility
    
    # Define your classification tasks here
    "tasks": [
        {'name': 'diagnosis', 'num_labels': 5},    # e.g., 5 disease categories
        {'name': 'urgency', 'num_labels': 2},      # e.g., High vs Low
        {'name': 'department', 'num_labels': 10}   # e.g., Cardiology, Neurology, etc.
    ]
}

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. CUSTOM DATASET CLASS
# ==========================================
class MultiTaskDataset(Dataset):
    def __init__(self, texts, labels_dict, tokenizer, max_len):
        """
        Args:
            texts (list): List of document strings.
            labels_dict (dict): Dictionary where keys are task names and values are lists of labels.
                                Example: {'diagnosis': [0, 1, ...], 'urgency': [1, 0, ...]}
            tokenizer: Transformer tokenizer.
            max_len: Maximum sequence length.
        """
        self.texts = texts
        self.labels = labels_dict
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task_names = list(labels_dict.keys())

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # Gather labels for this specific sample across all tasks
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': {}
        }

        for task in self.task_names:
            # Convert label to tensor (LongTensor for classification)
            label_val = self.labels[task][idx]
            item['labels'][task] = torch.tensor(label_val, dtype=torch.long)

        return item

# ==========================================
# 3. MODEL DEFINITION
# ==========================================
class MultiHeadClinicalBERT(nn.Module):
    def __init__(self, model_path, task_configs, freeze_bert=False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        
        # Optional: Freeze BERT backbone to save memory/time or prevent overfitting
        if freeze_bert:
            print("--> Freezing ClinicalBERT backbone")
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.heads = nn.ModuleDict()
        hidden_size = self.bert.config.hidden_size
        
        # Create a separate classifier head for each task
        for task in task_configs:
            self.heads[task['name']] = nn.Linear(hidden_size, task['num_labels'])

    def forward(self, input_ids, attention_mask, labels=None):
        # 1. Shared Encoder Pass
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Shape: [batch_size, 768]
        
        results = {'logits': {}}
        total_loss = 0
        
        # 2. Multi-Head Pass
        for task_name, head in self.heads.items():
            logits = head(pooled_output)
            results['logits'][task_name] = logits
            
            # 3. Loss Calculation
            if labels and task_name in labels:
                loss_fct = nn.CrossEntropyLoss()
                task_loss = loss_fct(logits, labels[task_name])
                
                # Simple summation of losses. 
                # You could add weights here: total_loss += (task_loss * weight)
                total_loss += task_loss
        
        if labels:
            results['loss'] = total_loss
            
        return results

# ==========================================
# 4. TRAINING & EVALUATION FUNCTIONS
# ==========================================
def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Move dictionary of labels to device
        labels = {k: v.to(device) for k, v in batch['labels'].items()}

        # --- BACKPROPAGATION STEPS ---
        model.zero_grad()                       # 1. Clear previous gradients
        outputs = model(input_ids, attention_mask, labels=labels) # 2. Forward pass
        
        loss = outputs['loss']
        total_loss += loss.item()
        
        loss.backward()                         # 3. Calculate gradients (Backprop)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Optional: Clip gradients
        optimizer.step()                        # 4. Update weights
        scheduler.step()                        # 5. Update learning rate
        
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return total_loss / len(data_loader)

def eval_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    correct_preds = {task['name']: 0 for task in CONFIG['tasks']}
    total_samples = 0
    
    with torch.no_grad(): # Disable gradient calculation for validation
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = {k: v.to(device) for k, v in batch['labels'].items()}
            
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs['loss']
            total_loss += loss.item()
            
            # Calculate accuracy for each head
            for task_name, logits in outputs['logits'].items():
                preds = torch.argmax(logits, dim=1)
                correct_preds[task_name] += torch.sum(preds == labels[task_name]).item()
                
            total_samples += input_ids.size(0)
            
    avg_loss = total_loss / len(data_loader)
    accuracies = {k: v / total_samples for k, v in correct_preds.items()}
    
    return avg_loss, accuracies

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # --- A. Dummy Data Generation (Replace with your real data loading) ---
    print("Generating dummy data...")
    num_samples = 1000
    texts = [f"Medical note number {i}: Patient shows signs of severe symptoms." for i in range(num_samples)]
    
    # Random labels for 3 tasks
    labels_data = {
        'diagnosis': np.random.randint(0, 5, num_samples),   # 5 classes
        'urgency': np.random.randint(0, 2, num_samples),     # 2 classes
        'department': np.random.randint(0, 10, num_samples)  # 10 classes
    }
    
    # --- B. Prepare Data ---
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    
    # Split data (Indices)
    indices = np.arange(num_samples)
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=CONFIG['seed'])
    
    # Slice data based on split
    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    
    train_labels = {k: v[train_idx] for k, v in labels_data.items()}
    val_labels = {k: v[val_idx] for k, v in labels_data.items()}
    
    # Create Datasets
    train_dataset = MultiTaskDataset(train_texts, train_labels, tokenizer, CONFIG['max_length'])
    val_dataset = MultiTaskDataset(val_texts, val_labels, tokenizer, CONFIG['max_length'])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    
    # --- C. Initialize Model ---
    model = MultiHeadClinicalBERT(
        CONFIG['model_name'], 
        CONFIG['tasks'], 
        freeze_bert=CONFIG['freeze_bert']
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}\n")

    # --- D. Optimizer & Scheduler ---
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    total_steps = len(train_loader) * CONFIG['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=total_steps
    )
    
    # --- E. Training Loop ---
    print("Starting training...")
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        print("-" * 10)
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        val_loss, val_acc = eval_model(model, val_loader, device)
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val Accuracies: {val_acc}")
        
        # Save Checkpoint (Optional)
        # torch.save(model.state_dict(), f'multi_head_bert_epoch_{epoch}.bin')

    print("\nTraining Complete!"
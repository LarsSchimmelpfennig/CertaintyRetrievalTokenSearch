import torch 
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# class MultiHeadClinicalBERT(nn.Module):
#     def __init__(self, model_path, task_configs):
#         """
#         Args:
#             model_path (str): Path to your pre-trained ClinicalBERT.
#             task_configs (list of dict): Configuration for each task.
#                 Example:
#                 [
#                     {'name': 'asthma', 'num_labels': 2},
#                     {'name': 'smoking', 'num_labels': 2},
#                     {'name': 'pneu', 'num_labels': 2},
#                     {'name': 'common_cold', 'num_labels': 2},
#                     {'name': 'pain', 'num_labels': 2},
#                     {'name': 'fever', 'num_labels': 3},
#                     {'name': 'antibiotics', 'num_labels': 2}
#                 ]
#         """
#         super().__init__()
        
#         # Load pre-trained ClinicalBERT
#         self.bert = AutoModel.from_pretrained(model_path)
        
#         # Create classification heads for each feature
#         self.heads = nn.ModuleDict()
#         hidden_size = self.bert.config.hidden_size
        
#         for task in task_configs:
#             task_name = task['name']
#             num_labels = task['num_labels']
            
#             # Simple linear head: 768 -> num_labels
#             # You can add Dropout or extra dense layers here if needed
#             self.heads[task_name] = nn.Linear(hidden_size, num_labels)

#     def forward(self, input_ids, attention_mask, labels=None):
#         """
#         Args:
#             input_ids, attention_mask: Standard BERT inputs
#             labels (dict): Dictionary of labels for each task 
#                            e.g., {'department': tensor([1, 0...]), 'urgency': ...}
#         """
#         # Pass input through the shared ClinicalBERT
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
#         # 'pooler_output' is the embedding of the [CLS] token
#         # This represents the entire document
#         pooled_output = outputs.pooler_output
        
#         results = {}
#         total_loss = 0
        
#         # Pass the pooled output through each specific head
#         for task_name, head_layer in self.heads.items():
#             logits = head_layer(pooled_output)
#             results[task_name] = logits
            
#             # Our Loss is going to be total loss accross all features concurrently
#             if labels and task_name in labels:
#                 loss_fct = nn.CrossEntropyLoss()
#                 # Calculate loss for this specific task
#                 task_loss = loss_fct(logits, labels[task_name])
                
#                 # Add to total loss (you can weight these if one task is harder)
#                 total_loss += task_loss
        
#         # Return a dictionary containing loss and logits for all heads
#         return {
#             "loss": total_loss if labels else None,
#             "logits": results
#         }

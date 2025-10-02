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
import sys
import numpy as np
from typing import Optional, List
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig
import json, csv
from sklearn.model_selection import train_test_split

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

def gen_prompt_no_shot(text, features, json_template):

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
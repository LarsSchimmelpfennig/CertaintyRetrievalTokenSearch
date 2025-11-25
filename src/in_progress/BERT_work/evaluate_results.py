# Base imports
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import sys, os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# Add some evaluation specific functions
from python_functions.evaluation_functions import *

############ Initialization 
features = ['asthma', 'smoking', 'pneu', 'common_cold', 'pain', 'fever', 'antibiotics']
ordered_cols = ['compact_note','asthma','smoking','pneu','common_cold','pain','fever','antibiotics','LLM_asthma','asthma_conf','LLM_smoking','smoking_conf','LLM_pneu','pneu_conf','LLM_common_cold','common_cold_conf','LLM_pain','pain_conf','LLM_fever','fever_conf','LLM_antibiotics','antibiotics_conf']

############ PERFORM EVALUATION
#LOAD_LOC = f'1k_ClinBERT_SimSum_Predictions.csv'
LOAD_LOC = f'1k_BERT_SimSum_Predictions.csv'
df = pd.read_csv(LOAD_LOC)
df['compact_note'] = np.nan
df = df[ordered_cols]
# model_name = 'ClinBERT'
model_name = 'BERT'

# Get Numeric Metrics
metrics_df, overall_weighted_f1 = compute_metrics_strict_macro(df)
# metrics_df.to_csv(f'results_data/SimSUM_classification_performance_ClinBERT.csv')
metrics_df.to_csv(f'results_data/SimSUM_classification_performance_BERT.csv')
print("\nOverall weighted F1 (across features):", round(overall_weighted_f1, 4))

# Perform Calibration
# calib_df = compute_calibration_and_brier(df, model_name=model_name, outdir="results_data/SimSUM_ClinBERT_calibration_plots", num_bins=10)
# calib_df.to_csv(f'results_data/SimSUM_calibration_performance_ClinBERT.csv')
calib_df = compute_calibration_and_brier(df, model_name=model_name, outdir="results_data/SimSUM_BERT_calibration_plots", num_bins=10)
calib_df.to_csv(f'results_data/SimSUM_calibration_performance_BERT.csv')
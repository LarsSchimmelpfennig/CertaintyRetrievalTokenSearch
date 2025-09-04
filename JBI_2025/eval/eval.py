import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from scipy.stats import entropy
import sys
import math
from datetime import datetime
import csv
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score

def brier_score(correct_array, confidence_array):
    total = 0
    for correct, confidence in zip(correct_array, confidence_array):
        total += (confidence - correct) ** 2
    return total / len(correct_array)

def calibration_plot(correct_array, confidence_array, label, file_name, model_id, num_bins=10):
    sorted_indices = np.argsort(confidence_array)
    sorted_conf = confidence_array[sorted_indices]
    
    sorted_corr = correct_array[sorted_indices]

    # 2. Determine how many samples per bin
    n = len(sorted_conf)
    bin_size = n // num_bins
    remainder = n % num_bins

    avg_confidences = []
    avg_accuracies = []

    start_idx = 0
    for i in range(num_bins):
        # Bin size might differ by 1 for some bins if not perfectly divisible
        current_bin_size = bin_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_bin_size

        bin_conf = sorted_conf[start_idx:end_idx]
        bin_corr = sorted_corr[start_idx:end_idx]

        # Compute average confidence (X) and average correctness (Y) in this bin
        if len(bin_conf) > 0:
            avg_confidences.append(np.mean(bin_conf))
            avg_accuracies.append(np.mean(bin_corr))
        else:
            # If bin is empty (possible edge case), skip
            pass

        start_idx = end_idx

    # 3. Plot
    #fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot the perfect calibration line (y = x)

    # Plot the average confidence vs average accuracy
    plt.clf()
    plt.plot(avg_confidences, avg_accuracies, label=label, marker='o', linewidth=2)

    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), '--', color='black', label='Perfect Calibration')

    # Formatting
    plt.xlabel("Predicted Confidence", fontsize=14)
    plt.ylabel("Observed Accuracy", fontsize=14)
    plt.title(model_id, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc="upper left", fontsize=12)
    plt.savefig(f'output_files/CeRTS/calibration_plots/{file_name}', dpi=300)
    #return fig, ax


def are_dates_equal(date_str1, date_str2):
    """
    Compares two date strings and determines if they represent the same date.
    
    Assumptions:
    - Dates are in MM/DD/YY or MM/DD/YYYY format.
    - The last two digits of the year belong to the 2000s if only two digits are provided.
    - Leading zeros do not affect the comparison.
    - If either date string is invalid, return False.
    
    Returns:
    - True if both dates are equivalent, False otherwise.
    """
    def parse_date(date_str):
        # Handle cases where the year is provided as YY and assume it is in the 2000s
        for fmt in ("%m/%d/%Y", "%m/%d/%y"):
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
        return None  # Return None if parsing fails

    date1 = parse_date(date_str1)
    date2 = parse_date(date_str2)

    if date1 is None or date2 is None:
        return False

    return date1 == date2

annotated_df = pd.read_csv('annotations/per_patient_gold.csv')

#all of these are 'Not Available', All 'Not available' Should be 0
annotated_df["Number of discharge medications (Gold Standard)"] = annotated_df["Number of discharge medications (Gold Standard)"].fillna(0).replace("Not Available", 0)

annotated_df['Prescribed Medications(Y/N) (Gold Standard)'] = annotated_df['Prescribed Medications(Y/N) (Gold Standard)'].fillna(
    'NO'
)

#reformat dates
annotated_df['Treatment date (Gold Standard)'] = pd.to_datetime(
    annotated_df['Treatment date (Gold Standard)'], errors='coerce'
).dt.strftime('%-m/%-d/%Y')

for feature in ["Age (Gold Standard)", "Treatment date (Gold Standard)", 'Blood Pressure Value (Gold Standard)']:
    annotated_df[feature] = annotated_df[feature].fillna('Not Available')

for feature in ["Age (Gold Standard)", "Number of discharge medications (Gold Standard)"]:
    mask = annotated_df[feature] != "Not Available"
    print(sum(mask))
    annotated_df.loc[mask, feature] = annotated_df.loc[mask, feature].apply(
        lambda x: str(x).split('.')[0] if '.' in str(x) else str(x)
    )

annotated_df['Treated with Immunotherapy'] = annotated_df['Treatment (Gold Standard)'].apply(
    lambda x: 'YES' if 'Immunotherapy' in str(x).split(', ') else 'NO'
)

#convert Yes to YES and NO
for feature in ['Mention of lung cancer/carcinoma (Gold Standard)', 'Prescribed Medications(Y/N) (Gold Standard)']:
    annotated_df[feature] = (
        annotated_df[feature]
        .replace({'Yes': 'YES', 'No': 'NO'})
    )

d_mrn_labels = []

for idx, feature in enumerate(["Number of discharge medications (Gold Standard)",
    'Mention of lung cancer/carcinoma (Gold Standard)', 'Age (Gold Standard)',
    'Treatment date (Gold Standard)', 'Blood Pressure Value (Gold Standard)',
    'Treated with Immunotherapy']):
    #for l in d_mrn_labels:
    d_mrn_labels.append({})
    for mrn, feature_val in annotated_df[['EPIC_MRN', feature]].values:
        if int(mrn) not in d_mrn_labels[idx]:
            d_mrn_labels[idx][int(mrn)] = str(feature_val)

#If a model does not produce a response then it is wrong

def safe_str(val):
    # If the value is numeric (like 0.0), convert to int first to avoid '0.0'
    if isinstance(val, (int, float)):
        val = str(int(val))
    return str(val)

def safe_str_float(val):
    # If the value is numeric (like 0.0), convert to int first to avoid '0.0'
    if isinstance(val, (float)):
        val = str(val)
    return str(val)

#What is the aggregate AUROC across all features. If a feature is easy then the auroc may be low since nearly all points are correct

top_p=1
thresh = 0.05
with open(f'output_files\CeRTS\CeRTS_performance.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['model', 'threshold', 'feature', 'accuracy', 'avg_top_2_delta', 'sem_top_2_delta', 'avg_prop_missing', 'confidence_acc_diff', 'avg_num_equiv_runs', 'brier_score', 'AUROC'])
    #'gemma-3-27b-it',
    for model_id in ['Qwen2.5-7B-Instruct-1M', 'gemma-2-9b-it', 'Phi-4-mini-instruct', 'Llama-3.1-8B-Instruct', 'Mistral-Small-24B-Instruct-2501', 'DeepSeek-R1-Distill-Llama-8B', 'Mixtral-8x7B-Instruct-v0.1', 'Hermes-3-Llama-3.1-8B']:
        
        aggregate_conf = []
        aggregate_correct = []
        
        for idx, feature in enumerate(["Number of discharge medications", 'Mention of lung cancer_carcinoma', 'Age', 'First treatment date', 'Blood pressure value at discharge', 'Treated with immunotherapy']):
            #print(model_id, feature)
            token_df = pd.read_csv(f'data/token_distributions/extract_{feature}_thresh_{thresh}_top_p_{top_p}_{model_id}_auto_special_tokens.csv')
            token_df['top_outputs'] = token_df['top_outputs'].fillna('missing')
            token_df['top_probs'] = token_df['top_probs'].fillna('0.0')

            if feature in ["Number of discharge medications", 'Age']:
                #convert any individual float answers to str ints
                token_df['top_outputs'] = token_df['top_outputs'].apply(safe_str)
                token_df['top_probs'] = token_df['top_probs'].apply(safe_str_float)

            top_2_delta = []
            top_outputs = []
            correct_labels = [] #is it correct
            answers = []
            gold_labels = []
            props_missing = []

            for patient_id, tokens, str_probs in token_df[['patient_id', 'top_outputs', 'top_probs']].values:
                #print(tokens, str_probs)
                outputs = tokens.split(',')
                probs = [float(str_prob) for str_prob in str_probs.split(',')]

                missing_prob = 1 - np.sum(probs)
                props_missing.append(missing_prob)
                #probs.append(missing_prob)
                #outputs.append('missing')

                sorted_indices = np.argsort(probs)[::-1]
                probs = np.array(probs)[sorted_indices]
                outputs = np.array(outputs)[sorted_indices]
            
                if int(patient_id) in d_mrn_labels[idx]:

                    if len(probs) > 1:
                        top_2_delta.append(probs[0] - probs[1])
                    else:
                        top_2_delta.append(probs[0])

                    answer = outputs[0]
                    gold_label = d_mrn_labels[idx][int(patient_id)]

                    if feature == 'First treatment date':
                        #print(d_mrn_labels[idx][int(patient_id)][0], answer)
                        if gold_label == 'Not Available' and answer == 'Not Available':
                            correct_labels.append(1)
                        elif gold_label == 'Not Available' or answer == 'Not Available':
                            correct_labels.append(0)

                        elif are_dates_equal(gold_label, answer):
                            #print('matched by function')
                            correct_labels.append(1)
                        else:
                            #print('no match')
                            correct_labels.append(0)
                        #print(correct_labels[-1])

                    else:
                        if gold_label == answer:
                            correct_labels.append(1)
                        else:
                            correct_labels.append(0)

                    aggregate_conf.append(top_2_delta[-1])
                    aggregate_correct.append(correct_labels[-1])
                    #if feature == 'Mention of lung cancer_carcinoma':
                    #    print(answer, gold_label, correct_labels[-1], top_2_delta[-1])

            #print(feature, len(correct_labels))
            
            #Average top_2_delta and accuracy.
            correct_labels = np.array(correct_labels)
            accuracy = sum(correct_labels) / len(correct_labels)
            top_2_delta_avg = np.mean(top_2_delta)
            top_2_delta_std = np.std(top_2_delta)
            top_2_delta_std_standard_error = stats.sem(top_2_delta)

            auroc = roc_auc_score(correct_labels, top_2_delta)
            brier = brier_score(np.array(correct_labels), np.array(top_2_delta))

            if math.isnan(auroc):
                auroc=1 

            csvwriter.writerow([model_id, thresh, feature, accuracy, top_2_delta_avg, top_2_delta_std_standard_error, np.mean(props_missing), top_2_delta_avg-accuracy, token_df['num_equivalent_runs'].mean(), brier, auroc])

            calibration_plot(np.array(correct_labels), np.array(top_2_delta), 'CeRTS', f'{feature}_top2_delta_calibration_10_bins_{model_id}_thresh_{thresh}_top_p_{top_p}_gold.png', model_id, num_bins=10)
            


        #AGGREGATE STATS TO PRINT
        
        auroc = roc_auc_score(aggregate_correct, aggregate_conf)
        brier = brier_score(np.array(aggregate_correct), np.array(aggregate_conf))

        print(model_id, 'brier',brier,'auroc',auroc)


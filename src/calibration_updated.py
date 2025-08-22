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
    plt.savefig(f'data/final_CeRTS_calibration_plots/{file_name}', dpi=300)
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


def normalized_entropy(probs, N=15):
    if len(probs) > N:
        raise ValueError("Input list length cannot exceed N.")
    if any(p < 0 for p in probs):
        raise ValueError("Probabilities must be non-negative.")
    total = sum(probs)
    if total > 1 + 1e-12:                 # small tolerance for float error
        raise ValueError("Probabilities sum to more than 1.")
    
    # Fill missing slots equally with the leftover probability mass
    missing = N - len(probs)
    if missing > 0:
        remainder = 1.0 - total
        fill_val = remainder / missing
        full_dist = probs + [fill_val] * missing
    else:
        # Already length N.  Re-normalise in case total<1 due to round-off.
        full_dist = probs + []
        if abs(total - 1.0) > 1e-12:
            full_dist = [p / total for p in full_dist]

    # Shannon entropy (natural log)
    entropy = -sum(p * math.log(p) for p in full_dist if p > 0.0)
    max_entropy = math.log(N)            # entropy of uniform dist of length N
    return 1 - (entropy / max_entropy)

#model_id = 'Llama-3.1-8B-Instruct'
#model_id = 'Qwen2.5-7B-Instruct-1M'

#data/token_distributions/extract_First treatment date_temp_1_thresh_0.05_top_p_0.9_Llama-3.1-8B-Instruct_log_prob.csv

#token_df = pd.read_csv('data/token_distributions/extract_Mention of lung cancer_carcinoma_temp_1_thresh_0.05_top_p_0.9_Llama-3.1-8B-Instruct_log_prob.csv')
#token_df = pd.read_csv('data/token_distributions/extract_Mention of lung cancer_carcinoma_temp_1_thresh_0.05_top_p_0.9_Qwen2.5-7B-Instruct-1M_log_prob.csv')

# annotated_df = pd.read_csv('annotations/LungCancerAnnotation_DATA_2024-07-24_1333.csv')

# annotated_df = pd.read_csv('annotations/all_data.csv')
# annotated_df = annotated_df.drop_duplicates(subset='EPIC_MRN', keep='first')
# annotated_df.to_csv('annotations/per_patient_gold.csv')

annotated_df = pd.read_csv('annotations/per_patient_gold.csv')

#replace invalid date with Not Available.
# for column in ["Number of discharge medications (Gold Standard)", 
#                'Mention of lung cancer/carcinoma (Gold Standard)', 
#                'Age (Gold Standard)', 
#                'Treatment date (Gold Standard)']:
    
#     num_missing = annotated_df[column].isna().sum()
#     print(f"Column '{column}' has {num_missing} missing values.")

# # Check values in 'Number of discharge medications' when the Gold Standard version is missing
# missing_mask = annotated_df["Number of discharge medications (Gold Standard)"].isna()
# missing_values = annotated_df.loc[missing_mask, ["EPIC_MRN","Number of discharge medications"]]

# print("\nValues of 'Number of discharge medications' where the Gold Standard is NaN:")
# print(missing_values)

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

#print(annotated_df['Number of discharge medications (Gold Standard)'].value_counts())
#print(annotated_df['Number of discharge medications (Gold Standard)'][0], type(annotated_df['Number of discharge medications (Gold Standard)'][0]))
#remove .0 from age and num medications
for feature in ["Age (Gold Standard)", "Number of discharge medications (Gold Standard)"]:
    mask = annotated_df[feature] != "Not Available"
    print(sum(mask))
    annotated_df.loc[mask, feature] = annotated_df.loc[mask, feature].apply(
        lambda x: str(x).split('.')[0] if '.' in str(x) else str(x)
    )
#print(annotated_df['Number of discharge medications (Gold Standard)'].value_counts())
#print(annotated_df['Number of discharge medications (Gold Standard)'][0], type(annotated_df['Number of discharge medications (Gold Standard)'][0]))
#sys.exit()
#Extract if immunology mentioned in treatment, convert No Information to Not Available
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


def fpr_fnr_plot(correct_labels, top_2_delta, feature, model):
    y_true = np.asarray(correct_labels).astype(int)
    confid = np.asarray(top_2_delta).astype(float)

    if y_true.shape != confid.shape:
        raise ValueError("correct_labels and top_2_delta must have the same shape")

    # Thresholds in descending order so the x-axis runs high→low confidence
    #thresholds = np.r_[np.unique(confid)[::-1], 0.0]
    thresholds = np.linspace(0, 1, 1000)

    fpr_vals = []
    fnr_vals = []

    # Positive = "predict correct" when confidence ≥ threshold
    for thr in thresholds:
        preds = (confid >= thr).astype(int)

        tp = ((preds == 1) & (y_true == 1)).sum()
        tn = ((preds == 0) & (y_true == 0)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()

        fpr_vals.append(fp / (fp + tn + 1e-9))   # add epsilon to avoid 0/0
        fnr_vals.append(fn / (fn + tp + 1e-9))

    idx = np.where(np.array(fpr_vals) <= 0.1)[0]
    #print(idx)

    if idx.size != 0:
        selected_threshold = thresholds[idx[0]]
        #print(model, feature, selected_threshold, 'FNR', fnr_vals[idx[0]])

        fig, ax = plt.subplots(figsize=(7, 5))

        plt.title(feature)
        ax.plot(thresholds, fpr_vals, label="FPR", color='#2ca02c')
        ax.plot(thresholds, fnr_vals, label="FNR", color='#ff7f0e')
        #ax.axhline(0.05, color='black', linestyle='--', linewidth=1, label='FPR=0.05')
        ax.axvline(selected_threshold, color='black', linestyle='--', linewidth=1, label='FPR<=0.1')
        ax.set_xlabel("CeRTS Confidence")
        ax.set_ylabel("")            # No y-axis label per request
        ax.set_xlim(thresholds.min(), thresholds.max())
        ax.set_ylim(0, 1)
        ax.legend(frameon=False)

        fig.savefig(f'data/fpr_fnr_plors/{feature}_{model}.png', dpi=300, bbox_inches="tight")
    else:
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.savefig(f'data/fpr_fnr_plors/{feature}_{model}.png', dpi=300, bbox_inches="tight")

    #return fig, ax

#What is the aggregate AUROC across all features. If a feature is easy then the auroc may be low since nearly all points are correct

d_incorrect_date_count_mrn = {}
d_incorrect_immunotherapy = {}

top_p=1
thresh = 0.05
with open(f'data/final_CeRTS_calibration_plots/top_2_delta_accuracy_alignment_thresh_{thresh}_top_p_{top_p}_gold.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['model', 'threshold', 'feature', 'accuracy', 'avg_top_2_delta', 'sem_top_2_delta', 'avg_prop_missing', 'confidence_acc_diff', 'avg_num_equiv_runs', 'brier_score', 'AUROC'])
    #'gemma-3-27b-it',


    aggregate_top_2_delta = []
    aggregate_top_1 = []
    aggregate_min_token = []
    aggregate_norm_entropy = []
    aggregate_correct = []


    for model_id in ['Qwen2.5-7B-Instruct-1M', 'gemma-2-9b-it', 'Phi-4-mini-instruct', 'Llama-3.1-8B-Instruct', 'Mistral-Small-24B-Instruct-2501', 'DeepSeek-R1-Distill-Llama-8B', 'Hermes-3-Llama-3.1-8B', 'Mixtral-8x7B-Instruct-v0.1']:


        for idx, feature in enumerate(["Number of discharge medications", 'Mention of lung cancer_carcinoma', 'Age', 'First treatment date', 'Blood pressure value at discharge', 'Treated with immunotherapy']):
            #print(model_id, feature)

            token_df = pd.read_csv(f"data/min_token_dist/{feature.replace('/', '_')}_{model_id}.csv")
            #token_df = token_df.dropna(subset=['top_outputs', 'top_probs'])
            #fill top_outputs NA with wrong, and top_probs with 0.
            token_df['top_outputs'] = token_df['top_outputs'].fillna('missing')
            token_df['top_probs'] = token_df['top_probs'].fillna('0.0')

            if feature in ["Number of discharge medications", 'Age']:
                #convert any individual float answers to str ints
                token_df['top_outputs'] = token_df['top_outputs'].apply(safe_str)
                token_df['top_probs'] = token_df['top_probs'].apply(safe_str_float)

            top_2_delta = []
            top_1 = []
            norm_entropy = []
            min_token_probs = []

            top_outputs = []
            correct_labels = [] #is it correct
            answers = []
            gold_labels = []
            props_missing = []

            for patient_id, tokens, str_probs, min_token_prob in token_df[['patient_id', 'top_outputs', 'top_probs', 'answer_min_token_prob']].values:
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

                    top_1.append(probs[0])
                    min_token_probs.append(float(min_token_prob))

                    norm_entropy.append(normalized_entropy(list(probs)))

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
                        
                        if correct_labels[-1] == 0:
                            
                            if patient_id not in d_incorrect_date_count_mrn: #add the overconfidence
                                d_incorrect_date_count_mrn[patient_id] = top_2_delta[-1]
                                
                            else:
                                d_incorrect_date_count_mrn[patient_id] += top_2_delta[-1]

                    else:
                        if gold_label == answer:
                            correct_labels.append(1)
                        else:
                            correct_labels.append(0)
                            
                        if feature == 'Treated with immunotherapy' and gold_label != answer:
                            
                            if patient_id not in d_incorrect_immunotherapy:
                                d_incorrect_immunotherapy[patient_id] = top_2_delta[-1]
                            else:
                                d_incorrect_immunotherapy[patient_id] += top_2_delta[-1]
                            
                            
                    aggregate_top_2_delta.append(top_2_delta[-1])
                    aggregate_top_1.append(top_1[-1])
                    aggregate_min_token.append(min_token_probs[-1])
                    aggregate_norm_entropy.append(norm_entropy[-1])
                    aggregate_correct.append(correct_labels[-1])

                    #print(probs)
                    #print('top 1', top_1[-1])
                    #print('min', min_token_probs[-1])
                    #print('norm entropy', norm_entropy[-1])
                    #print('correct', correct_labels[-1])


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

            fpr_fnr_plot(correct_labels, top_2_delta, feature, model_id)



            csvwriter.writerow([model_id, thresh, feature, accuracy, top_2_delta_avg, top_2_delta_std_standard_error, np.mean(props_missing), top_2_delta_avg-accuracy, token_df['num_equivalent_runs'].mean(), brier, auroc])
        #calibration_plot(np.array(correct_labels), np.array(top_2_delta), 'CeRTS', f'{feature}_{model_id}.png', model_id, num_bins=10)
        
        
        
sorted_items = sorted(d_incorrect_immunotherapy.items(), key=lambda item: item[1], reverse=True)
# Get the key(s) from the first item(s) in the sorted list
print(sorted_items[:5])


# highest_key_value = 
# highest_key = highest_key_value[0]
# print(highest_key)     
        
        
    #AGGREGATE STATS TO PRINT
# for name, conf_array in [('top_2_delta', aggregate_top_2_delta), ('norm_entropy', aggregate_norm_entropy),
#                             ('top_1', aggregate_top_1), ('min_token_prob', aggregate_min_token)]:

#     auroc = roc_auc_score(aggregate_correct, conf_array)
#     brier = brier_score(np.array(aggregate_correct), np.array(conf_array))

#     if math.isnan(auroc):
#         auroc=1 

#     print('final aggregate', name, 'brier',brier,'auroc',auroc)
#     print()
            

                #print(top_p, model_id, feature, 'brier score', brier_score(np.array(correct_labels), np.array(top_2_delta)))
                #print(top_p, model_id, feature, 'auroc', auroc)

                #fpr, tpr, thresholds = roc_curve(correct_labels, top_2_delta)
                

                # plt.clf()

                # # Plot the ROC curve
                # plt.figure(figsize=(8, 6))
                # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auroc:.2f})')
                # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
                # plt.xlim([0.0, 1.0])
                # plt.ylim([0.0, 1.05])
                # plt.xlabel('False Positive Rate')
                # plt.ylabel('True Positive Rate')
                # plt.title('Receiver Operating Characteristic (ROC)')
                # plt.legend(loc='lower right')
                # plt.savefig(f'data/final_calibration_plots/{feature}_{model_id}_thresh_{thresh}_top_p_{top_p}_gold_ROC.png', dpi=300)


    #can also apply null model for probabilities
    #How well do the different metrics relate to accuracy.



#calibration_plot(np.array(correct_labels), np.array(entropy_vals), '1 - Entropy', 'lung_cancer_mention_entropy_calibration_10_bins.png', num_bins=10)
#calibration_plot(np.array(correct_labels), np.array(top_2_delta), 'Top 2 Tokens Delta', f'lung_cancer_mention_top2_delta_calibration_10_bins_{model_id}.png', num_bins=10)

#Can make a null model for this probably
#print('brier score entropy', brier_score(np.array(correct_labels), np.array(entropy_vals)))
#print('brier score top_2_delta', brier_score(np.array(correct_labels), np.array(top_2_delta)))


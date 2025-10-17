import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import sys, os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

### UNIVERSAL VARIABLES 
BIN_FEATURES = ["asthma", "smoking", "pneu", "common_cold", "pain", "antibiotics"]
BIN_ALLOWED  = {"yes", "no"}
MC_FEATURES  = {"fever": ["high", "low", "no"]}
INVALID = "__invalid__"

FEATURES = [
    ("asthma", "LLM_asthma", "asthma_conf"),
    ("smoking", "LLM_smoking", "smoking_conf"),
    ("pneu", "LLM_pneu", "pneu_conf"),
    ("common_cold", "LLM_common_cold", "common_cold_conf"),
    ("pain", "LLM_pain", "pain_conf"),
    ("fever", "LLM_fever", "fever_conf"),                 # multiclass is fine: correctness is exact-match
    ("antibiotics", "LLM_antibiotics", "antibiotics_conf"),
]

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
    plt.savefig(f'{file_name}', dpi=300)
    
    
def _strict(series: pd.Series, allowed: set) -> pd.Series:
    s = series.astype(str)
    return s.where(s.isin(allowed), other=INVALID)

def compute_metrics_strict_macro(df: pd.DataFrame):
    rows = []

    # --- Binary features: macro over yes & no (invalid counted as errors) ---
    for feat in BIN_FEATURES:
        y_true = _strict(df[feat], BIN_ALLOWED)
        y_pred = _strict(df[f"LLM_{feat}"], BIN_ALLOWED)

        rep = classification_report(
            y_true, y_pred,
            labels=["yes", "no", INVALID],  # include INVALID to penalize
            output_dict=True, zero_division=0
        )

        # macro over just the valid classes
        f1_macro = (rep["yes"]["f1-score"] + rep["no"]["f1-score"]) / 2.0
        prec_macro = (rep["yes"]["precision"] + rep["no"]["precision"]) / 2.0
        rec_macro  = (rep["yes"]["recall"]    + rep["no"]["recall"])    / 2.0

        support_valid = int(rep["yes"]["support"] + rep["no"]["support"])
        invalid_preds = int((y_pred == INVALID).sum())
        acc = rep.get("accuracy", 0.0)

        rows.append({
            "feature": feat,
            "type": "binary (macro over yes|no)",
            "precision": round(prec_macro, 4),
            "recall": round(rec_macro, 4),
            "f1": round(f1_macro, 4),
            "accuracy": round(acc, 4),
            "support": support_valid,                 # only valid GT rows
            "invalid_llm_preds": invalid_preds,
        })

    # --- Multiclass (fever): macro over valid classes ---
    for feat, classes in MC_FEATURES.items():
        allowed = set(classes)
        y_true = _strict(df[feat], allowed)
        y_pred = _strict(df[f"LLM_{feat}"], allowed)

        rep = classification_report(
            y_true, y_pred,
            labels=classes + [INVALID],
            output_dict=True, zero_division=0
        )

        prec_macro = np.mean([rep[c]["precision"] for c in classes])
        rec_macro  = np.mean([rep[c]["recall"]    for c in classes])
        f1_macro   = np.mean([rep[c]["f1-score"]  for c in classes])
        support_valid = int(sum(rep[c]["support"] for c in classes))
        invalid_preds = int((y_pred == INVALID).sum())
        acc = rep.get("accuracy", 0.0)

        rows.append({
            "feature": feat,
            "type": f"multiclass macro ({'|'.join(classes)})",
            "precision": round(float(prec_macro), 4),
            "recall": round(float(rec_macro), 4),
            "f1": round(float(f1_macro), 4),
            "accuracy": round(acc, 4),
            "support": support_valid,
            "invalid_llm_preds": invalid_preds,
        })

    metrics_df = pd.DataFrame(rows).set_index("feature").sort_index()

    # Overall weighted F1 across features (weights = per-feature support)
    weights = metrics_df["support"].to_numpy()
    f1s     = metrics_df["f1"].to_numpy()
    overall_weighted_f1 = float(np.average(f1s, weights=weights)) if weights.sum() > 0 else 0.0

    return metrics_df, overall_weighted_f1

def compute_calibration_and_brier(
    df: pd.DataFrame,
    model_name: str,
    outdir: str = "data/calibration_plots",
    num_bins: int = 10,
) -> pd.DataFrame:
    rows = []
    all_correct, all_conf = [], []

    for feat, pred_col, conf_col in FEATURES:
        # Strict correctness: exact string match (no normalization)
        y_true = df[feat].astype(str)
        y_pred = df[pred_col].astype(str)
        conf   = pd.to_numeric(df[conf_col], errors="coerce")

        # Keep rows with non-null confidence and non-null labels/preds
        mask = (~y_true.isna()) & (~y_pred.isna()) & (~conf.isna())
        if mask.sum() == 0:
            rows.append({
                "feature": feat, "n": 0,
                "brier_score": np.nan, "roc_auc": np.nan,
                "mean_conf": np.nan, "accuracy": np.nan,
                "calibration_plot": None
            })
            continue

        correct = (y_true[mask].values == y_pred[mask].values).astype(int)
        scores  = np.clip(conf[mask].values.astype(float), 0.0, 1.0)

        # Metrics
        brier = float(brier_score(correct, scores))
        try:
            rocauc = float(roc_auc_score(correct, scores)) if len(np.unique(correct)) > 1 else np.nan
        except Exception:
            rocauc = np.nan
        acc = float(np.mean(correct))
        mean_conf = float(np.mean(scores))

        # Save calibration plot
        safe_feat = feat.replace("/", "_")
        safe_model = model_name.replace("/", "_").replace(" ", "_")
        fname = os.path.join(outdir, f"{safe_feat}__{safe_model}.png")
        calibration_plot(
            correct_array=correct,
            confidence_array=scores,
            label=f"{feat} ({len(correct)} samples)",
            file_name=fname,
            model_id=f"{model_name} · {feat}",
            num_bins=num_bins,
        )

        # Record
        rows.append({
            "feature": feat,
            "n": int(len(correct)),
            "brier_score": round(brier, 6),
            "roc_auc": round(rocauc, 6) if rocauc == rocauc else np.nan,  # keep NaN if undefined
            "mean_conf": round(mean_conf, 6),
            "accuracy": round(acc, 6),
            "calibration_plot": fname,
        })

        # aggregate for "all-features" plot
        all_correct.append(correct)
        all_conf.append(scores)

    # Combined across all features
    if len(all_correct) and sum(len(c) for c in all_correct) > 0:
        all_correct = np.concatenate(all_correct)
        all_conf = np.concatenate(all_conf)
        brier_all = float(brier_score(all_correct, all_conf))
        try:
            rocauc_all = float(roc_auc_score(all_correct, all_conf)) if len(np.unique(all_correct)) > 1 else np.nan
        except Exception:
            rocauc_all = np.nan
        acc_all = float(np.mean(all_correct))
        mean_conf_all = float(np.mean(all_conf))

        fname_all = os.path.join(outdir, f"ALL_FEATURES__{model_name.replace('/', '_').replace(' ', '_')}.png")
        calibration_plot(
            correct_array=all_correct,
            confidence_array=all_conf,
            label=f"all ({len(all_correct)} samples)",
            file_name=fname_all,
            model_id=f"{model_name} · ALL",
            num_bins=num_bins,
        )

        rows.append({
            "feature": "ALL_FEATURES",
            "n": int(len(all_correct)),
            "brier_score": round(brier_all, 6),
            "roc_auc": round(rocauc_all, 6) if rocauc_all == rocauc_all else np.nan,
            "mean_conf": round(mean_conf_all, 6),
            "accuracy": round(acc_all, 6),
            "calibration_plot": fname_all,
        })

    out = pd.DataFrame(rows).set_index("feature")
    return out
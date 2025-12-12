# cal_plot_all_features_probe.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Inputs (normalized slashes) ----------
paths_raw = {
    # Probe models
    "Qwen Probe":  r"data\hidden_layer_probes\test\Qwen2.5-1.5B-Instruct_probes_xgb_no_CE\test_with_probe_confs_canonical_last_L25.csv",
    "Llama Probe": r"data\hidden_layer_probes\test\Llama-3.2-3B-Instruct_probes_xgb_no_CE\test_with_probe_confs_canonical_last_L24.csv",

    # SFT-CeRTS fine-tuned models
    "Llama SFT-CeRTS": r"fine_tuning_work\results_data\CeRTS_SFT_llama-3.2-3b_SimSum.csv",
    "Qwen SFT-CeRTS":  r"fine_tuning_work\results_data\CeRTS_SFT_Qwen2.5-1.5B_SimSum.csv",

    # BERT models
    "ClinBERT": r"BERT_work\1k_BERT_SimSum_Predictions.csv",
    "BERT":  r"BERT_work\1k_ClinBERT_SimSum_Predictions.csv",
}
paths = {k: v.replace("\\", "/") for k, v in paths_raw.items()}

# ---------- Config ----------
FEATURES = ["asthma","smoking","pneu","common_cold","pain","fever","antibiotics"]
OUT_DIR = "data/combined_calibration_plots"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH_BASE = os.path.join(OUT_DIR, "combined_calibration_supervised")

# Color palette
COLOR_MAP = {
    "Qwen SFT-CeRTS":  "#6A3D9A",  # dark violet
    "Llama SFT-CeRTS": "#E3A018",  # amber
    "Qwen Probe":      "#1B9E9E",  # teal
    "Llama Probe":     "#C94C7C",  # rose
    "ClinBERT": "#E64A19",  # deep orange
    "BERT":     "#000000",  # black
}

# Legend / plotting order
LINE_ORDER = ["Qwen SFT-CeRTS", "Llama SFT-CeRTS", "Qwen Probe", "Llama Probe", "ClinBERT", "BERT"]

NUM_BINS = 10  # used for BOTH schemes

# ---------- Helpers ----------
def stack_all_features(df: pd.DataFrame, source_kind: str):
    """Stack confidence and correctness across all features."""
    confs, corrects = [], []
    for feat in FEATURES:
        truth = df.get(feat)
        pred  = df.get(f"LLM_{feat}")
        if truth is None or pred is None:
            continue
        conf_col = f"probe_conf_{feat}" if source_kind == "probe" else f"{feat}_conf"
        conf = df.get(conf_col)
        if conf is None:
            continue
        mask = (~pd.isna(truth)) & (~pd.isna(pred)) & (~pd.isna(conf))
        if not mask.any():
            continue
        t = truth[mask].astype(str).values
        p = pred[mask].astype(str).values
        c = np.clip(conf[mask].astype(float).values, 0.0, 1.0)
        corr = (t == p).astype(float)
        confs.append(c)
        corrects.append(corr)
    if not confs:
        return np.array([]), np.array([])
    return np.concatenate(confs), np.concatenate(corrects)

def cal_curve_equal_mass(confs: np.ndarray, corrects: np.ndarray, num_bins: int = NUM_BINS):
    """Return (avg_conf, avg_acc) using equal-mass (quantile) bins."""
    if confs.size == 0:
        return [], []
    order = np.argsort(confs)
    s_sorted = confs[order]
    c_sorted = corrects[order].astype(float)
    n = s_sorted.size
    bin_size = n // num_bins
    remainder = n % num_bins
    xs, ys = [], []
    start = 0
    for i in range(num_bins):
        cur = bin_size + (1 if i < remainder else 0)
        end = start + cur
        if cur <= 0:
            continue
        bin_conf = s_sorted[start:end]
        bin_corr = c_sorted[start:end]
        xs.append(float(bin_conf.mean()))
        ys.append(float(bin_corr.mean()))
        start = end
    return xs, ys

def cal_curve_equal_width(confs: np.ndarray, corrects: np.ndarray, num_bins: int = NUM_BINS):
    """Return (avg_conf, avg_acc) using equal-width bins over [0,1]."""
    if confs.size == 0:
        return [], []
    confs = np.clip(confs, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    xs, ys = [], []
    # include right edge on last bin
    for i in range(num_bins):
        lo, hi = edges[i], edges[i+1]
        if i < num_bins - 1:
            mask = (confs >= lo) & (confs < hi)
        else:
            mask = (confs >= lo) & (confs <= hi)
        if mask.sum() == 0:
            continue  # skip empty bin
        xs.append(float(confs[mask].mean()))
        ys.append(float(corrects[mask].mean()))
    return xs, ys

def infer_kind(label: str):
    return "probe" if "Probe" in label else "sft"

def plot_and_save(curves_dict, out_path):
    plt.clf()
    plt.figure(figsize=(6, 4.5))
    for label in LINE_ORDER:
        x, y = curves_dict.get(label, ([], []))
        if not x:
            continue
        plt.plot(x, y, marker='o', linewidth=2, markersize=5,
                 label=label, color=COLOR_MAP[label])
    # diagonal
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', linewidth=1)
    plt.xlabel("Predicted Confidence (ALL FEATURES)", fontsize=10)
    plt.ylabel("Observed Accuracy", fontsize=10)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
               ncol=4, frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined calibration plot: {out_path}")

# ---------- Load once; compute both schemes ----------
data = {}
for label in LINE_ORDER:
    df = pd.read_csv(paths[label])
    kind = infer_kind(label)
    confs, corrects = stack_all_features(df, kind)
    data[label] = (confs, corrects)

# Build curves for equal-mass
curves_mass = {}
for label in LINE_ORDER:
    confs, corrects = data[label]
    curves_mass[label] = cal_curve_equal_mass(confs, corrects, num_bins=NUM_BINS)

# Build curves for equal-width
curves_width = {}
for label in LINE_ORDER:
    confs, corrects = data[label]
    curves_width[label] = cal_curve_equal_width(confs, corrects, num_bins=NUM_BINS)

# ---------- Save both figures ----------
plot_and_save(curves_mass,  f"{OUT_PATH_BASE}__equal_mass.png")
plot_and_save(curves_width, f"{OUT_PATH_BASE}__equal_width.png")

# quick summary
for scheme_name, curves in [("equal_mass", curves_mass), ("equal_width", curves_width)]:
    for label, (x, _) in curves.items():
        print(f"{scheme_name} | {label}: {len(x)} bins plotted")

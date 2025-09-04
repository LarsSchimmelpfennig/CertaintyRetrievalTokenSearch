import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('output_files\CeRTS\CeRTS_performance.csv')

# Desired fixed model order
desired_model_order = ['Qwen-2.5', 'Llama-3.1', 'Hermes-3', 'Phi-4', 'Mixtral', 'Gemma-2', 'Mistral', 'DeepSeek-R1']

for metric, metric_label, cmap_name in [
        ('brier_score', 'CeRTS Brier Score', 'viridis_r'),
        ('AUROC', 'CeRTS AUROC', 'viridis'),
        ('accuracy', 'Feature Extraction Accuracy', 'viridis'),
        ('confidence_acc_diff', 'CeRTS Overconfidence Score', sns.diverging_palette(220, 20, as_cmap=True)),
        ('avg_num_equiv_runs', 'CeRTS Average # Equivalent Runs', 'viridis_r'),
        ('avg_prop_missing', "Average 'Missing' Proportion", 'viridis_r')]:

    print(metric)
    # Pivot data
    heatmap_data = df.pivot(index="model", columns="feature", values=metric)

    # Optional: Rename feature labels for better display
    feature_renames = {
        "Age": "Age",
        "First treatment date": "First Treatment\nDate",
        "Number of discharge medications": "# Discharge\nMedications",
        "Mention of lung cancer_carcinoma": "Lung Cancer\nMention",
        "Blood pressure value at discharge": "Blood Pressure\nat Discharge",
        "Treated with immunotherapy": "Immunotherapy"
    }

    model_renames = {
        'Qwen2.5-7B-Instruct-1M': 'Qwen-2.5',
        'gemma-2-9b-it': 'Gemma-2',
        'Phi-4-mini-instruct': 'Phi-4',
        'Llama-3.1-8B-Instruct': 'Llama-3.1',
        'Mistral-Small-24B-Instruct-2501': 'Mistral',
        'DeepSeek-R1-Distill-Llama-8B': 'DeepSeek-R1',
        'Mixtral-8x7B-Instruct-v0.1': 'Mixtral',
        'Hermes-3-Llama-3.1-8B': 'Hermes-3'
    }

    column_order = [
        "# Discharge\nMedications",
        "First Treatment\nDate",
        "Lung Cancer\nMention",
        "Immunotherapy",
        "Blood Pressure\nat Discharge",
        "Age"
    ]

    # Rename and reindex
    heatmap_data = heatmap_data.rename(index=model_renames)
    heatmap_data = heatmap_data.rename(columns=feature_renames)
    heatmap_data = heatmap_data.loc[desired_model_order]
    heatmap_data = heatmap_data[column_order]

    # Create clustermap with fixed row order and no clustering
    g = sns.clustermap(
        heatmap_data,
        row_cluster=False,
        col_cluster=False,
        cmap=cmap_name,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        figsize=(12, 6),
        annot_kws={"fontsize": 12},
        cbar_kws={"label": metric_label}
    )

    # Align colorbar exactly with the heatmap
    heatmap_pos = g.ax_heatmap.get_position()
    g.cax.set_position([
        heatmap_pos.x1 + 0.105,  # x: just to the right of heatmap
        heatmap_pos.y0,         # y: align with bottom
        0.02,                   # width
        heatmap_pos.height      # height
    ])

    # Formatting
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=0, fontsize=12)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=12)
    g.ax_heatmap.set_xlabel("")  # Remove axis label 'feature'
    g.ax_heatmap.set_ylabel("")  # Remove axis label 'model'

    # Colorbar font
    g.cax.tick_params(labelsize=10)
    g.cax.set_ylabel(metric_label, fontsize=12)

    # Save the figure
    g.figure.savefig(f'output_files\CeRTS\heatmaps\{metric}.png', dpi=600, bbox_inches='tight')

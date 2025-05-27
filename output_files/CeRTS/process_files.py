import os

# Define the directory relative to the script
input_dir = os.path.join(os.path.dirname(__file__), "CeRTS_calibration_plots")

# Ensure the directory exists
if not os.path.isdir(input_dir):
    raise FileNotFoundError(f"Directory not found: {input_dir}")

# Loop through all PNG files in the directory
for filename in os.listdir(input_dir):
    if not filename.endswith(".png"):
        continue

    parts = filename.split("top2_delta_calibration_10_bins_")
    if len(parts) != 2:
        print(f"[SKIP] No matching pattern in: {filename}")
        continue

    feature = parts[0].rstrip("_")
    rest = parts[1]

    if "_thresh_" not in rest:
        print(f"[SKIP] Missing '_thresh_' in: {filename}")
        continue

    model_name = rest.split("_thresh_")[0]
    new_name = f"{feature}_{model_name}.png"

    src = os.path.join(input_dir, filename)
    dst = os.path.join(input_dir, new_name)

    if not os.path.exists(src):
        print(f"[ERROR] Source file does not exist:\n  {src}")
        continue

    if src != dst:
        print(f"Renaming:\n  FROM: {filename}\n  TO:   {new_name}")
        os.rename(src, dst)

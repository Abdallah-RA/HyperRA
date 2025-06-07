#!/usr/bin/env python3
import os
import sys
import subprocess

# -- ensure required packages --
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "matplotlib", "numpy"])
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

# -- load CSVs --
gpu_csv = "results.csv"
npu_csv = "results_npu.csv"
gpu = pd.read_csv(gpu_csv)
npu = pd.read_csv(npu_csv)

# -- normalize model names in both --
def clean_model(name):
    # remove exported/ prefix and .pt suffix if present
    return name.replace("exported/", "").replace(".pt", "")

gpu["model"] = gpu["model"].apply(clean_model)
npu["model"] = npu["model"].apply(clean_model)

# -- merge --
df = pd.merge(gpu, npu, on="model", suffixes=("_gpu","_npu"))

# -- prepare plots directory --
os.makedirs("plots", exist_ok=True)

# -- which metrics to plot: (column_base, y_label, filename) --
plots = [
    ("avg_latency_ms",        "Average Latency (ms)",          "latency.png"),
    ("avg_delta_power_w",     "Average Δ Power (W)",           "power.png"),
    ("throughput_items_per_s","Throughput (items/s)",          "throughput.png"),
    ("area_used_mm2",         "Area Used (mm²)",               "area.png"),
]

x = np.arange(len(df))
width = 0.35

for key, ylabel, fname in plots:
    plt.figure(figsize=(10,6))
    plt.bar(x - width/2, df[f"{key}_gpu"], width, label="GPU")
    plt.bar(x + width/2, df[f"{key}_npu"], width, label="NPU")
    plt.xticks(x, df["model"], rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(f"GPU vs NPU Comparison: {ylabel}")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join("plots", fname)
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()

print("All plots generated in ./plots/")

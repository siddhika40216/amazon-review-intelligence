
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "reports" / "model_comparison.csv"
OUTPUT_PATH = ROOT / "reports" / "model_comparison.png"

df = pd.read_csv(CSV_PATH)

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
models = df["Model"].tolist()

x = np.arange(len(models))
width = 0.18

fig, ax = plt.subplots(figsize=(11, 6))

for i, metric in enumerate(metrics):
    values = df[metric].to_numpy()
    ax.bar(
        x + (i - 1.5) * width,
        values,
        width,
        label=metric
    )

ax.set_title("Amazon Review Sentiment Models: Performance Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("Machine Learning Models")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 1)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.show()

print(f"Chart saved to: {OUTPUT_PATH}")

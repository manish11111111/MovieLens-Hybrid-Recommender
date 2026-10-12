
import matplotlib.pyplot as plt
import numpy as np

# Model ra metrics tayar garam
metrics = ["Precision@10", "Recall@10", "F1@10"]

cf_scores = [0.0146, 0.0126, 0.0112]
hybrid_scores = [0.0276, 0.0246, 0.0210]

x = np.arange(len(metrics))
width = 0.35

# Chart banaam
fig, ax = plt.subplots(figsize=(9, 5.5))

cf_bars = ax.bar(
    x - width / 2,
    cf_scores,
    width,
    label="Collaborative Filtering"
)

hybrid_bars = ax.bar(
    x + width / 2,
    hybrid_scores,
    width,
    label="Hybrid (50/50)"
)

# Labels ra title halam
ax.set_title("MovieLens Recommender: Test Results")
ax.set_ylabel("Score")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(axis="y", alpha=0.25)

# Bar mathi score dekhaam
ax.bar_label(cf_bars, fmt="%.4f", padding=3)
ax.bar_label(hybrid_bars, fmt="%.4f", padding=3)

fig.tight_layout()

# Chart save garam
plt.savefig("test_results.png", dpi=200, bbox_inches="tight")
plt.show()

print("Test results chart saved as test_results.png")

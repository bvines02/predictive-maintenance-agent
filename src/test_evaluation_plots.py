"""Stage 14h - Step 11: plots for the official FD001 test-set evaluation.

Reuses src/error_analysis_plots.py and src/uncertainty_plots.py wherever the
column names line up (see src/evaluate_test_set.py's naming choices) -
this module only adds the one new plot those don't already cover: the
health-state confusion matrix as a heatmap.
"""

import matplotlib

matplotlib.use("Agg")  # headless: save figures to disk, never open a window
import matplotlib.pyplot as plt
import pandas as pd

from src.health_state import HEALTH_STATES


def plot_health_state_confusion_matrix(matrix: pd.DataFrame, save_path) -> None:
    """Heatmap of the actual-vs-predicted health-state confusion matrix (Step 8's states)."""
    matrix = matrix.reindex(index=HEALTH_STATES, columns=HEALTH_STATES, fill_value=0)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(matrix.values, cmap="Blues")

    ax.set_xticks(range(len(HEALTH_STATES)))
    ax.set_yticks(range(len(HEALTH_STATES)))
    ax.set_xticklabels(HEALTH_STATES)
    ax.set_yticklabels(HEALTH_STATES)
    ax.set_xlabel("Predicted state")
    ax.set_ylabel("Actual state")
    ax.set_title("FD001 test set: health-state confusion matrix")

    max_value = matrix.values.max() if matrix.values.max() > 0 else 1
    for i in range(len(HEALTH_STATES)):
        for j in range(len(HEALTH_STATES)):
            value = matrix.values[i, j]
            color = "white" if value > max_value / 2 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)

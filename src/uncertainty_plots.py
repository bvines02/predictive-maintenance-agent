"""Stage 14h - plots for Random Forest ensemble-disagreement uncertainty."""

import matplotlib

matplotlib.use("Agg")  # headless: save figures to disk, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_engine_uncertainty(df: pd.DataFrame, unit_number: int, save_path) -> None:
    """Actual vs. predicted RUL for one engine, with the ensemble range shaded.

    A narrow shaded band means the trees agreed closely on that cycle; a
    wide band means they scattered. Watch what happens to the band's width
    as the engine approaches failure, not just where the lines sit.
    """
    engine = df[df["unit_number"] == unit_number].sort_values("time_cycles")

    plt.figure(figsize=(8, 5))
    plt.fill_between(
        engine["time_cycles"], engine["prediction_p10"], engine["prediction_p90"],
        color="tab:blue", alpha=0.18, label="Ensemble range (p10-p90, NOT a calibrated interval)",
    )
    plt.plot(engine["time_cycles"], engine["actual_rul"], color="black", label="Actual RUL (capped)")
    plt.plot(
        engine["time_cycles"], engine["predicted_rul"], color="tab:blue", linestyle="--",
        label="Predicted RUL (ensemble mean)",
    )
    plt.xlabel("time_cycles")
    plt.ylabel("RUL")
    plt.title(f"Engine {unit_number}: prediction with ensemble disagreement band")
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_error_vs_uncertainty(df: pd.DataFrame, save_path, n_bins: int = 8) -> None:
    """Scatter of absolute_error vs. prediction_std, plus a binned mean-error line.

    The scatter shows every row; the line answers "on average, do rows
    where the trees disagreed more end up more wrong?" without eyeballing
    a noisy cloud.
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(df["prediction_std"], df["absolute_error"], alpha=0.25, s=10, color="tab:blue")

    bins = pd.qcut(df["prediction_std"], q=n_bins, duplicates="drop")
    binned = df.groupby(bins, observed=True)["absolute_error"].mean()
    bin_centers = [interval.mid for interval in binned.index]
    plt.plot(bin_centers, binned.values, color="black", marker="o", label=f"Mean absolute error ({len(binned)} bins)")

    plt.xlabel("prediction_std (tree disagreement)")
    plt.ylabel("absolute_error")
    plt.title("Prediction error vs. ensemble disagreement")
    plt.legend()
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

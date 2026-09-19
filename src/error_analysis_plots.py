"""Stage 14h (diagnostic) - plots for RUL error analysis.

Three simple, standard diagnostic plots, kept deliberately basic (see
CLAUDE.md's "keep it simple" principle) rather than combined into one
dense figure, so each one answers exactly one question.
"""

import matplotlib

matplotlib.use("Agg")  # headless: save figures to disk, never open a window
import matplotlib.pyplot as plt
import pandas as pd


def plot_error_histogram(predictions_df: pd.DataFrame, save_path) -> None:
    """Histogram of (predicted_rul - actual_rul), with a reference line at zero.

    Left of zero: conservative predictions (predicted less life than the
    engine actually had). Right of zero: optimistic predictions (predicted
    more life than the engine actually had). A histogram centred on zero
    and roughly symmetric suggests no strong systematic bias; a histogram
    shifted to one side suggests the model is consistently too
    conservative or too optimistic.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(predictions_df["error"], bins=40, color="tab:blue", edgecolor="white")
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Prediction error (predicted_rul - actual_rul)")
    plt.ylabel("Number of validation rows")
    plt.title("Error distribution: left = conservative, right = optimistic")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_actual_vs_predicted(predictions_df: pd.DataFrame, save_path) -> None:
    """Scatter of predicted RUL vs. actual RUL, with a y=x reference line.

    A point on the line is a perfect prediction. Points consistently above
    the line (predicted > actual) show systematic over-prediction/optimism;
    points consistently below the line show systematic under-prediction.
    Points that fan out further from the line as actual RUL grows would
    indicate errors that grow with the size of the true RUL itself, rather
    than errors that grow near failure specifically (see the error-vs-actual
    plot for that).
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(
        predictions_df["actual_rul"], predictions_df["predicted_rul"], alpha=0.3, s=10, color="tab:blue"
    )
    axis_max = max(predictions_df["actual_rul"].max(), predictions_df["predicted_rul"].max())
    plt.plot([0, axis_max], [0, axis_max], color="black", linewidth=1, label="y = x (perfect prediction)")
    plt.xlabel("Actual RUL (capped)")
    plt.ylabel("Predicted RUL")
    plt.title("Actual vs. predicted RUL")
    plt.legend()
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_error_vs_actual_rul(predictions_df: pd.DataFrame, save_path) -> None:
    """Scatter of prediction error vs. actual RUL, with a reference line at zero.

    This is the plot most directly aimed at the operational question this
    step cares about: do errors get worse (larger magnitude, or more
    one-sided) as actual RUL approaches zero - i.e. right where a wrong
    answer matters most - rather than just asking whether errors are big
    on average across the whole range.
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(
        predictions_df["actual_rul"], predictions_df["error"], alpha=0.3, s=10, color="tab:blue"
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Actual RUL (capped)")
    plt.ylabel("Prediction error (predicted_rul - actual_rul)")
    plt.title("Prediction error vs. actual RUL")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

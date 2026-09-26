"""Stage 14h - health-state report from the Step 7 validation predictions.

Reads results/validation_predictions.csv (written by src.run_error_analysis,
temporal model, same 20 validation engines as Steps 5-7) and does NOT retrain
anything. Adds health states, prints the diagnostics, saves results/ files
and trajectory plots.
"""

import pandas as pd

from src.config import (
    HEALTH_STATE_ARTIFACTS_DIR,
    HEALTH_STATE_CONFUSION_MATRIX_PATH,
    HEALTH_STATE_PREDICTIONS_PATH,
    HEALTH_THRESHOLDS,
    RESULTS_DIR,
    VALIDATION_PREDICTIONS_PATH,
)
from src.health_state import (
    add_health_states,
    conservative_over_classifications,
    serious_under_classifications,
    state_confusion_matrix,
    state_distribution,
    state_metrics,
)
from src.health_state_plots import plot_health_state_trajectory


def load_validation_predictions(path=VALIDATION_PREDICTIONS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Generate it first with: python -m src.run_error_analysis"
        )
    return pd.read_csv(path)


def _print_section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


if __name__ == "__main__":
    predictions = load_validation_predictions()
    df = add_health_states(predictions, HEALTH_THRESHOLDS)

    _print_section(f"Thresholds (predicted RUL <= value): {HEALTH_THRESHOLDS}")

    _print_section("Predicted health-state distribution")
    print(state_distribution(df["health_state"]).to_string())
    _print_section("Actual health-state distribution")
    print(state_distribution(df["actual_health_state"]).to_string())

    _print_section("State confusion matrix (rows = actual, columns = predicted)")
    matrix = state_confusion_matrix(df)
    print(matrix.to_string())

    _print_section("State-level metrics")
    for key, value in state_metrics(df).items():
        print(f"  {key}: {value:.3f}")

    _print_section("Most serious under-classifications (model less concerned than reality)")
    under = serious_under_classifications(df)
    print(under.to_string(index=False) if len(under) else "  none")

    _print_section("Most conservative over-classifications (model more concerned than reality)")
    print(conservative_over_classifications(df).to_string(index=False))

    _print_section("Under-classification pair counts (actual -> predicted)")
    pairs = (
        df[df["state_error"] < 0]
        .groupby(["actual_health_state", "health_state"])
        .size()
        .sort_values(ascending=False)
    )
    print(pairs.to_string())

    plot_units = [int(df["unit_number"].min())]
    worst = serious_under_classifications(df, n=1)
    if len(worst) and int(worst.loc[0, "unit_number"]) not in plot_units:
        plot_units.append(int(worst.loc[0, "unit_number"]))
    for unit in plot_units:
        path = HEALTH_STATE_ARTIFACTS_DIR / f"engine_{unit}_health_state.png"
        plot_health_state_trajectory(df, unit, path)
        print(f"\nSaved health-state plot for engine {unit} to {path}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_columns = [
        "unit_number", "time_cycles", "actual_rul", "predicted_rul", "error",
        "health_state", "actual_health_state", "predicted_severity", "actual_severity", "state_error",
    ]
    df[save_columns].to_csv(HEALTH_STATE_PREDICTIONS_PATH, index=False)
    matrix.to_csv(HEALTH_STATE_CONFUSION_MATRIX_PATH)
    print(f"Saved {HEALTH_STATE_PREDICTIONS_PATH}")
    print(f"Saved {HEALTH_STATE_CONFUSION_MATRIX_PATH}")

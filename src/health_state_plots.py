"""Stage 14h - health-state trajectory plots for one engine."""

import matplotlib

matplotlib.use("Agg")  # headless: save figures to disk, never open a window
import matplotlib.pyplot as plt
import pandas as pd

from src.config import HEALTH_THRESHOLDS
from src.health_state import HEALTH_STATES, STATE_SEVERITY


def plot_health_state_trajectory(
    df: pd.DataFrame, unit_number: int, save_path, thresholds: dict = HEALTH_THRESHOLDS
) -> None:
    """Two stacked panels for one engine, sharing the time_cycles axis.

    Top: true vs predicted RUL, with horizontal lines at the state thresholds.
    Bottom: actual vs predicted health state as an ordinal severity step plot
    (0 = HEALTHY ... 3 = ACTION). Where the predicted line sits BELOW the
    actual line, the model is less concerned than reality.
    """
    engine = df[df["unit_number"] == unit_number].sort_values("time_cycles")

    fig, (ax_rul, ax_state) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax_rul.plot(engine["time_cycles"], engine["actual_rul"], color="black", label="Actual RUL (capped)")
    ax_rul.plot(
        engine["time_cycles"], engine["predicted_rul"], color="tab:blue", linestyle="--", label="Predicted RUL"
    )
    for name in ("watch", "plan", "action"):
        ax_rul.axhline(thresholds[name], color="tab:red", linewidth=0.8, linestyle=":")
        ax_rul.text(engine["time_cycles"].min(), thresholds[name] + 1, f"{name.upper()} <= {thresholds[name]}", fontsize=8)
    ax_rul.set_ylabel("RUL (cycles)")
    ax_rul.set_title(f"Engine {unit_number}: RUL and health state")
    ax_rul.legend(loc="upper right")

    ax_state.step(
        engine["time_cycles"], engine["actual_severity"], where="post", color="black", label="Actual state"
    )
    ax_state.step(
        engine["time_cycles"],
        engine["predicted_severity"],
        where="post",
        color="tab:blue",
        linestyle="--",
        label="Predicted state",
    )
    ax_state.set_yticks([STATE_SEVERITY[s] for s in HEALTH_STATES])
    ax_state.set_yticklabels(HEALTH_STATES)
    ax_state.set_xlabel("time_cycles")
    ax_state.set_ylabel("Health state")
    ax_state.legend(loc="upper left")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)

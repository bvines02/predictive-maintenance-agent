"""Pipeline Explorer - export real V2 model output as the app's static data.

    python -m src.export_pipeline_fixture

Responsible for:
- Training the exact Step 6/7 temporal Random Forest ("Model B") on the
  exact Step 5 engine split, via the same helpers every V2 report uses
- Scoring every cycle of all 100 train_FD001 engines, with the per-row
  spread of the 200 trees behind each prediction (Step 10)
- Writing app/public/data/index.json (model, sensors, features, engine
  list) and app/public/data/engines/<unit>.json (one engine's whole life,
  column-oriented) so the React app needs no Python at runtime
- Regenerating the Python<->TypeScript parity cases alongside

Not responsible for:
- Health states or decisions. Those depend on parameters the user edits in
  the app, so the app computes them (app/src/pipeline/rules.ts); only the
  model's output is frozen here.

IMPORTANT - in-sample vs held-out: 80 of the 100 engines are the model's
own TRAINING engines, so their predictions are fitted, not forecast, and
look far better than the model really is. Every engine carries its `split`
and the app labels it; only the 20 "validation" engines show honest error.
"""

import json
import shutil

import numpy as np
import pandas as pd

from src.cmapss_loader import SENSOR_COLUMNS
from src.config import (
    HEALTH_THRESHOLDS,
    OPERATIONAL_SETTING_METADATA,
    PROJECT_ROOT,
    SENSOR_METADATA,
)
from src.decision_engine import ACTION_MEANINGS
from src.evaluate_rul import evaluate_rul_predictions
from src.pipeline_explorer import (
    DEFAULT_LEAD_TIME_CYCLES,
    EXPLORER_RULE_IDS,
    LEAD_TIME_BUFFER_CYCLES,
    write_parity_cases,
)
from src.rul import DEFAULT_RUL_CAP
from src.run_error_analysis import _prepare_split
from src.sensor_screening import screen_sensors
from src.train_rul_baseline import RANDOM_STATE, TARGET_COLUMN, build_modeling_dataset, train_baseline_rf
from src.train_rul_temporal import ROLLING_WINDOWS, TREND_WINDOW, classify_feature
from src.uncertainty import (
    assign_model_confidence,
    assign_uncertainty_band,
    compute_uncertainty_thresholds,
    extract_tree_predictions,
)

FIXTURE_DIR = PROJECT_ROOT / "app" / "public" / "data"
TOP_FEATURE_COUNT = 12
TREE_QUANTILES = (0, 10, 25, 50, 75, 90, 100)


def _round(values, decimals: int) -> list:
    return [round(float(v), decimals) for v in values]


def _feature_label(feature: str) -> dict:
    """Split e.g. 'sensor_11_roll_mean_10' into its sensor and its transform."""
    for sensor in sorted(SENSOR_METADATA, key=len, reverse=True):
        if feature == sensor or feature.startswith(sensor + "_"):
            return {"sensor": sensor, "symbol": SENSOR_METADATA[sensor]["symbol"],
                    "transform": feature[len(sensor) + 1:] or "raw"}
    return {"sensor": None, "symbol": feature, "transform": "raw"}


def score_all_engines():
    """Fit Model B once, then score every row of every engine (train + validation)."""
    train_split_df, val_split_df, _, feature_columns = _prepare_split()
    X_train, y_train = build_modeling_dataset(train_split_df, feature_columns)
    model = train_baseline_rf(X_train, y_train)

    # Step 10: confidence cut points come from TRAINING rows only.
    train_trees = extract_tree_predictions(model, X_train)
    uncertainty_thresholds = compute_uncertainty_thresholds(pd.Series(train_trees.std(axis=1, ddof=0)))

    frames = []
    for split_name, split_df in (("train", train_split_df), ("validation", val_split_df)):
        X, _ = build_modeling_dataset(split_df, feature_columns)
        trees = extract_tree_predictions(model, X)
        scored = split_df.reset_index(drop=True).copy()
        scored["split"] = split_name
        scored["predicted_rul"] = model.predict(X)
        scored["prediction_std"] = trees.std(axis=1, ddof=0)
        for q in TREE_QUANTILES:
            scored[f"tree_p{q}"] = np.percentile(trees, q, axis=1)
        scored["model_confidence"] = assign_model_confidence(
            assign_uncertainty_band(scored["prediction_std"], uncertainty_thresholds)
        ).to_numpy()
        frames.append(scored)

    scored_df = pd.concat(frames).sort_values(["unit_number", "time_cycles"]).reset_index(drop=True)
    return model, feature_columns, scored_df, uncertainty_thresholds


def build_index(model, feature_columns, scored_df, uncertainty_thresholds, kept_sensors, top_features) -> dict:
    val = scored_df[scored_df["split"] == "validation"]
    val_metrics = evaluate_rul_predictions(val[TARGET_COLUMN], val["predicted_rul"])

    # Same data Step 4 screens (all of train_FD001 - see get_kept_sensor_columns).
    screening = screen_sensors(scored_df, sensor_columns=SENSOR_COLUMNS)
    sensors = [
        {
            "column": row["sensor"],
            **SENSOR_METADATA[row["sensor"]],
            "status": row["status"],
            "unique_values": int(row["unique_values"]),
        }
        for _, row in screening.iterrows()
    ]

    importances = pd.Series(model.feature_importances_, index=feature_columns)
    by_category = importances.groupby([classify_feature(f) for f in feature_columns]).agg(["sum", "count"])

    engines = (
        scored_df.groupby("unit_number")
        .agg(split=("split", "first"), n_cycles=("time_cycles", "size"))
        .reset_index()
    )

    return {
        "_comment": "Generated by `python -m src.export_pipeline_fixture`. Real model output; do not edit by hand.",
        "dataset": {
            "name": "NASA C-MAPSS FD001 (train_FD001)",
            "n_engines": int(engines.shape[0]),
            "n_rows": int(len(scored_df)),
            "rul_cap": DEFAULT_RUL_CAP,
        },
        "model": {
            "type": "RandomForestRegressor",
            "n_trees": len(model.estimators_),
            "random_state": RANDOM_STATE,
            "target": TARGET_COLUMN,
            "n_features": len(feature_columns),
            "rolling_windows": list(ROLLING_WINDOWS),
            "trend_window": TREND_WINDOW,
            "validation_mae": round(val_metrics["mae"], 2),
            "validation_rmse": round(val_metrics["rmse"], 2),
            "feature_categories": {
                category: {"count": int(row["count"]), "importance": round(float(row["sum"]), 4)}
                for category, row in by_category.iterrows()
            },
            "top_features": [
                {
                    "feature": feature,
                    "category": classify_feature(feature),
                    "importance": round(float(importances[feature]), 4),
                    **_feature_label(feature),
                }
                for feature in top_features
            ],
        },
        "operational_settings": OPERATIONAL_SETTING_METADATA,
        "sensors": sensors,
        "kept_sensors": kept_sensors,
        "uncertainty": {
            "prediction_std_low_max": round(uncertainty_thresholds["low_max"], 3),
            "prediction_std_medium_max": round(uncertainty_thresholds["medium_max"], 3),
        },
        "defaults": {
            "health_thresholds": HEALTH_THRESHOLDS,
            "maintenance_lead_time_cycles": DEFAULT_LEAD_TIME_CYCLES,
            "lead_time_buffer_cycles": LEAD_TIME_BUFFER_CYCLES,
        },
        "rule_ids": list(EXPLORER_RULE_IDS),
        "action_meanings": {action.name: meaning for action, meaning in ACTION_MEANINGS.items()},
        "engines": [
            {"unit": int(r.unit_number), "split": r.split, "n_cycles": int(r.n_cycles)}
            for r in engines.itertuples()
        ],
    }


def build_engine(engine_df: pd.DataFrame, kept_sensors, top_features) -> dict:
    return {
        "unit": int(engine_df["unit_number"].iloc[0]),
        "split": engine_df["split"].iloc[0],
        "cycle": engine_df["time_cycles"].astype(int).tolist(),
        "actual_rul": engine_df["rul"].astype(int).tolist(),
        "actual_rul_capped": engine_df["rul_capped"].astype(int).tolist(),
        "predicted_rul": _round(engine_df["predicted_rul"], 2),
        "prediction_std": _round(engine_df["prediction_std"], 2),
        "tree_quantiles": {f"p{q}": _round(engine_df[f"tree_p{q}"], 1) for q in TREE_QUANTILES},
        "model_confidence": engine_df["model_confidence"].tolist(),
        "sensors": {s: _round(engine_df[s], 4) for s in kept_sensors},
        "features": {f: [float(f"{v:.5g}") for v in engine_df[f]] for f in top_features},
    }


def export(fixture_dir=FIXTURE_DIR) -> None:
    model, feature_columns, scored_df, uncertainty_thresholds = score_all_engines()
    kept_sensors = [c for c in feature_columns if c in SENSOR_COLUMNS]
    importances = pd.Series(model.feature_importances_, index=feature_columns)
    top_features = importances.sort_values(ascending=False).head(TOP_FEATURE_COUNT).index.tolist()

    engines_dir = fixture_dir / "engines"
    if engines_dir.exists():
        shutil.rmtree(engines_dir)
    engines_dir.mkdir(parents=True)

    index = build_index(model, feature_columns, scored_df, uncertainty_thresholds, kept_sensors, top_features)
    (fixture_dir / "index.json").write_text(json.dumps(index, indent=1) + "\n")

    for unit, engine_df in scored_df.groupby("unit_number"):
        engine = build_engine(engine_df, kept_sensors, top_features)
        (engines_dir / f"{int(unit)}.json").write_text(json.dumps(engine, separators=(",", ":")))

    write_parity_cases()

    total_kb = sum(p.stat().st_size for p in fixture_dir.rglob("*.json")) / 1024
    print(f"Exported {len(index['engines'])} engines ({index['dataset']['n_rows']} cycles) "
          f"to {fixture_dir.relative_to(PROJECT_ROOT)} - {total_kb:,.0f} KB")
    print(f"Validation MAE {index['model']['validation_mae']}, RMSE {index['model']['validation_rmse']} "
          f"(on {sum(e['split'] == 'validation' for e in index['engines'])} held-out engines)")


if __name__ == "__main__":
    export()

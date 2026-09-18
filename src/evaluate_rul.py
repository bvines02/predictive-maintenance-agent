"""Stage 14f - RUL regression evaluation.

Responsible for:
- Measuring regression quality (MAE, RMSE) for a predicted vs. actual RUL

This is the regression counterpart to src/evaluate.py (V1's classification
metrics). Precision/recall/F1 do not apply to a continuous target - there is
no "positive class" for a predicted cycle count, so this module uses
error-magnitude metrics instead.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_rul_predictions(y_true, y_pred) -> dict:
    """Compute MAE and RMSE for predicted RUL vs. actual RUL.

    RMSE is computed as sqrt(mean_squared_error(...)) rather than passing
    squared=False to mean_squared_error, so this works the same way across
    scikit-learn versions regardless of whether that argument is supported.

    Returns:
        {"mae": float, "rmse": float}
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    return {"mae": mae, "rmse": rmse}

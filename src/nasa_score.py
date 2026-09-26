"""Stage 14h - the PHM08 / C-MAPSS asymmetric scoring function.

Responsible for:
- The standard asymmetric scoring formula from the original C-MAPSS /
  PHM08 Challenge (Saxena, Goebel, Simon & Eklund, 2008), as a SECONDARY
  metric alongside MAE/RMSE - never a replacement for them.

Formula (using this project's error convention, error = predicted - actual):

    d = predicted_rul - actual_rul
    score(d) = exp(-d / 13) - 1   if d < 0   (early / conservative prediction)
    score(d) = exp( d / 10) - 1   if d >= 0  (late / optimistic prediction)
    total_score = sum(score(d) for every engine)

This is the formula consistently cited across the C-MAPSS literature and
the original PHM08 challenge scoring script. It is included here with that
provenance stated explicitly - cross-check it against NASA's own
documentation before relying on it outside this educational project.

Why it exists, and why MAE/RMSE alone aren't enough: MAE and RMSE treat a
20-cycle over-prediction and a 20-cycle under-prediction as equally bad.
Operationally they are not - see Step 7's asymmetry discussion. This score
makes that asymmetry explicit and steep: the exp() growth means a handful
of badly-late (optimistic, d >> 0) predictions can dominate the total score
even if most predictions are good, which is intentional - it is a
worst-case-sensitive metric, not an average-case one like MAE.
"""

import numpy as np


def nasa_cmapss_scores(errors) -> np.ndarray:
    """Per-observation PHM08 score. `errors` = predicted_rul - actual_rul."""
    errors = np.asarray(errors, dtype=float)
    return np.where(errors < 0, np.exp(-errors / 13) - 1, np.exp(errors / 10) - 1)


def nasa_cmapss_score(errors) -> float:
    """Total PHM08 score across all observations (lower is better; 0 = perfect)."""
    return float(nasa_cmapss_scores(errors).sum())

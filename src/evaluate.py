"""Stage 5 - Model evaluation.

Responsible for:
- Measuring model quality on held-out test data
- Reporting precision, recall, and a confusion matrix

In maintenance, a missed failure (false negative) usually costs far more than
an unnecessary inspection (false positive), so accuracy alone is misleading.
"""

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate a trained classifier on held-out data.

    Deliberately does not include plain accuracy: with ~3% failures in this
    dataset, a model that always predicts "no failure" would score ~97%
    accuracy while catching zero real failures. The metrics below focus on
    the failure class specifically.

    Returns:
        A dict with precision, recall, f1, roc_auc, and confusion_matrix.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probability of class 1 (failure)

    return {
        # Of the rows we flagged as failures, how many really were?
        # Low precision = too many false alarms / unnecessary inspections.
        "precision": precision_score(y_test, y_pred, zero_division=0),
        # Of the real failures, how many did we catch?
        # Low recall = missed failures - usually the costlier mistake.
        "recall": recall_score(y_test, y_pred, zero_division=0),
        # Harmonic mean of precision and recall - one number balancing both.
        "f1": f1_score(y_test, y_pred, zero_division=0),
        # How well the model ranks failures above non-failures across all
        # thresholds, not just the default 0.5 cutoff. 0.5 = random, 1.0 = perfect.
        "roc_auc": roc_auc_score(y_test, y_proba),
        # Rows: actual [no-failure, failure]. Columns: predicted [no-failure, failure].
        # confusion_matrix[1][0] = false negatives = missed failures.
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

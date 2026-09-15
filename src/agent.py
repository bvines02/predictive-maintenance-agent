"""Agent orchestration.

Responsible for running the full pipeline for one asset, in this order:
    predict.py -> decision_rules.py -> asset_context.py -> explainer.py
and returning a recommendation flagged for human review when risk is high.

This module coordinates the steps; it contains no ML, rule, or LLM logic itself.
"""

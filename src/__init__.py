"""Predictive maintenance agent.

Pipeline:
    telemetry -> features -> ML model -> risk score
    -> deterministic rules -> LLM explanation -> human review
"""

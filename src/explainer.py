"""Stage 10 - LLM explanation layer.

Responsible for:
- Taking the risk score, rule decision, and asset context as INPUT
- Asking an LLM to explain them in plain maintenance language

This is the only module allowed to call an LLM. It receives decisions that
have already been made (by predict.py + decision_rules.py); it must not
change the risk level, the recommended action, or whether human review is
required. The prompt below says so explicitly, and nothing this module
returns is fed back into decide_maintenance_action() - it's a dead end that
produces text for a human to read, not a decision the pipeline acts on.
"""

import os

import anthropic
from dotenv import load_dotenv

load_dotenv()  # populates os.environ from a local .env file, if present

MODEL_NAME = "claude-sonnet-5"


def explanation_available() -> bool:
    """True if an API key is configured, so callers can check before trying."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def generate_explanation(prediction: dict, decision: dict, asset: dict) -> str:
    """Ask an LLM to explain an already-made decision in maintenance language.

    Args:
        prediction: Output of predict.predict_failure_risk()
            ({"failure_probability": float, "risk_band": str}).
        decision: Output of decision_rules.decide_maintenance_action()
            ({"recommended_action", "urgency", "human_review_required",
            "rationale_code"}).
        asset: An asset context record from asset_context.load_asset_context().

    Returns:
        A short plain-language explanation for a maintenance reviewer.

    Raises:
        RuntimeError: If ANTHROPIC_API_KEY isn't configured. Callers should
            check explanation_available() first and skip this call entirely
            rather than relying on catching this.
    """
    if not explanation_available():
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set - cannot generate an explanation. "
            "Copy .env.example to .env and add a key, or check "
            "explanation_available() before calling this."
        )

    prompt = f"""You are explaining a maintenance decision to a human reviewer.
The decision has ALREADY been made by deterministic rules - do not change it,
second-guess it, or suggest a different action. Only explain it clearly.

Asset: {asset["asset_id"]} ({asset["asset_type"]}, {asset["system"]})
Criticality: {asset["criticality"]}
Single point of failure: {asset["single_point_of_failure"]}
Redundancy: {asset["redundancy"]}
Known failure modes: {", ".join(asset["known_failure_modes"])}
Operational workaround: {asset["operational_workaround"]}

Model prediction:
Failure probability: {prediction["failure_probability"]}
Risk band: {prediction["risk_band"]}

Decision (already made, do not change):
Recommended action: {decision["recommended_action"]}
Urgency: {decision["urgency"]}
Human review required: {decision["human_review_required"]}
Rationale code: {decision["rationale_code"]}

Write 2-4 short sentences in plain maintenance language explaining why this
risk level and action make sense together, referencing the asset's specific
situation (criticality, redundancy, failure modes). Do not invent numbers or
facts not given above."""

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


if __name__ == "__main__":
    if not explanation_available():
        print("ANTHROPIC_API_KEY not set - copy .env.example to .env and add a key.")
    else:
        example_prediction = {"failure_probability": 0.97, "risk_band": "High"}
        example_decision = {
            "recommended_action": "Immediate engineering review",
            "urgency": "Immediate",
            "human_review_required": True,
            "rationale_code": "HIGH_RISK_CRITICAL_SPOF_NO_REDUNDANCY",
        }
        example_asset = {
            "asset_id": "pump-01",
            "asset_type": "Centrifugal pump",
            "system": "Primary feedwater",
            "criticality": "High",
            "single_point_of_failure": True,
            "redundancy": "None",
            "known_failure_modes": ["Bearing wear", "Seal failure", "Cavitation"],
            "operational_workaround": "None - process must shut down if this pump fails",
        }
        print(generate_explanation(example_prediction, example_decision, example_asset))

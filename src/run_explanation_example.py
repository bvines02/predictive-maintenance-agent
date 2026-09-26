"""Stage 10 - worked example of the LLM explanation layer end to end.

Wires together the pieces explainer.py depends on but doesn't itself load:
- decision_rules.decide_maintenance_action() makes the decision
- asset_context.load_asset_context() supplies the asset's context
- explainer.generate_explanation() explains that already-made decision

Does not touch predict.py / the trained model - the failure probability
below is a representative stand-in so this example runs without first
training a model (see predict.py if you want a real prediction instead).
"""

from src.asset_context import has_redundancy, load_asset_context
from src.decision_rules import decide_maintenance_action
from src.explainer import explanation_available, generate_explanation

EXAMPLE_ASSET_ID = "pump-01"
EXAMPLE_PREDICTION = {"failure_probability": 0.82, "risk_band": "High"}


def _section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


if __name__ == "__main__":
    asset = load_asset_context(EXAMPLE_ASSET_ID)
    decision = decide_maintenance_action(
        failure_probability=EXAMPLE_PREDICTION["failure_probability"],
        risk_band=EXAMPLE_PREDICTION["risk_band"],
        asset_criticality=asset["criticality"],
        is_single_point_of_failure=asset["single_point_of_failure"],
        redundancy_available=has_redundancy(asset),
    )

    _section(f"Asset context - {EXAMPLE_ASSET_ID}")
    print(asset)

    _section("Model prediction (representative, not from a trained model)")
    print(EXAMPLE_PREDICTION)

    _section("Decision (decision_rules.decide_maintenance_action)")
    print(decision)

    _section("LLM explanation (explainer.generate_explanation)")
    if not explanation_available():
        print("ANTHROPIC_API_KEY not set - copy .env.example to .env and add a key.")
    else:
        print(generate_explanation(EXAMPLE_PREDICTION, decision, asset))

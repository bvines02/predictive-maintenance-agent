"""Stage 12 - Simple Streamlit UI.

Responsible for:
- Letting a user enter telemetry and pick an asset
- Calling the same prediction/decision logic the API uses
- Showing the risk score, risk band, maintenance recommendation, and (if
  configured) an LLM explanation

Deliberately calls src.predict / src.decision_rules / src.asset_context
directly rather than going over HTTP to src.api - fine for a single-user
local demo, since the UI and the model run in the same process. A "real"
frontend (a separate team, a separate deployment, a non-Python client)
should instead call the FastAPI endpoints the way the curl examples do -
see the README's "How this could become a proper frontend" note. That
swap would only touch this file; predict.py/decision_rules.py wouldn't
need to change at all, which is the payoff of having built them first.
"""

import sys
from pathlib import Path

# `streamlit run src/ui.py` puts this file's own directory (src/) on
# sys.path, not the project root - so `import src...` fails no matter what
# directory you launch it from. Add the project root explicitly, once,
# before importing anything from src/. Same fix as notebooks/01_exploration.ipynb.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from src.asset_context import has_redundancy, list_asset_ids, load_asset_context
from src.decision_rules import decide_maintenance_action
from src.explainer import explanation_available, generate_explanation
from src.predict import load_model, predict_failure_risk

st.set_page_config(page_title="Predictive Maintenance Agent", page_icon="🔧")


@st.cache_resource
def _get_model():
    # Cached across reruns (Streamlit reruns the whole script on every
    # interaction) - without this, every button click would reload the
    # 3 MB model file from disk.
    return load_model()


st.title("🔧 Predictive Maintenance Agent")
st.caption(
    "Telemetry -> ML risk score -> deterministic decision rules -> LLM explanation. "
    "The LLM only explains the decision below; it never makes or changes it."
)

asset_ids = list_asset_ids()
asset_id = st.selectbox("Asset", asset_ids)
asset = load_asset_context(asset_id)

with st.expander("Asset context", expanded=False):
    st.write(asset)

st.subheader("Telemetry snapshot")
col1, col2 = st.columns(2)
with col1:
    air_temperature = st.number_input("Air temperature [K]", value=300.0, step=0.1)
    process_temperature = st.number_input("Process temperature [K]", value=310.0, step=0.1)
    rotational_speed = st.number_input("Rotational speed [rpm]", value=1500.0, step=10.0)
with col2:
    torque = st.number_input("Torque [Nm]", value=40.0, step=1.0)
    tool_wear = st.number_input("Tool wear [min]", value=10.0, step=1.0)

if st.button("Predict", type="primary"):
    telemetry = {
        "Air temperature [K]": air_temperature,
        "Process temperature [K]": process_temperature,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }

    prediction = predict_failure_risk(telemetry, model=_get_model())
    decision = decide_maintenance_action(
        failure_probability=prediction["failure_probability"],
        risk_band=prediction["risk_band"],
        asset_criticality=asset["criticality"],
        is_single_point_of_failure=asset["single_point_of_failure"],
        redundancy_available=has_redundancy(asset),
    )

    st.subheader("Result")
    col1, col2 = st.columns(2)
    col1.metric("Failure probability", f"{prediction['failure_probability']:.0%}")
    col2.metric("Risk band", prediction["risk_band"])

    st.markdown(f"**Recommended action:** {decision['recommended_action']}")
    st.markdown(f"**Urgency:** {decision['urgency']}")
    st.markdown(f"**Rationale code:** `{decision['rationale_code']}`")

    if decision["human_review_required"]:
        st.warning("⚠️ Human review required before acting on this recommendation.")
    else:
        st.info("No human review required at this risk level.")

    st.subheader("Explanation")
    if explanation_available():
        with st.spinner("Asking the LLM to explain this decision..."):
            try:
                explanation = generate_explanation(prediction, decision, asset)
                st.write(explanation)
            except Exception as exc:  # noqa: BLE001 - surface any API error to the user
                st.error(f"Could not generate an explanation: {exc}")
    else:
        st.info(
            "No ANTHROPIC_API_KEY configured, so no LLM explanation is shown. "
            "Copy .env.example to .env and add a key to enable this."
        )

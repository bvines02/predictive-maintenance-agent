"""Tests for src/explainer.py.

Mocks the Anthropic client so tests don't need a real API key or network
access - they check that our code builds a request and returns its response
correctly, not that Claude's API itself works.
"""

import pytest

import src.explainer as explainer_module

SAMPLE_PREDICTION = {"failure_probability": 0.97, "risk_band": "High"}
SAMPLE_DECISION = {
    "recommended_action": "Immediate engineering review",
    "urgency": "Immediate",
    "human_review_required": True,
    "rationale_code": "HIGH_RISK_CRITICAL_SPOF_NO_REDUNDANCY",
}
SAMPLE_ASSET = {
    "asset_id": "pump-01",
    "asset_type": "Centrifugal pump",
    "system": "Primary feedwater",
    "criticality": "High",
    "single_point_of_failure": True,
    "redundancy": "None",
    "known_failure_modes": ["Bearing wear", "Seal failure"],
    "operational_workaround": "None",
}


class _FakeTextBlock:
    def __init__(self, text):
        self.text = text


class _FakeMessage:
    def __init__(self, text):
        self.content = [_FakeTextBlock(text)]


class _FakeMessages:
    def __init__(self, response_text):
        self._response_text = response_text
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeMessage(self._response_text)


class _FakeAnthropicClient:
    def __init__(self, response_text="This is a fake explanation."):
        self.messages = _FakeMessages(response_text)


def test_explanation_available_true_when_key_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    assert explainer_module.explanation_available() is True


def test_explanation_available_false_when_key_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert explainer_module.explanation_available() is False


def test_generate_explanation_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    try:
        explainer_module.generate_explanation(SAMPLE_PREDICTION, SAMPLE_DECISION, SAMPLE_ASSET)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "ANTHROPIC_API_KEY" in str(exc)


def test_generate_explanation_returns_model_text(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    fake_client = _FakeAnthropicClient(response_text="Escalate now: no redundancy.")
    monkeypatch.setattr(explainer_module.anthropic, "Anthropic", lambda: fake_client)

    result = explainer_module.generate_explanation(SAMPLE_PREDICTION, SAMPLE_DECISION, SAMPLE_ASSET)

    assert result == "Escalate now: no redundancy."
    # The prompt should carry the decision through untouched, not re-derive it.
    prompt = fake_client.messages.last_kwargs["messages"][0]["content"]
    assert "Immediate engineering review" in prompt
    assert "HIGH_RISK_CRITICAL_SPOF_NO_REDUNDANCY" in prompt
    assert "pump-01" in prompt


SAMPLE_RUL_DECISION = {
    "predicted_rul": 12.34,
    "health_state": "ACTION",
    "base_action": "SCHEDULE_MAINTENANCE",
    "recommended_action": "INTERVENE_NOW",
    "recommended_action_meaning": "Immediate operational / maintenance response is required.",
    "model_confidence": "HIGH",
    "maintenance_lead_time_cycles": 15.0,
    "requires_human_review": True,
    "rule_ids": ["BASE_01", "LEAD_01"],
    "trace": [
        {"rule_id": "BASE_01", "reason": "Health state ACTION maps to SCHEDULE_MAINTENANCE."},
        {"rule_id": "LEAD_01", "reason": "Predicted RUL 12.34 <= maintenance lead time 15."},
    ],
}
SAMPLE_RUL_PREDICTION = {"unit_number": 7, "time_cycles": 180, "prediction_p10": 8.0, "prediction_p90": 19.5}


def _rul_prompt():
    return explainer_module.build_rul_explanation_prompt(
        SAMPLE_RUL_DECISION, SAMPLE_RUL_PREDICTION, {"action": 15, "plan": 30, "watch": 60}, 5
    )


def test_rul_prompt_carries_decision_and_trace_unchanged():
    prompt = _rul_prompt()
    assert "INTERVENE_NOW" in prompt
    assert "LEAD_01: Predicted RUL 12.34" in prompt
    assert "Engine 7, flight cycle 180" in prompt
    assert "8.0 to 19.5 cycles" in prompt
    assert "do not change it" in prompt


def test_rul_prompt_has_no_criticality_or_redundancy_values():
    prompt = _rul_prompt()
    for excluded in ("Criticality:", "Redundancy:", "CRITICAL", "FULL"):
        assert excluded not in prompt


def test_generate_rul_explanation_sends_prompt_verbatim(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    fake_client = _FakeAnthropicClient(response_text="Act now.")
    monkeypatch.setattr(explainer_module.anthropic, "Anthropic", lambda: fake_client)

    prompt = _rul_prompt()
    assert explainer_module.generate_rul_explanation(prompt) == "Act now."
    assert fake_client.messages.last_kwargs["messages"][0]["content"] == prompt


def test_generate_rul_explanation_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        explainer_module.generate_rul_explanation(_rul_prompt())

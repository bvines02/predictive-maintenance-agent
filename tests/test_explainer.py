"""Tests for src/explainer.py.

Mocks the Anthropic client so tests don't need a real API key or network
access - they check that our code builds a request and returns its response
correctly, not that Claude's API itself works.
"""

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

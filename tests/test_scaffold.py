"""Smoke tests: confirm the project structure and imports work.

These will be replaced by real tests as each stage is implemented.
"""

import importlib

import pytest

from src import config

MODULES = [
    "src.config",
    "src.data_loader",
    "src.features",
    "src.train_model",
    "src.evaluate",
    "src.predict",
    "src.decision_rules",
    "src.asset_context",
    "src.explainer",
    "src.agent",
    "src.api",
]
# src.ui is deliberately excluded: it calls Streamlit widget functions
# (st.selectbox, etc.) at import time, which only work inside a real
# Streamlit run (`streamlit run src/ui.py`), not a plain import.


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)


def test_expected_folders_exist():
    for folder in [
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.ASSETS_DIR,
        config.MODELS_DIR,
    ]:
        assert folder.is_dir(), f"Missing folder: {folder}"

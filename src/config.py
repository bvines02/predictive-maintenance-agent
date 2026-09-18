"""Central place for file paths and settings.

Every other module imports paths from here instead of hard-coding them,
so moving a folder means changing one line, not hunting through the code.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ASSETS_DIR = DATA_DIR / "assets"
PROCESS_GRAPH_DIR = DATA_DIR / "process_graph"
TE_PFD_GRAPH_FILE = PROCESS_GRAPH_DIR / "te_pfd_graph.json"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODELS_DIR = PROJECT_ROOT / "models"

# The AI4I 2020 Predictive Maintenance Dataset (see README.md "Dataset" section).
RAW_DATA_FILE = RAW_DATA_DIR / "ai4i2020.csv"

# NASA C-MAPSS turbofan degradation dataset (V2 - RUL prediction).
# FD001 only for now, per the V2 principle of not touching FD002-FD004 yet.
CMAPSS_FD001_DIR = RAW_DATA_DIR / "cmapss" / "FD001"
CMAPSS_FD001_TRAIN_FILE = CMAPSS_FD001_DIR / "train_FD001.txt"
CMAPSS_FD001_TEST_FILE = CMAPSS_FD001_DIR / "test_FD001.txt"
CMAPSS_FD001_RUL_FILE = CMAPSS_FD001_DIR / "RUL_FD001.txt"

# V2 Step 5 - baseline RUL regressor (current-cycle features, no rolling/lag yet).
RUL_BASELINE_MODEL_PATH = MODELS_DIR / "random_forest_baseline.joblib"
RUL_BASELINE_FEATURES_PATH = MODELS_DIR / "random_forest_baseline_features.json"
RUL_BASELINE_ARTIFACTS_DIR = ARTIFACTS_DIR / "rul_baseline"

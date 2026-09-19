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

RESULTS_DIR = PROJECT_ROOT / "results"

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

# V2 Step 7 - operational error analysis outputs.
ERROR_ANALYSIS_ARTIFACTS_DIR = ARTIFACTS_DIR / "error_analysis"
VALIDATION_PREDICTIONS_PATH = RESULTS_DIR / "validation_predictions.csv"
ERROR_BY_RUL_BAND_PATH = RESULTS_DIR / "error_by_rul_band.csv"


# C-MAPSS column metadata, from NASA's C-MAPSS documentation (Saxena et al., 2008).
# The raw files carry no names, so this is the only place the physical meaning
# of each column is written down. Units are imperial.
#
# expected_trend is the direction each sensor moves as an FD001 engine nears
# failure (HPC degradation), checked against the training data via correlation
# with RUL - it is descriptive of this dataset, not a general engineering rule.
# "constant" means the sensor never changes in FD001 (a fixed input).
OPERATIONAL_SETTING_METADATA = {
    "operational_setting_1": {"symbol": "alt", "name": "Altitude", "unit": "ft"},
    "operational_setting_2": {"symbol": "Mach", "name": "Mach number", "unit": "-"},
    "operational_setting_3": {"symbol": "TRA", "name": "Throttle resolver angle", "unit": "deg"},
}

SENSOR_METADATA = {
    "sensor_1": {"symbol": "T2", "name": "Total temperature at fan inlet", "unit": "degR", "expected_trend": "constant"},
    "sensor_2": {"symbol": "T24", "name": "Total temperature at LPC outlet", "unit": "degR", "expected_trend": "rises"},
    "sensor_3": {"symbol": "T30", "name": "Total temperature at HPC outlet", "unit": "degR", "expected_trend": "rises"},
    "sensor_4": {"symbol": "T50", "name": "Total temperature at LPT outlet", "unit": "degR", "expected_trend": "rises"},
    "sensor_5": {"symbol": "P2", "name": "Pressure at fan inlet", "unit": "psia", "expected_trend": "constant"},
    "sensor_6": {"symbol": "P15", "name": "Total pressure in bypass duct", "unit": "psia", "expected_trend": "constant"},
    "sensor_7": {"symbol": "P30", "name": "Total pressure at HPC outlet", "unit": "psia", "expected_trend": "falls"},
    "sensor_8": {"symbol": "Nf", "name": "Physical fan speed", "unit": "rpm", "expected_trend": "rises"},
    "sensor_9": {"symbol": "Nc", "name": "Physical core speed", "unit": "rpm", "expected_trend": "rises"},
    "sensor_10": {"symbol": "epr", "name": "Engine pressure ratio (P50/P2)", "unit": "-", "expected_trend": "constant"},
    "sensor_11": {"symbol": "Ps30", "name": "Static pressure at HPC outlet", "unit": "psia", "expected_trend": "rises"},
    "sensor_12": {"symbol": "phi", "name": "Ratio of fuel flow to Ps30", "unit": "pps/psi", "expected_trend": "falls"},
    "sensor_13": {"symbol": "NRf", "name": "Corrected fan speed", "unit": "rpm", "expected_trend": "rises"},
    "sensor_14": {"symbol": "NRc", "name": "Corrected core speed", "unit": "rpm", "expected_trend": "rises"},
    "sensor_15": {"symbol": "BPR", "name": "Bypass ratio", "unit": "-", "expected_trend": "rises"},
    "sensor_16": {"symbol": "farB", "name": "Burner fuel-air ratio", "unit": "-", "expected_trend": "constant"},
    "sensor_17": {"symbol": "htBleed", "name": "Bleed enthalpy", "unit": "-", "expected_trend": "rises"},
    "sensor_18": {"symbol": "Nf_dmd", "name": "Demanded fan speed", "unit": "rpm", "expected_trend": "constant"},
    "sensor_19": {"symbol": "PCNfR_dmd", "name": "Demanded corrected fan speed", "unit": "%", "expected_trend": "constant"},
    "sensor_20": {"symbol": "W31", "name": "HPT coolant bleed", "unit": "lbm/s", "expected_trend": "falls"},
    "sensor_21": {"symbol": "W32", "name": "LPT coolant bleed", "unit": "lbm/s", "expected_trend": "falls"},
}

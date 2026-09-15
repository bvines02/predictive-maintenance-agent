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

MODELS_DIR = PROJECT_ROOT / "models"

# The AI4I 2020 Predictive Maintenance Dataset (see README.md "Dataset" section).
RAW_DATA_FILE = RAW_DATA_DIR / "ai4i2020.csv"


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
MODELS = OUTPUTS / "models"
SHAP_DIR = OUTPUTS / "shap"

for folder in [DATA_PROCESSED, FIGURES, TABLES, MODELS, SHAP_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_RAW / "NTVZ_LIB_ALL.csv"

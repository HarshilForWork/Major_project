"""Single source of truth for every path in the project.

Everything resolves from the repository root, so scripts run correctly
regardless of the working directory they are invoked from.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # <repo>/src/trace_pd/config.py -> <repo>

DATA          = ROOT / "data"
RAW           = DATA / "raw"
RAW_PPMI      = RAW / "ppmi_csv"
RAW_ZIPS      = RAW / "zips_as_downloaded"
INTERIM       = DATA / "interim"
PROCESSED     = DATA / "processed"

LONG_TABLE    = PROCESSED / "ppmi_tdpigd_long.csv"
DICTIONARY    = PROCESSED / "ppmi_tdpigd_dictionary.csv"
COLUMN_INV    = PROCESSED / "ppmi_tdpigd_column_inventory.csv"

MODELS        = ROOT / "models"
REPORTS       = ROOT / "reports"
FIGURES       = REPORTS / "figures"
METRICS       = REPORTS / "metrics"
DOCS          = ROOT / "docs"
CONFIGS       = ROOT / "configs"

for _p in (INTERIM, PROCESSED, MODELS, FIGURES, METRICS):
    _p.mkdir(parents=True, exist_ok=True)

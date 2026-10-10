"""Project-wide constants and paths."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_TSV = ROOT / "data" / "raw" / "tarc.tsv"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
PREDICTIONS = RESULTS / "predictions"

TARC_URL = "https://raw.githubusercontent.com/eligugliotta/tarc/master/tarc.tsv"

SEED = 42
LABELS = ["TUN", "FOR"]  # TUN = Tunisian Arabizi, FOR = foreign (French / English)
COLORS = {"TUN": "#2a78d6", "FOR": "#eb6834"}

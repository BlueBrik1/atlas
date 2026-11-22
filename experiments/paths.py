from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "gaia_bright_stars.csv"
OUTPUT = ROOT / "runs"
OUTPUT.mkdir(exist_ok=True)

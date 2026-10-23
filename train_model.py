from pathlib import Path
from src.modeling import train_model

ROOT = Path(__file__).resolve().parent
csv_path = ROOT / "data" / "features.csv"
if not csv_path.exists():
    raise SystemExit("Run `python prepare_dataset.py` first.")
train_model(str(csv_path))
print("Saved trained model to models/quality_model.joblib")
print("Important: evaluate on held-out recordings before relying on its predictions.")

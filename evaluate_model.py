"""
Evaluate on a held-out CSV of precomputed features.
Expected file: data/test_features.csv with the same columns as data/features.csv.
The test recordings must not have been used to fit/tune the model.
"""
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score
from src.modeling import load_model_bundle, LABELS

ROOT = Path(__file__).resolve().parent
test_path = ROOT / "data" / "test_features.csv"
bundle = load_model_bundle()
if bundle is None:
    raise SystemExit("No trained model found. Run `python train_model.py` first.")
if not test_path.exists():
    raise SystemExit("Create data/test_features.csv from held-out recordings.")
df = pd.read_csv(test_path)
y_true = df["label"].astype(str)
X = df[bundle["feature_columns"]]
y_pred = bundle["pipeline"].predict(X)
print("Balanced accuracy:", round(balanced_accuracy_score(y_true, y_pred), 4))
print("Macro F1:", round(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0), 4))
print("\nClassification report:\n", classification_report(y_true, y_pred, labels=LABELS, zero_division=0))
print("Confusion matrix (rows=true, columns=predicted; order Poor, Average, Good, Excellent):")
print(confusion_matrix(y_true, y_pred, labels=LABELS))

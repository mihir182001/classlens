from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "quality_model.joblib"
LABELS = ["Poor", "Average", "Good", "Excellent"]
FEATURE_COLUMNS = [
    "duration_seconds", "frame_count", "visual_change", "mean_frame_brightness",
    "speech_ratio", "mean_rms", "rms_variability", "silence_ratio", "clip_ratio",
    "word_count", "words_per_minute", "question_count", "exclamation_count", "segment_count",
]


def train_model(features_csv: str, output_path: str = str(MODEL_PATH)):
    df = pd.read_csv(features_csv)
    required = {"label", *FEATURE_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in features CSV: {sorted(missing)}")
    df = df[df["label"].isin(LABELS)].copy()
    if df["label"].nunique() < 2:
        raise ValueError("Training requires at least two different quality labels.")
    if df["label"].value_counts().min() < 2:
        raise ValueError("Need at least two recordings per class to fit this basic classifier.")
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    pipeline.fit(df[FEATURE_COLUMNS], df["label"])
    bundle = {"pipeline": pipeline, "feature_columns": FEATURE_COLUMNS, "labels": LABELS}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    return bundle


def load_model_bundle():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None


def predict_quality(features: dict, bundle=None):
    bundle = bundle if bundle is not None else load_model_bundle()
    if bundle is None:
        score = float(features.get("evidence_score", 50))
        label = "Poor" if score < 40 else "Average" if score < 60 else "Good" if score < 80 else "Excellent"
        return {"label": label, "confidence": None, "mode": "heuristic"}
    cols = bundle["feature_columns"]
    X = pd.DataFrame([{c: features.get(c, np.nan) for c in cols}])
    pipeline = bundle["pipeline"]
    label = str(pipeline.predict(X)[0])
    confidence = None
    if hasattr(pipeline, "predict_proba"):
        confidence = float(np.max(pipeline.predict_proba(X)[0]))
    return {"label": label, "confidence": confidence, "mode": "trained"}

"""
Extract features from each labelled video to create data/features.csv.

1. Put recordings under data/videos/
2. Fill data/labels.csv with columns: filename,label
   Example: lesson_01.mp4,Good
3. Run: python prepare_dataset.py
"""
import os
from pathlib import Path
import pandas as pd
from src.analysis import analyze_video

ROOT = Path(__file__).resolve().parent
labels_path = ROOT / "data" / "labels.csv"
videos_dir = ROOT / "data" / "videos"
output_path = ROOT / "data" / "features.csv"

def main():
    if not labels_path.exists():
        raise SystemExit("Create data/labels.csv first. See data/labels_template.csv.")
    labels = pd.read_csv(labels_path)
    if not {"filename", "label"}.issubset(labels.columns):
        raise SystemExit("labels.csv must have columns: filename,label")
    rows = []
    for _, row in labels.iterrows():
        path = videos_dir / str(row["filename"])
        if not path.exists():
            print(f"SKIP missing video: {path}")
            continue
        print(f"Analysing {path.name} ({row['label']})")
        try:
            result = analyze_video(str(path))
            rows.append({**result["features"], "filename": path.name, "label": row["label"]})
        except Exception as exc:
            print(f"FAILED {path.name}: {exc}")
    if not rows:
        raise SystemExit("No recordings were processed successfully.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote {len(rows)} rows to {output_path}")

if __name__ == "__main__":
    main()

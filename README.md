# ClassLens — 8x8 Class Quality Assessment

A local Streamlit prototype that analyses an online-class video and produces a transparent, evidence-linked report. It extracts audio with FFmpeg, transcribes speech with Faster-Whisper, calculates basic acoustic/video features, and optionally uses a scikit-learn classifier trained on labelled sample recordings.

## Important limitations

- The initial rating is a heuristic baseline, **not a validated measure of teaching quality**.
- The classifier only learns from the features included in this project. It does not yet understand the educational correctness of explanations or whether slides are pedagogically useful.
- Pixel variation is not a measure of visual quality. Low-energy audio is not necessarily silence or poor teaching.
- ASR errors, accent, language, background noise and recording quality can affect results.
- With a very small labelled dataset, model performance estimates are highly uncertain. Never report training-set performance as generalisation performance.
- Do not upload children's recordings to external APIs without appropriate permission and privacy review. The default project does not send video/audio to an LLM API.

## 1. Prerequisites

- Python 3.10 or 3.11 recommended.
- FFmpeg installed and available on your PATH.
  - Windows: install FFmpeg (for example via `winget install Gyan.FFmpeg`) and reopen the terminal.
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
- VS Code with the Python extension.
- Several GB of disk space may be needed for speech-model downloads.

Check FFmpeg:

```bash
ffmpeg -version
```

## 2. Create an environment

Open this folder in VS Code, then run in its terminal.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### macOS/Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Run the application

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`). Upload an MP4/MOV/MKV/AVI/WEBM file and choose **Analyse class**. The first run downloads the Faster-Whisper model.

For faster initial tests, select `tiny`; use `base` or `small` for better transcription quality if your computer can handle it.


## Download the provided Google Drive dataset

The project code and the class recordings are separate. This project includes
`download_drive_dataset.py`, preconfigured with the shared folder URL provided
in the conversation.

After installing requirements, run:

```bash
python download_drive_dataset.py
```

This works only if the folder/files are accessible for download without signing
in. If it reports that no files could be downloaded, open the Drive folder in
your browser and check whether you can view and download the videos. If it is
restricted to invited accounts, use an authenticated Google Drive API/OAuth setup
or download the recordings manually into `data/videos/`. Do not share passwords
or session cookies.

After downloading, inspect the actual folder structure and filenames. Update
`data/labels.csv` with the real filename-to-label mapping. If the organiser
provided the labels in a separate file or in subfolder names, map those labels
carefully; do not infer quality labels from filenames unless the organiser
explicitly defined that convention.

## 4. Add the labelled sample recordings

Create this structure:

```text
data/
  labels.csv
  videos/
    lesson_01.mp4
    lesson_02.mp4
```

Copy your actual recordings into `data/videos/`. Create `data/labels.csv` with exactly these columns:

```csv
filename,label
lesson_01.mp4,Poor
lesson_02.mp4,Average
lesson_03.mp4,Good
lesson_04.mp4,Excellent
```

Use the exact filenames and one of these labels: `Poor`, `Average`, `Good`, `Excellent`. The sample template is illustrative; replace it with the real filenames and labels.

## 5. Extract features and train the baseline classifier

```bash
python prepare_dataset.py
python train_model.py
```

`prepare_dataset.py` runs transcription and feature extraction for every labelled video. This can take time. It writes `data/features.csv`. `train_model.py` fits a balanced logistic regression model and saves `models/quality_model.joblib`.

Restart or rerun the Streamlit analysis after training. The app uses the trained classifier when the saved model exists; otherwise it clearly identifies the fallback as a heuristic.

**Data-size warning:** Logistic regression can technically fit with a small dataset, but that does not mean the resulting model is reliable. If there are only a few recordings per class, present it as a prototype and emphasise the need for more labelled examples.

## 6. Evaluation

Use recordings held out from all feature/model development as the test set. Do not use the same recordings for training and evaluation. For a tiny dataset, consider leave-one-out or stratified cross-validation for exploratory analysis, while clearly reporting uncertainty.

Create `data/test_features.csv` containing extracted features for held-out videos. Easiest workflow:
1. Use `prepare_dataset.py` to process all recordings into `data/features.csv`.
2. Before training, separate whole recordings into train and test sets, ensuring no test recording appears in training.
3. Save held-out rows to `data/test_features.csv` and training rows to `data/features.csv`.
4. Run `python train_model.py`, then `python evaluate_model.py`.

Report:
- Macro-F1 (treats all four classes equally).
- Balanced accuracy.
- Per-class precision, recall and F1.
- Confusion matrix.
- Baseline comparison (e.g., majority-class prediction).
- A short review of incorrect predictions and whether the generated explanations are supported by timestamps/transcript evidence.

If several recordings are from the same teacher or class session, split by teacher/session where possible to reduce leakage. Never randomly split overlapping segments from the same recording across train and test.

## 7. Recommended improvements

1. Add human-reviewed transcript criteria: clarity, relevance, examples, coherent sequencing and checks for understanding.
2. Add a vision-language model to describe sampled frames and whether visible material appears related to the transcript. Validate these observations manually.
3. Add diarization if reliable teacher/student turn separation is important; basic Whisper transcription does not identify speaker identities.
4. Use a held-out human-rated set to tune thresholds and calibrate confidence.
5. Add evidence timestamps to every substantive explanation and allow “Insufficient evidence” when the recording is unusable.
6. Review privacy, consent and data-retention requirements because the recordings involve children.

## Project structure

```text
8x8_class_quality/
├── app.py
├── prepare_dataset.py
├── train_model.py
├── evaluate_model.py
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── analysis.py
│   ├── transcription.py
│   └── modeling.py
├── data/
│   ├── labels_template.csv
│   └── videos/
└── models/
```

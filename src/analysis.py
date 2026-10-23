
import os
import subprocess
import tempfile

import cv2
import numpy as np

from src.transcription import transcribe_audio


def _extract_audio(video_path: str, wav_path: str):
    cmd = [
        "ffmpeg", "-y", "-i", video_path, "-vn",
        "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
        wav_path,
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)

    if completed.returncode != 0:
        raise RuntimeError(
            "FFmpeg could not extract audio. Install FFmpeg and ensure "
            "`ffmpeg` is on PATH. " + completed.stderr[-1200:]
        )


def _video_features(video_path: str, frame_interval: int = 10):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Could not open this video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = total_frames / fps if fps > 0 else 0.0
    interval = max(1, int((fps if fps > 0 else 1) * frame_interval))

    signatures = []
    brightness = []
    index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if index % interval == 0:
            small = cv2.resize(frame, (96, 54))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            signatures.append(gray.astype(np.float32) / 255.0)
            brightness.append(float(gray.mean()) / 255.0)

        index += 1

    cap.release()

    if len(signatures) > 1:
        differences = [
            float(np.mean(np.abs(signatures[i] - signatures[i - 1])))
            for i in range(1, len(signatures))
        ]
        visual_change = float(np.mean(differences))
    else:
        visual_change = 0.0

    return {
        "duration_seconds": float(duration),
        "frame_count": len(signatures),
        "visual_change": visual_change,
        "mean_frame_brightness": (
            float(np.mean(brightness)) if brightness else 0.0
        ),
    }


def _audio_features(wav_path: str):
    try:
        import librosa
        y, _ = librosa.load(wav_path, sr=16000, mono=True)
    except Exception as exc:
        raise RuntimeError(f"Audio feature extraction failed: {exc}") from exc

    if y.size == 0:
        return {
            "speech_ratio": 0.0,
            "mean_rms": 0.0,
            "rms_variability": 0.0,
            "silence_ratio": 1.0,
            "clip_ratio": 0.0,
        }

    import librosa

    rms = librosa.feature.rms(
        y=y, frame_length=16000, hop_length=8000, center=True
    )[0]

    threshold = max(float(np.percentile(rms, 70)) * 0.12, 0.003)
    active = rms > threshold
    active_ratio = float(np.mean(active)) if active.size else 0.0

    return {
        # This is energy activity, not a true speech detector.
        "speech_ratio": active_ratio,
        "mean_rms": float(np.mean(rms)) if rms.size else 0.0,
        "rms_variability": float(np.std(rms)) if rms.size else 0.0,
        "silence_ratio": 1.0 - active_ratio,
        "clip_ratio": float(np.mean(np.abs(y) >= 0.99)),
    }


def _build_observations(features, transcript, segments):
    observations = []

    if transcript.strip():
        observations.append({
            "title": "Speech transcript available",
            "detail": (
                f"The speech recogniser produced {features['word_count']} "
                f"words across {features['segment_count']} segments. "
                "Review the transcript to judge explanation clarity, "
                "accuracy and relevance. Word count alone does not establish "
                "teaching quality."
            ),
            "timestamp": segments[0]["start"] if segments else None,
        })

        if features["question_count"]:
            observations.append({
                "title": "Possible questions detected",
                "detail": (
                    f"The transcript contains {features['question_count']} "
                    "question mark(s). Check the recording to see whether "
                    "these are meaningful checks for understanding. "
                    "Automatic punctuation can be inaccurate."
                ),
                "timestamp": None,
            })
        else:
            observations.append({
                "title": "No questions detected in transcript",
                "detail": (
                    "No question marks were recognised. This does not mean "
                    "that the lesson lacks interaction; transcription and "
                    "punctuation may be imperfect."
                ),
                "timestamp": None,
            })
    else:
        observations.append({
            "title": "No usable transcript",
            "detail": (
                "No speech text was produced. Check the audio track, "
                "spoken language and background noise."
            ),
            "timestamp": None,
        })

    observations.append({
        "title": "Audio activity",
        "detail": (
            f"Approximately {features['silence_ratio']:.0%} of analysed "
            "audio frames were classified as low-energy. This is a signal "
            "estimate, not a true speech detector. Pauses may be intentional."
        ),
        "timestamp": None,
    })

    observations.append({
        "title": "Visual variation",
        "detail": (
            f"Mean pixel difference between sampled frames: "
            f"{features['visual_change']:.3f}. This measures visual change, "
            "not slide quality or teaching effectiveness."
        ),
        "timestamp": None,
    })

    if features["clip_ratio"] > 0.01:
        observations.append({
            "title": "Possible audio clipping",
            "detail": (
                f"{features['clip_ratio']:.1%} of audio samples are near "
                "the maximum digital level. Listen to the recording to "
                "check for distortion."
            ),
            "timestamp": None,
        })

    return observations


def analyze_video(
    video_path: str,
    model_size: str = "base",
    frame_interval: int = 10,
):
    with tempfile.TemporaryDirectory() as workdir:
        wav_path = os.path.join(workdir, "audio.wav")
        _extract_audio(video_path, wav_path)
        audio = _audio_features(wav_path)
        transcript, segments = transcribe_audio(
            wav_path, model_size=model_size
        )

    video = _video_features(
        video_path, frame_interval=frame_interval
    )

    words = transcript.split()
    duration_minutes = max(video["duration_seconds"] / 60.0, 1e-6)

    features = {
        **video,
        **audio,
        "word_count": len(words),
        "words_per_minute": len(words) / duration_minutes,
        "question_count": transcript.count("?"),
        "exclamation_count": transcript.count("!"),
        "segment_count": len(segments),
    }

    observations = _build_observations(features, transcript, segments)

    return {
        "features": features,
        "transcript": transcript,
        "segments": segments,
        "observations": observations,
        # Deliberately no invented quality score or trained prediction.
        "evidence_score": None,
    }

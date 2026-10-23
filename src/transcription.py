from functools import lru_cache


@lru_cache(maxsize=3)
def _get_model(model_size: str):
    from faster_whisper import WhisperModel
    # CPU int8 is the most portable default; use CUDA separately if configured.
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe_audio(audio_path: str, model_size: str = "base"):
    model = _get_model(model_size)
    segments_iter, info = model.transcribe(
        audio_path, beam_size=3, vad_filter=True, word_timestamps=False
    )
    segments = []
    text_parts = []
    for segment in segments_iter:
        text = segment.text.strip()
        text_parts.append(text)
        segments.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "text": text,
        })
    return " ".join(text_parts).strip(), segments

"""Speaker diarization backed by NVIDIA's Nemotron-3-Diarization model.

Replaces the previous pyannote.audio-based whisperx.DiarizationPipeline step.
`diarize()` returns a pandas DataFrame shaped exactly like
whisperx.DiarizationPipeline's output (columns: start, end, speaker), so it
plugs directly into whisperx.assign_word_speakers without any change to the
existing word/sentence assignment logic or the output schema clients depend on.

It additionally returns per-segment confidence scores (derived from the raw
frame-level speaker-activity probabilities), used to decide which speaker
turns are ambiguous enough to send to a SpeakerConfirmationProvider.
"""
import pandas as pd
import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor

MODEL_ID = "nvidia/Nemotron-3-Diarization"

_processor = None
_model = None


def _load():
    global _processor, _model
    if _model is None:
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = AutoModelForAudioFrameClassification.from_pretrained(
            MODEL_ID, device_map="auto"
        )
        _model.eval()
    return _processor, _model


def unload():
    """Frees the model from GPU memory, mirroring the teardown pattern already
    used for the whisper/align/diarize models in predict.py."""
    global _processor, _model
    _processor = None
    _model = None
    torch.cuda.empty_cache()


def diarize(audio, max_num_speakers: int = 8):
    """Runs Nemotron-3-Diarization on an already-loaded 16kHz mono waveform
    (a float32 numpy array, e.g. from whisperx.load_audio -- the same array
    already used for transcription elsewhere in the pipeline).

    Returns:
        diarize_df: pandas.DataFrame with columns [start, end, speaker],
            shaped like whisperx.DiarizationPipeline's output.
        segment_confidences: list of dicts {start, end, speaker, confidence},
            confidence being the mean margin between the winning speaker's
            frame-level probability and the runner-up's (0..1, higher = more
            certain). Used to gate confirmation-provider calls.
    """
    processor, model = _load()
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits  # [1, num_frames, num_speaker_channels]

    probs = torch.sigmoid(logits)[0].cpu()  # [num_frames, num_speaker_channels]
    speaker_segments = processor.extract_speaker_dict(logits)[0]  # [(speaker_idx, start, end), ...]

    frame_seconds = getattr(processor, "frame_duration_seconds", 0.01)
    num_channels = probs.shape[-1]
    k = min(2, num_channels)
    top_values, _ = probs.topk(k, dim=-1)
    if k == 2:
        margin = top_values[:, 0] - top_values[:, 1]
    else:
        margin = top_values[:, 0]

    rows = []
    segment_confidences = []
    for speaker_idx, start, end in speaker_segments:
        speaker_label = f"SPEAKER_{int(speaker_idx):02d}"
        rows.append({"start": float(start), "end": float(end), "speaker": speaker_label})

        frame_start = max(0, int(start / frame_seconds))
        frame_end = max(frame_start + 1, int(end / frame_seconds))
        segment_margin = margin[frame_start:frame_end]
        confidence = float(segment_margin.mean()) if len(segment_margin) else 0.0
        segment_confidences.append(
            {
                "start": float(start),
                "end": float(end),
                "speaker": speaker_label,
                "confidence": confidence,
            }
        )

    diarize_df = pd.DataFrame(rows, columns=["start", "end", "speaker"])
    return diarize_df, segment_confidences


def confidence_for_range(segment_confidences, start: float, end: float) -> float:
    """Looks up the diarization confidence for a transcript segment's time
    range, matching against whichever diarization segment overlaps it most."""
    best_overlap = 0.0
    best_confidence = 1.0
    for seg in segment_confidences:
        overlap = min(seg["end"], end) - max(seg["start"], start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_confidence = seg["confidence"]
    return best_confidence


def distinct_speakers_in_range(segment_confidences, start: float, end: float) -> set:
    """Raw Nemotron speaker labels with any diarization activity overlapping
    (start, end) -- e.g. a silence gap in the transcript. Used to tell a
    clean two-party turn from a window where a third voice was picked up,
    even briefly, before trusting a two-way speaker-identification guess."""
    return {
        seg["speaker"]
        for seg in segment_confidences
        if min(seg["end"], end) > max(seg["start"], start)
    }

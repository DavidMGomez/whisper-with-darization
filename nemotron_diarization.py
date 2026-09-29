"""Speaker diarization backed by NVIDIA's Nemotron-3-Diarization model.

Replaces the previous pyannote.audio-based whisperx.DiarizationPipeline step.
`diarize()` returns a pandas DataFrame shaped exactly like
whisperx.DiarizationPipeline's output (columns: start, end, speaker), so it
plugs directly into whisperx.assign_word_speakers without any change to the
existing word/sentence assignment logic or the output schema clients depend on.

It returns speaker-specific activity probabilities for assignment gating.
These are model scores, not calibrated probabilities of a real-world identity.
"""
import math
import pandas as pd
import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor

# Baked into the image at build time (see .github/workflows/main.yml's
# "Download Nemotron model" step) rather than loaded from the HF Hub id at
# request time: Replicate's runtime network path to huggingface.co goes
# through an internal proxy that isn't fully reliable (hit both an outright
# SSRF-policy refusal for NLTK data and, separately, a flaky
# "peer closed connection" mid-download for this model's safetensors file).
# Baking removes that dependency entirely instead of hoping retries succeed.
MODEL_PATH = "./models/nemotron-3-diarization"

_processor = None
_model = None


def _load():
    global _processor, _model
    if _model is None:
        _processor = AutoProcessor.from_pretrained(MODEL_PATH)
        _model = AutoModelForAudioFrameClassification.from_pretrained(
            MODEL_PATH, device_map="auto"
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
            confidence is the assigned channel's mean activity probability.
    """
    processor, model = _load()
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits  # [1, num_frames, num_speaker_channels]

    probs = torch.sigmoid(logits)[0].cpu()  # [num_frames, num_speaker_channels]
    # extract_speaker_dict returns, per batch sample, a list of dicts like
    # {"Start": 0.0, "End": 15.43, "Speaker": 0} -- not tuples. Confirmed
    # against transformers' own processing_nemotron3_diarization.py /
    # test_processing_nemotron3_diarization.py, since the model card's
    # abbreviated usage snippet doesn't show this method at all.
    speaker_segments = processor.extract_speaker_dict(logits, inputs.get("attention_mask"))[0]

    frame_seconds = (
        processor.feature_extractor.hop_length / processor.feature_extractor.sampling_rate
    )
    rows = []
    segment_confidences = []
    for seg in speaker_segments:
        speaker_idx, start, end = seg["Speaker"], seg["Start"], seg["End"]
        speaker_label = f"SPEAKER_{int(speaker_idx):02d}"
        rows.append({"start": float(start), "end": float(end), "speaker": speaker_label})

        frame_start = max(0, int(start / frame_seconds))
        frame_end = max(frame_start + 1, int(end / frame_seconds))
        activity = probs[frame_start:frame_end, int(speaker_idx)]
        confidence = float(activity.mean()) if len(activity) else 0.0
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
    best_confidence = 0.0
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


def assignment_confidence(segment_confidences, start, end, speaker):
    """Weighted activity for this speaker; uncovered time contributes zero.

    Simultaneous speakers remain ambiguous for exclusive transcript attribution.
    Never borrow confidence from whichever other voice happened to be loudest.
    """
    if not speaker or not math.isfinite(start) or not math.isfinite(end) or end <= start:
        return 0.0
    weighted, covered = 0.0, 0.0
    for row in segment_confidences:
        overlap = max(0.0, min(end, row['end']) - max(start, row['start']))
        if not overlap:
            continue
        if row['speaker'] != speaker:
            return 0.0
        score = row.get('confidence', 0)
        if not math.isfinite(score) or not 0 <= score <= 1:
            continue
        weighted += overlap * score
        covered += overlap
    return min(1.0, weighted / max(end - start, covered)) if covered else 0.0

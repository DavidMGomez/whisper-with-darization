import torch

import nemotron_diarization


class _FakeProcessor:
    frame_duration_seconds = 0.01

    def __call__(self, audio, sampling_rate, return_tensors):
        return {"input_values": torch.zeros(1, 1600)}

    def extract_speaker_dict(self, logits):
        # Matches _fake_logits(): SPEAKER_00 for the first half-second,
        # SPEAKER_01 for the second.
        return [[(0, 0.0, 0.5), (1, 0.5, 1.0)]]


class _FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class _FakeModel:
    device = "cpu"

    def __call__(self, **_kwargs):
        return _FakeOutput(_fake_logits())


def _fake_logits():
    # 100 frames of 10ms each (1s total), 2 speaker channels.
    logits = torch.zeros(1, 100, 2)
    logits[0, :50, 0] = 5.0    # sigmoid(5) ~= 0.993 -> speaker 0 confidently active
    logits[0, :50, 1] = -5.0   # sigmoid(-5) ~= 0.007 -> large margin, high confidence
    logits[0, 50:, 0] = 0.05
    logits[0, 50:, 1] = 0.0    # near-tied probabilities -> small margin, low confidence
    return logits


def test_diarize_returns_whisperx_shaped_dataframe(monkeypatch):
    monkeypatch.setattr(nemotron_diarization, "_load", lambda: (_FakeProcessor(), _FakeModel()))

    diarize_df, segment_confidences = nemotron_diarization.diarize(torch.zeros(16000), max_num_speakers=2)

    assert list(diarize_df.columns) == ["start", "end", "speaker"]
    assert diarize_df.iloc[0].to_dict() == {"start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"}
    assert diarize_df.iloc[1].to_dict() == {"start": 0.5, "end": 1.0, "speaker": "SPEAKER_01"}

    assert len(segment_confidences) == 2
    high_confidence_segment, low_confidence_segment = segment_confidences
    assert high_confidence_segment["speaker"] == "SPEAKER_00"
    assert low_confidence_segment["speaker"] == "SPEAKER_01"
    assert high_confidence_segment["confidence"] > 0.9
    assert low_confidence_segment["confidence"] < 0.1
    assert high_confidence_segment["confidence"] > low_confidence_segment["confidence"]


def test_confidence_for_range_picks_the_most_overlapping_segment():
    segment_confidences = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "confidence": 0.9},
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01", "confidence": 0.2},
    ]

    assert nemotron_diarization.confidence_for_range(segment_confidences, 0.1, 0.9) == 0.9
    assert nemotron_diarization.confidence_for_range(segment_confidences, 1.1, 1.9) == 0.2


def test_confidence_for_range_defaults_to_full_confidence_when_nothing_overlaps():
    assert nemotron_diarization.confidence_for_range([], 0.0, 1.0) == 1.0


def test_distinct_speakers_in_range_only_counts_actual_overlap():
    segment_confidences = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "confidence": 0.9},
        {"start": 5.0, "end": 5.4, "speaker": "SPEAKER_05", "confidence": 0.8},
        {"start": 20.0, "end": 21.0, "speaker": "SPEAKER_01", "confidence": 0.9},
    ]

    # (1.0, 10.0) touches the first segment only at its boundary (no overlap)
    # and fully contains the second -> only SPEAKER_05 counts.
    assert nemotron_diarization.distinct_speakers_in_range(segment_confidences, 1.0, 10.0) == {"SPEAKER_05"}
    assert nemotron_diarization.distinct_speakers_in_range(segment_confidences, 100.0, 101.0) == set()

"""Nemotron labels stay intact; voice embeddings and turn reassignment are retired."""
import numpy as np
import pandas as pd
import pytest

import predict

LEGACY_SEGMENT_KEYS = {"start", "end", "text", "words", "speaker", "speaker_confidence", "speaker_confidence_source"}


def _fake_transcript_result():
    return {
        "language": "es",
        "segments": [
            {
                "start": 0.0, "end": 1.0, "text": "hola mundo",
                "words": [
                    {"word": "hola", "start": 0.0, "end": 0.4, "score": 0.9},
                    {"word": "mundo", "start": 0.5, "end": 1.0, "score": 0.9},
                ],
            },
            {
                "start": 1.2, "end": 2.0, "text": "como estas",
                "words": [
                    {"word": "como", "start": 1.2, "end": 1.6, "score": 0.9},
                    {"word": "estas", "start": 1.6, "end": 2.0, "score": 0.9},
                ],
            },
        ],
    }


def _assign_fake_speakers(_diarize_df, result):
    for segment in result["segments"]:
        segment["speaker"] = "SPEAKER_00"
        for word in segment["words"]:
            word["speaker"] = "SPEAKER_00"
    return result


def _base_kwargs(**overrides):
    kwargs = dict(
        file_url="https://example.com/audio.mp4",
        language="es",
        batch_size=8,
        multimedia_part_id=None,
        project_id=None,
        topic_id=None,
        credentials=None,
        hf_token=None,
        min_num_speakers=None,
        max_num_speakers=None,
        whisper_model="large-v3-turbo",
        use_speaker_confirmation=False,
        confirmation_provider="jev",
        jev_api_key=None,
        confirmation_confidence_threshold=0.67,
        confirmation_gap_threshold_seconds=1.5,
        confirmation_context_window=3,
        confirmation_batch_size=8,
        classify_speaker_roles=False,
        speaker_role_window_seconds=1800,
        hearing_roles=None,
        openai_api_key=None,
        openai_model="gpt-4o-mini",
    )
    kwargs.update(overrides)
    return kwargs


@pytest.fixture
def predictor(monkeypatch):
    p = predict.Predictor()

    # Skip real network I/O -- not what this test is about.
    monkeypatch.setattr(p, "download_audio_and_convert_to_wav", lambda file_url, temp_wav_filename: temp_wav_filename)

    fake_model = type("FakeWhisperModel", (), {
        "transcribe": staticmethod(lambda audio, batch_size: _fake_transcript_result())
    })()
    monkeypatch.setattr(predict.whisperx, "load_model", lambda *a, **k: fake_model)
    monkeypatch.setattr(predict.whisperx, "load_audio", lambda path: np.zeros(16000, dtype=np.float32))
    monkeypatch.setattr(predict.whisperx, "load_align_model", lambda language_code, device: (object(), object()))
    monkeypatch.setattr(
        predict.whisperx, "align",
        lambda segments, model_a, metadata, audio, device, return_char_alignments: {
            "language": "es", "segments": segments
        }
    )
    monkeypatch.setattr(predict.whisperx, "assign_word_speakers", _assign_fake_speakers)

    monkeypatch.setattr(
        predict.nemotron_diarization, "diarize",
        lambda audio, max_num_speakers: (pd.DataFrame(columns=["start", "end", "speaker"]), [])
    )
    monkeypatch.setattr(predict.nemotron_diarization, "unload", lambda: None)

    # torch.cuda.empty_cache() is safe on a real GPU (production target) but
    # this suite deliberately runs on CPU-only torch to stay CI-friendly.
    monkeypatch.setattr(predict.torch.cuda, "empty_cache", lambda: None)

    return p


def test_output_schema_unchanged_when_confirmation_disabled(predictor):
    output = predictor.predict(**_base_kwargs())

    assert len(output.segments) == 2
    for segment in output.segments:
        assert LEGACY_SEGMENT_KEYS.issubset(segment.keys())
        assert set(segment.keys()) == LEGACY_SEGMENT_KEYS  # no new keys leak in by default
        assert segment["speaker"] == "SPEAKER_00"
        assert "speaker_embedding" not in segment


def test_legacy_confirmation_flag_cannot_reassign_nemotron_labels(monkeypatch, predictor):
    monkeypatch.setattr(predict.confirmation, "annotate_and_confirm", lambda *a, **k: pytest.fail("Turn reassignment must not run"))
    class _StubProvider(predict.confirmation.SpeakerConfirmationProvider):
        def confirm_speaker(self, text, candidate_speakers, context_before=None, context_after=None):
            return predict.confirmation.ConfirmationResult(speaker="SPEAKER_00", confidence=0.99, raw={})

        def confirm_continuation(self, established_context, established_speaker, candidate_texts):
            return [
                predict.confirmation.ContinuationResult(same_speaker=True, confidence=0.99, raw={})
                for _ in candidate_texts
            ]

    monkeypatch.setattr(predict.confirmation, "get_confirmation_provider", lambda name, api_key: _StubProvider())
    monkeypatch.setattr(predict.nemotron_diarization, "confidence_for_range", lambda *a, **k: 0.1)  # force it on

    output = predictor.predict(**_base_kwargs(use_speaker_confirmation=True))

    for segment in output.segments:
        assert LEGACY_SEGMENT_KEYS.issubset(segment.keys())
        assert segment["speaker"] == "SPEAKER_00"
        assert "jev_checked" not in segment
        assert "speaker_embedding" not in segment


def test_legacy_identity_flag_does_not_call_identity_providers(monkeypatch, predictor):
    monkeypatch.setattr(predict.confirmation, "get_confirmation_provider",
                        lambda *a, **k: pytest.fail("Identity must run in the CPU function"))
    output = predictor.predict(**_base_kwargs(use_speaker_confirmation=True, classify_speaker_roles=True))
    for segment in output.segments:
        assert LEGACY_SEGMENT_KEYS.issubset(segment.keys())
        assert "speaker_identity" not in segment
        assert "speaker_role" not in segment
        assert "speaker_name" not in segment

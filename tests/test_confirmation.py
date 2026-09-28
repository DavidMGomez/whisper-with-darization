import pytest

import confirmation
import nemotron_diarization


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_jev_provider_builds_expected_payload_and_parses_response(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured.update(url=url, headers=headers, json=json, timeout=timeout)
        return _FakeResponse({"answers": {"speaker": {"choice": "SPEAKER_01", "confidence": 0.87}}})

    monkeypatch.setattr(confirmation.requests, "post", fake_post)

    provider = confirmation.JevConfirmationProvider(api_key="secret-key")
    result = provider.confirm_speaker(
        text="y luego que paso",
        candidate_speakers=["SPEAKER_00", "SPEAKER_01"],
        context_before="cuentame la historia",
        context_after="no lo puedo creer",
    )

    assert result.speaker == "SPEAKER_01"
    assert result.confidence == 0.87
    assert captured["url"] == "https://api.typesafe.ai/v1/systemone"
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert captured["json"]["model"] == "jev-latest"
    assert captured["json"]["state"]["utterance"] == "y luego que paso"
    assert captured["json"]["state"]["previous_utterance"] == "cuentame la historia"
    assert captured["json"]["questions"]["speaker"]["type"] == "choice"
    assert set(captured["json"]["questions"]["speaker"]["criteria"]) == {"SPEAKER_00", "SPEAKER_01"}


def test_jev_provider_requires_api_key():
    with pytest.raises(ValueError):
        confirmation.JevConfirmationProvider(api_key=None)


def test_get_confirmation_provider_rejects_unknown_name():
    with pytest.raises(ValueError):
        confirmation.get_confirmation_provider("not-a-real-provider", api_key="x")


def test_get_confirmation_provider_falls_back_to_env_var(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "from-env")
    provider = confirmation.get_confirmation_provider("jev", api_key=None)
    assert provider.api_key == "from-env"


def _segment(start, end, text, speaker="SPEAKER_00"):
    return {"start": start, "end": end, "text": text, "speaker": speaker, "words": []}


def test_annotate_and_confirm_triggers_only_on_low_confidence_or_big_gap(monkeypatch):
    fixed_confidences = {
        (0.0, 1.0): 0.9,    # A: high confidence, first segment -> no gap check
        (1.1, 2.0): 0.9,    # B: high confidence, small gap after A -> should NOT trigger
        (2.1, 3.0): 0.2,    # C: low confidence, small gap -> SHOULD trigger (confidence)
        (10.0, 11.0): 0.9,  # D: high confidence, big gap after C -> SHOULD trigger (gap)
    }
    monkeypatch.setattr(
        nemotron_diarization,
        "confidence_for_range",
        lambda segment_confidences, start, end: fixed_confidences[(start, end)],
    )

    segments = [
        _segment(0.0, 1.0, "a"),
        _segment(1.1, 2.0, "b"),
        _segment(2.1, 3.0, "c"),
        _segment(10.0, 11.0, "d"),
    ]

    class _RecordingProvider(confirmation.SpeakerConfirmationProvider):
        def __init__(self):
            self.confirmed_texts = []

        def confirm_speaker(self, text, candidate_speakers, context_before=None, context_after=None):
            self.confirmed_texts.append(text)
            return confirmation.ConfirmationResult(speaker="SPEAKER_00", confidence=0.99, raw={})

    provider = _RecordingProvider()
    confirmation.annotate_and_confirm(
        segments, segment_confidences=[], confidence_threshold=0.5, gap_threshold_seconds=1.5, provider=provider
    )

    assert provider.confirmed_texts == ["c", "d"]


def test_annotate_and_confirm_overrides_speaker_on_confident_disagreement(monkeypatch):
    monkeypatch.setattr(nemotron_diarization, "confidence_for_range", lambda *a, **k: 0.1)

    segments = [_segment(0.0, 1.0, "hola")]
    segments[0]["words"] = [{"word": "hola", "speaker": "SPEAKER_00"}]

    class _DisagreeingProvider(confirmation.SpeakerConfirmationProvider):
        def confirm_speaker(self, text, candidate_speakers, context_before=None, context_after=None):
            return confirmation.ConfirmationResult(speaker="SPEAKER_01", confidence=0.9, raw={})

    confirmation.annotate_and_confirm(
        segments, [], confidence_threshold=0.5, gap_threshold_seconds=1.5, provider=_DisagreeingProvider()
    )

    assert segments[0]["speaker"] == "SPEAKER_01"
    assert segments[0]["words"][0]["speaker"] == "SPEAKER_01"
    assert segments[0]["speaker_confirmed_by"] == "jev_override"


def test_annotate_and_confirm_keeps_speaker_on_low_confidence_disagreement(monkeypatch):
    monkeypatch.setattr(nemotron_diarization, "confidence_for_range", lambda *a, **k: 0.1)

    segments = [_segment(0.0, 1.0, "hola")]

    class _WeakDisagreeingProvider(confirmation.SpeakerConfirmationProvider):
        def confirm_speaker(self, text, candidate_speakers, context_before=None, context_after=None):
            return confirmation.ConfirmationResult(speaker="SPEAKER_01", confidence=0.2, raw={})

    confirmation.annotate_and_confirm(
        segments, [], confidence_threshold=0.5, gap_threshold_seconds=1.5, provider=_WeakDisagreeingProvider()
    )

    assert segments[0]["speaker"] == "SPEAKER_00"
    assert segments[0]["speaker_confirmed_by"] == "jev_confirmed"


def test_annotate_and_confirm_swallows_provider_errors(monkeypatch):
    monkeypatch.setattr(nemotron_diarization, "confidence_for_range", lambda *a, **k: 0.1)

    segments = [_segment(0.0, 1.0, "hola")]

    class _BrokenProvider(confirmation.SpeakerConfirmationProvider):
        def confirm_speaker(self, *a, **k):
            raise RuntimeError("boom")

    result = confirmation.annotate_and_confirm(
        segments, [], confidence_threshold=0.5, gap_threshold_seconds=1.5, provider=_BrokenProvider()
    )

    assert result[0]["speaker"] == "SPEAKER_00"
    assert "speaker_confirmed_by" not in result[0]

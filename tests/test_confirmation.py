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


def test_jev_provider_confirm_speaker_builds_expected_payload(monkeypatch):
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
    )

    assert result.speaker == "SPEAKER_01"
    assert result.confidence == 0.87
    assert captured["url"] == "https://api.typesafe.ai/v1/systemone"
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert captured["json"]["state"]["utterance"] == "y luego que paso"
    assert captured["json"]["questions"]["speaker"]["type"] == "choice"
    assert set(captured["json"]["questions"]["speaker"]["criteria"]) == {"SPEAKER_00", "SPEAKER_01"}


def test_jev_provider_confirm_continuation_batches_into_one_request(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured.update(json=json)
        return _FakeResponse({
            "answers": {
                "segment_0": {"noul": 0.9},
                "segment_1": {"noul": 0.1},
                "segment_2": {"noul": 0.8},
            }
        })

    monkeypatch.setattr(confirmation.requests, "post", fake_post)
    post_calls = []
    original_post = confirmation.requests.post

    def counting_post(*a, **k):
        post_calls.append(1)
        return original_post(*a, **k)

    monkeypatch.setattr(confirmation.requests, "post", counting_post)

    provider = confirmation.JevConfirmationProvider(api_key="secret-key")
    results = provider.confirm_continuation(
        established_context="hola como estas. bien y tu.",
        established_speaker="SPEAKER_00",
        candidate_texts=["todo bien por aca", "no lo puedo creer", "sigo aqui"],
    )

    assert len(post_calls) == 1  # three segments, one HTTP request
    assert len(captured["json"]["questions"]) == 3
    assert [r.same_speaker for r in results] == [True, False, True]
    assert results[1].confidence == 0.1


def test_jev_provider_detect_identifying_content_batches_into_one_request(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured.update(json=json)
        return _FakeResponse({
            "answers": {
                "segment_0": {"noul": 0.95},
                "segment_1": {"noul": 0.05},
            }
        })

    monkeypatch.setattr(confirmation.requests, "post", fake_post)

    provider = confirmation.JevConfirmationProvider(api_key="secret-key")
    flags = provider.detect_identifying_content([
        "soy el juez de este despacho",
        "procedamos con la audiencia",
    ])

    assert flags == [True, False]
    assert len(captured["json"]["questions"]) == 2
    assert captured["json"]["questions"]["segment_0"]["type"] == "noul"
    # Segments go into the shared state dict, referenced by name in backticks
    # from each question's structured instructions object -- not string-
    # concatenated directly into a plain instructions string.
    assert captured["json"]["state"] == {
        "seg_0": "soy el juez de este despacho",
        "seg_1": "procedamos con la audiencia",
    }
    assert "`seg_0`" in captured["json"]["questions"]["segment_0"]["instructions"]["question"]
    assert "true" in captured["json"]["questions"]["segment_0"]["criteria"]
    assert "false" in captured["json"]["questions"]["segment_0"]["criteria"]


def test_jev_provider_classify_roles_batches_all_speakers_into_one_request(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured.update(json=json)
        return _FakeResponse({
            "answers": {
                "speaker_0": {"choice": "Juez", "confidence": 0.92},
                "speaker_1": {"choice": "Demandante", "confidence": 0.81},
            }
        })

    monkeypatch.setattr(confirmation.requests, "post", fake_post)

    provider = confirmation.JevConfirmationProvider(api_key="secret-key")
    results = provider.classify_roles(
        texts_by_label={
            "SPEAKER_00": "soy la jueza de este despacho",
            "SPEAKER_01": "represento al demandante",
        },
        candidate_roles={
            "Juez": "The speaker is a judge",
            "Demandante": "The speaker is the plaintiff",
            "Demandado": "The speaker is the defendant",
            "No determinado": "Cannot be determined from this text",
        },
    )

    assert results["SPEAKER_00"].speaker == "Juez"
    assert results["SPEAKER_00"].confidence == 0.92
    assert results["SPEAKER_01"].speaker == "Demandante"
    assert len(captured["json"]["questions"]) == 2  # one HTTP request, two speakers
    assert captured["json"]["questions"]["speaker_0"]["type"] == "choice"
    assert captured["json"]["questions"]["speaker_0"]["criteria"] == {
        "Juez": {"what": "The speaker is a judge"},
        "Demandante": {"what": "The speaker is the plaintiff"},
        "Demandado": {"what": "The speaker is the defendant"},
        "No determinado": {"what": "Cannot be determined from this text"},
    }
    assert captured["json"]["state"] == {
        "speaker_0": "soy la jueza de este despacho",
        "speaker_1": "represento al demandante",
    }
    assert "`speaker_0`" in captured["json"]["questions"]["speaker_0"]["instructions"]["question"]


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


class _RecordingProvider(confirmation.SpeakerConfirmationProvider):
    """Test double: continuation always says "same speaker" unless the text
    contains "CHANGE", in which case it signals a switch; identification
    always returns SPEAKER_99 (a stand-in for "someone new")."""

    def __init__(self):
        self.continuation_calls = []
        self.identification_calls = []

    def confirm_speaker(self, text, candidate_speakers, context_before=None, context_after=None):
        self.identification_calls.append(text)
        return confirmation.ConfirmationResult(speaker="SPEAKER_99", confidence=0.9, raw={})

    def confirm_continuation(self, established_context, established_speaker, candidate_texts):
        self.continuation_calls.append(list(candidate_texts))
        return [
            confirmation.ContinuationResult(same_speaker="CHANGE" not in text, confidence=0.9, raw={})
            for text in candidate_texts
        ]


def test_annotate_and_confirm_triggers_only_on_low_confidence_or_big_gap(monkeypatch):
    fixed_confidences = {
        (0.0, 1.0): 0.9,   # A: high confidence, first segment -> establishes context
        (1.1, 2.0): 0.2,   # B: low confidence -> ambiguous
        (2.1, 3.0): 0.9,   # C: high confidence, small gap -> not ambiguous, resets the run
        (10.0, 11.0): 0.9, # D: high confidence but big gap after C -> ambiguous
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

    provider = _RecordingProvider()
    confirmation.annotate_and_confirm(
        segments, segment_confidences=[], confidence_threshold=0.5, gap_threshold_seconds=1.5, provider=provider
    )

    # "b" and "d" are not consecutive ambiguous ("c" sits in between and is
    # confident enough to reset the run), so each gets its own single-segment
    # continuation batch instead of being merged together.
    assert provider.continuation_calls == [["b"], ["d"]]
    assert provider.identification_calls == []


def test_annotate_and_confirm_batches_a_consecutive_ambiguous_run(monkeypatch):
    monkeypatch.setattr(nemotron_diarization, "confidence_for_range", lambda *a, **k: 0.1)  # always ambiguous

    segments = [
        _segment(0.0, 1.0, "a"),   # first segment: no established context yet -> identified directly
        _segment(1.1, 2.0, "b"),
        _segment(2.1, 3.0, "c"),
    ]

    provider = _RecordingProvider()
    confirmation.annotate_and_confirm(
        segments, [], confidence_threshold=0.5, gap_threshold_seconds=1.5, provider=provider, batch_size=8
    )

    assert provider.identification_calls == ["a"]  # only the first, contextless segment
    assert provider.continuation_calls == [["b", "c"]]  # b and c batched into one call


def test_annotate_and_confirm_escalates_only_on_gap_driven_change_with_no_extra_speaker(monkeypatch):
    fixed_confidences = {(0.0, 1.0): 0.9, (10.0, 11.0): 0.9, (11.1, 12.0): 0.9}
    monkeypatch.setattr(
        nemotron_diarization,
        "confidence_for_range",
        lambda segment_confidences, start, end: fixed_confidences[(start, end)],
    )

    segments = [
        _segment(0.0, 1.0, "hola", speaker="SPEAKER_00"),  # high confidence, no gap -> establishes context
        _segment(10.0, 11.0, "CHANGE now someone else talks", speaker="SPEAKER_00"),  # big gap after "hola" -> ambiguous
        _segment(11.1, 12.0, "still the new person", speaker="SPEAKER_00"),  # small gap after that -> not ambiguous
    ]
    # No other speaker activity anywhere near the (1.0, 10.0) gap.
    segment_confidences = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "confidence": 0.9}]

    provider = _RecordingProvider()
    confirmation.annotate_and_confirm(
        segments, segment_confidences, confidence_threshold=0.5, gap_threshold_seconds=1.5, provider=provider
    )

    # Gap-driven change, nothing else detected in the gap -> safe to escalate.
    assert provider.identification_calls == ["CHANGE now someone else talks"]
    assert segments[1]["speaker"] == "SPEAKER_99"
    assert segments[1]["speaker_confirmed_by"] == "jev_override"
    # Third segment wasn't ambiguous at all -- just trusted and used to advance context.
    assert segments[2]["speaker"] == "SPEAKER_00"
    assert "speaker_confirmed_by" not in segments[2]


def test_annotate_and_confirm_does_not_escalate_when_extra_speaker_detected_in_gap(monkeypatch):
    fixed_confidences = {(0.0, 1.0): 0.9, (10.0, 11.0): 0.9}
    monkeypatch.setattr(
        nemotron_diarization,
        "confidence_for_range",
        lambda segment_confidences, start, end: fixed_confidences[(start, end)],
    )

    segments = [
        _segment(0.0, 1.0, "hola", speaker="SPEAKER_00"),
        _segment(10.0, 11.0, "CHANGE now someone else talks", speaker="SPEAKER_02"),
    ]
    # Nemotron's raw diarization picked up a third voice briefly inside the gap.
    segment_confidences = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "confidence": 0.9},
        {"start": 5.0, "end": 5.4, "speaker": "SPEAKER_05", "confidence": 0.8},
    ]

    provider = _RecordingProvider()
    confirmation.annotate_and_confirm(
        segments, segment_confidences, confidence_threshold=0.5, gap_threshold_seconds=1.5, provider=provider
    )

    # A gap-driven change, but a third speaker was picked up in it -> too risky to
    # guess between just the known speakers, so identification is never called.
    assert provider.identification_calls == []
    assert segments[1]["speaker"] == "SPEAKER_02"  # Nemotron's own raw guess, untouched
    assert segments[1]["speaker_confirmed_by"] == "uncertain_not_escalated"


def test_annotate_and_confirm_does_not_escalate_on_confidence_only_change(monkeypatch):
    fixed_confidences = {(0.0, 1.0): 0.9, (1.1, 2.0): 0.1}
    monkeypatch.setattr(
        nemotron_diarization,
        "confidence_for_range",
        lambda segment_confidences, start, end: fixed_confidences[(start, end)],
    )

    segments = [
        _segment(0.0, 1.0, "hola", speaker="SPEAKER_00"),
        _segment(1.1, 2.0, "CHANGE now someone else talks", speaker="SPEAKER_02"),  # low conf, tiny gap
    ]

    provider = _RecordingProvider()
    confirmation.annotate_and_confirm(
        segments, [], confidence_threshold=0.5, gap_threshold_seconds=1.5, provider=provider
    )

    # Low-confidence-only ambiguity (no real gap) isn't a safe enough signal to
    # justify a multi-way identification guess, even though continuity flagged
    # a change -- keep Nemotron's own raw guess.
    assert provider.identification_calls == []
    assert segments[1]["speaker"] == "SPEAKER_02"
    assert segments[1]["speaker_confirmed_by"] == "uncertain_not_escalated"


def test_annotate_and_confirm_overrides_speaker_on_confirmed_continuation_mismatch(monkeypatch):
    monkeypatch.setattr(nemotron_diarization, "confidence_for_range", lambda *a, **k: 0.1)

    segments = [
        _segment(0.0, 1.0, "hola", speaker="SPEAKER_00"),
        _segment(1.1, 2.0, "sigo aqui", speaker="SPEAKER_01"),  # Nemotron guessed a switch...
    ]
    segments[1]["words"] = [{"word": "sigo", "speaker": "SPEAKER_01"}]

    class _SaysStillSameProvider(confirmation.SpeakerConfirmationProvider):
        def confirm_speaker(self, text, candidate_speakers, context_before=None, context_after=None):
            return confirmation.ConfirmationResult(speaker="SPEAKER_00", confidence=0.9, raw={})

        def confirm_continuation(self, established_context, established_speaker, candidate_texts):
            # ...but the continuation check disagrees: still the established speaker.
            return [confirmation.ContinuationResult(same_speaker=True, confidence=0.95, raw={}) for _ in candidate_texts]

    confirmation.annotate_and_confirm(
        segments, [], confidence_threshold=0.5, gap_threshold_seconds=1.5, provider=_SaysStillSameProvider()
    )

    assert segments[1]["speaker"] == "SPEAKER_00"
    assert segments[1]["words"][0]["speaker"] == "SPEAKER_00"
    assert segments[1]["speaker_confirmed_by"] == "jev_override"


def test_annotate_and_confirm_swallows_continuation_errors(monkeypatch):
    monkeypatch.setattr(nemotron_diarization, "confidence_for_range", lambda *a, **k: 0.1)

    segments = [
        _segment(0.0, 1.0, "hola", speaker="SPEAKER_00"),
        _segment(1.1, 2.0, "sigo aqui", speaker="SPEAKER_00"),
    ]

    class _BrokenProvider(confirmation.SpeakerConfirmationProvider):
        def confirm_speaker(self, *a, **k):
            return confirmation.ConfirmationResult(speaker="SPEAKER_00", confidence=0.9, raw={})

        def confirm_continuation(self, *a, **k):
            raise RuntimeError("boom")

    result = confirmation.annotate_and_confirm(
        segments, [], confidence_threshold=0.5, gap_threshold_seconds=1.5, provider=_BrokenProvider()
    )

    assert result[1]["speaker"] == "SPEAKER_00"
    assert "speaker_confirmed_by" not in result[1]

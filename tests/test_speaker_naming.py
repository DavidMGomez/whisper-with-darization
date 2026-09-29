import json

import confirmation
import speaker_naming


class _FakeProvider(confirmation.SpeakerConfirmationProvider):
    """A provider whose detect/classify answers are scripted per test."""

    def __init__(self, flags_by_text, role_by_text):
        self._flags_by_text = flags_by_text
        self._role_by_text = role_by_text
        self.classify_calls = []

    def confirm_speaker(self, *a, **k):
        raise NotImplementedError

    def confirm_continuation(self, *a, **k):
        raise NotImplementedError

    def detect_identifying_content(self, texts):
        return [self._flags_by_text.get(text, False) for text in texts]

    def classify_roles(self, texts_by_label, candidate_roles):
        self.classify_calls.append((dict(texts_by_label), tuple(candidate_roles)))
        results = {}
        for label, text in texts_by_label.items():
            speaker, confidence = self._role_by_text.get(text, (speaker_naming.CANNOT_DETERMINE, 1.0))
            results[label] = confirmation.ConfirmationResult(speaker=speaker, confidence=confidence, raw={})
        return results


def _segments():
    return [
        {"start": 0.0, "end": 2.0, "text": "Buenos dias, soy la jueza de este despacho.", "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 4.0, "text": "Gracias senoria, represento al demandante.", "speaker": "SPEAKER_01"},
        {"start": 4.0, "end": 6.0, "text": "Procedamos entonces con la audiencia.", "speaker": "SPEAKER_00"},
        {"start": 3000.0, "end": 3002.0, "text": "algo mucho mas tarde en la audiencia", "speaker": "SPEAKER_01"},
    ]


def test_identify_speaker_roles_resolves_flagged_speakers():
    segments = _segments()
    provider = _FakeProvider(
        flags_by_text={
            "Buenos dias, soy la jueza de este despacho.": True,
            "Gracias senoria, represento al demandante.": True,
        },
        role_by_text={
            "Buenos dias, soy la jueza de este despacho.": ("Juez", 0.95),
            "Gracias senoria, represento al demandante.": ("Demandante", 0.9),
        },
    )

    result = speaker_naming.identify_speaker_roles(segments, provider=provider)

    by_speaker = {seg["speaker"]: seg["speaker_role"] for seg in result}
    assert by_speaker["SPEAKER_00"] == "Juez"
    assert by_speaker["SPEAKER_01"] == "Demandante"
    # The far-later segment inherits its speaker's resolved role too.
    assert result[-1]["speaker_role"] == "Demandante"


def test_identify_speaker_roles_leaves_unflagged_speakers_unresolved():
    segments = _segments()
    provider = _FakeProvider(flags_by_text={}, role_by_text={})

    result = speaker_naming.identify_speaker_roles(segments, provider=provider)

    assert all(seg["speaker_role"] is None for seg in result)


def test_identify_speaker_roles_cannot_determine_stays_unresolved():
    segments = _segments()[:2]
    provider = _FakeProvider(
        flags_by_text={
            "Buenos dias, soy la jueza de este despacho.": True,
            "Gracias senoria, represento al demandante.": True,
        },
        role_by_text={
            "Buenos dias, soy la jueza de este despacho.": (speaker_naming.CANNOT_DETERMINE, 0.9),
            "Gracias senoria, represento al demandante.": ("Demandante", 0.3),  # below default threshold
        },
    )

    result = speaker_naming.identify_speaker_roles(segments, provider=provider)

    assert all(seg["speaker_role"] is None for seg in result)


def test_identify_speaker_roles_provider_failure_degrades_gracefully():
    class _BrokenProvider(confirmation.SpeakerConfirmationProvider):
        def confirm_speaker(self, *a, **k):
            raise NotImplementedError

        def confirm_continuation(self, *a, **k):
            raise NotImplementedError

        def detect_identifying_content(self, texts):
            raise RuntimeError("boom")

    result = speaker_naming.identify_speaker_roles(_segments(), provider=_BrokenProvider())

    assert all(seg["speaker_role"] is None for seg in result)


def test_identify_speaker_roles_no_provider_is_noop():
    segments = _segments()
    result = speaker_naming.identify_speaker_roles(segments, provider=None)
    assert result is segments
    assert all("speaker_role" not in seg for seg in result)


def test_identify_speaker_roles_explicit_roles_skip_extraction(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("extract_candidate_identities should not be called when roles is given")

    monkeypatch.setattr(speaker_naming, "extract_candidate_identities", _boom)

    provider = _FakeProvider(
        flags_by_text={"Buenos dias, soy la jueza de este despacho.": True},
        role_by_text={"Buenos dias, soy la jueza de este despacho.": ("Juez", 0.95)},
    )

    speaker_naming.identify_speaker_roles(
        _segments(), provider=provider, roles=["Juez", "Demandante"], openai_api_key="unused"
    )

    classified_roles = provider.classify_calls[0][1]
    assert set(classified_roles) == {"Juez", "Demandante", speaker_naming.CANNOT_DETERMINE}


def test_identify_speaker_roles_uses_extracted_identities_when_available(monkeypatch):
    monkeypatch.setattr(
        speaker_naming, "extract_candidate_identities", lambda texts, api_key, model: ["Maria Fernanda Restrepo"]
    )

    provider = _FakeProvider(
        flags_by_text={"Buenos dias, soy la jueza de este despacho.": True},
        role_by_text={"Buenos dias, soy la jueza de este despacho.": ("Maria Fernanda Restrepo", 0.95)},
    )

    result = speaker_naming.identify_speaker_roles(
        _segments(), provider=provider, openai_api_key="sk-fake"
    )

    classified_roles = provider.classify_calls[0][1]
    assert set(classified_roles) == {"Maria Fernanda Restrepo", speaker_naming.CANNOT_DETERMINE}
    assert result[0]["speaker_role"] == "Maria Fernanda Restrepo"


def test_extract_candidate_identities_formats_name_and_role_combinations(monkeypatch):
    captured = {}

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [{
                    "message": {
                        "content": json.dumps([
                            {"name": "Maria Fernanda Restrepo", "role": "Juez"},
                            {"name": None, "role": "Demandante"},
                            {"name": "Carlos Gomez", "role": None},
                        ])
                    }
                }]
            }

    def fake_post(url, headers, json, timeout):
        captured.update(url=url, headers=headers, json=json)
        return _FakeResponse()

    monkeypatch.setattr(speaker_naming.requests, "post", fake_post)

    identities = speaker_naming.extract_candidate_identities(
        ["soy la jueza Maria Fernanda Restrepo", "represento al demandante", "mi nombre es Carlos Gomez"],
        api_key="sk-fake",
    )

    assert identities == ["Maria Fernanda Restrepo (Juez)", "Demandante", "Carlos Gomez"]
    assert captured["headers"]["Authorization"] == "Bearer sk-fake"
    assert captured["url"] == speaker_naming.OPENAI_CHAT_ENDPOINT


def test_format_identity_prefers_name_with_role():
    assert speaker_naming._format_identity("Maria Restrepo", "Juez") == "Maria Restrepo (Juez)"
    assert speaker_naming._format_identity(None, "Juez") == "Juez"
    assert speaker_naming._format_identity("Maria Restrepo", None) == "Maria Restrepo"
    assert speaker_naming._format_identity(None, None) is None
    assert speaker_naming._format_identity("  ", "  ") is None


def test_extract_candidate_identities_no_api_key_short_circuits(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("should not make a network call without an api_key")

    monkeypatch.setattr(speaker_naming.requests, "post", _boom)

    assert speaker_naming.extract_candidate_identities(["algo"], api_key=None) == []


def test_extract_candidate_identities_failure_degrades_gracefully(monkeypatch):
    def fake_post(*a, **k):
        raise RuntimeError("network boom")

    monkeypatch.setattr(speaker_naming.requests, "post", fake_post)

    assert speaker_naming.extract_candidate_identities(["algo"], api_key="sk-fake") == []


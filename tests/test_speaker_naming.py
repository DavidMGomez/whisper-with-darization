import json

import confirmation
import speaker_naming


class _FakeProvider(confirmation.SpeakerConfirmationProvider):
    """A provider whose detect/classify answers are scripted per test."""

    def __init__(self, flags_by_text, role_by_text):
        self._flags_by_text = flags_by_text
        self._role_by_text = role_by_text
        self.classify_calls = []

    def verify_identities(self, identities):
        return {label: 0.99 for label in identities}

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
            "Gracias senoria, represento al demandante.": ("Demandante", 0.95),
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
        speaker_naming, "extract_candidate_identities", lambda texts, api_key, model, **kwargs: ["Maria Fernanda Restrepo"]
    )

    provider = _FakeProvider(
        flags_by_text={"Buenos dias, soy la jueza de este despacho.": True},
        role_by_text={"Buenos dias, soy la jueza de este despacho.": ("Maria Fernanda Restrepo", 0.95)},
    )

    segments = _segments()
    segments[0]["text"] = "Soy Maria Fernanda Restrepo"
    provider._flags_by_text = {segments[0]["text"]: True}
    provider._role_by_text = {segments[0]["text"]: ("Maria Fernanda Restrepo", .99)}
    result = speaker_naming.identify_speaker_roles(
        segments, provider=provider, openai_api_key="sk-fake"
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



def test_rejects_low_unknown_and_nonfinite_confidence():
    import pytest
    for candidate, score in [('Juez', .67), ('Intruso', 1), ('Juez', float('nan')), ('Juez', float('inf'))]:
        text = _segments()[0]['text']
        provider = _FakeProvider({text: True}, {text: (candidate, score)})
        result = speaker_naming.identify_speaker_roles(_segments(), provider, roles=['Juez'])
        assert result[0]['speaker_identity'] is None


def test_identification_remains_a_suggestion_without_secondary_verification():
    text = 'La abogada Maria Restrepo tiene la palabra.'
    provider = _FakeProvider({text: True}, {text: ('Maria Restrepo', .99)})
    provider.verify_identities = lambda identities: {key: .1 for key in identities}
    segments = [dict(_segments()[0], text=text)]
    result = speaker_naming.identify_speaker_roles(segments, provider, roles=['Maria Restrepo'])
    assert result[0]['speaker_name'] == 'Maria Restrepo'
    assert result[0]['speaker_identity']['status'] == 'suggested'
    assert result[0]['speaker_identity']['verified'] is False


def test_named_identity_has_evidence_and_separate_fields():
    text = 'Mi nombre es Maria Restrepo y soy la jueza.'
    provider = _FakeProvider({text: True}, {text: ('Maria Restrepo (Juez)', .98)})
    result = speaker_naming.identify_speaker_roles([dict(_segments()[0], text=text)], provider,
                                                 roles=['Maria Restrepo (Juez)'])
    identity = result[0]['speaker_identity']
    assert (identity['name'], identity['role'], identity['confidence']) == ('Maria Restrepo', 'Juez', .98)
    assert identity['evidence'][0]['text'] == text


def test_duplicate_suggestions_keep_separate_clusters_without_verification():
    text = 'Mi nombre es Maria Restrepo.'
    provider = _FakeProvider({text: True}, {text: ('Maria Restrepo', .99)})
    segments = [dict(_segments()[0], text=text), dict(_segments()[1], text=text)]
    result = speaker_naming.identify_speaker_roles(segments, provider, roles=['Maria Restrepo'])
    assert all(s['speaker_identity']['status'] == 'suggested' for s in result)
    assert result[0]['speaker'] != result[1]['speaker']
    provider.verify_identities = lambda identities: {}
    result = speaker_naming.identify_speaker_roles(segments[:1], provider, roles=['Maria Restrepo'])
    assert result[0]['speaker_identity']['name'] == 'Maria Restrepo'


def test_role_only_extraction_does_not_become_person_name():
    candidate = speaker_naming._format_identity(None, 'Jueza de familia')
    assert speaker_naming.split_identity(candidate) == (None, 'Jueza de familia')


def test_jev_verification_batches_and_checks_explicit_self_identification(monkeypatch):
    provider = confirmation.JevConfirmationProvider(api_key='unused')
    captured = {}
    def post(**kwargs):
        captured.update(kwargs)
        return {'identity_0': {'noul': .99}, 'identity_1': {'noul': .2}}
    monkeypatch.setattr(provider, '_post', post)
    answer = provider.verify_identities({
        'S0': {'text': 'Soy Maria Restrepo', 'identity': 'Maria Restrepo'},
        'S1': {'text': 'Escuchemos a Maria Restrepo', 'identity': 'Maria Restrepo'},
    })
    assert answer == {'S0': .99, 'S1': .2}
    assert len(captured['questions']) == 2
    assert captured['questions']['identity_0']['type'] == 'noul'
    assert 'another person' in captured['questions']['identity_0']['instructions']['focus']


def test_identity_threshold_cannot_be_lowered():
    import pytest
    with pytest.raises(ValueError):
        speaker_naming.identify_speaker_roles(_segments(), _FakeProvider({}, {}), confidence_threshold=.5)


def test_identity_accepts_supported_brief_introduction_at_seventy_percent():
    text = 'Carlos Perez, defensor.'
    provider = _FakeProvider({text: True}, {text: ('Carlos Perez (Defensor)', .70)})
    provider.verify_identities = lambda identities: {key: .70 for key in identities}
    result = speaker_naming.identify_speaker_roles([dict(_segments()[0], text=text)], provider,
                                                 roles=['Carlos Perez (Defensor)'])
    assert result[0]['speaker_identity']['name'] == 'Carlos Perez'
    assert result[0]['speaker_identity']['confidence'] == .70


def test_mario_introduction_keeps_name_when_exact_role_is_uncertain():
    text = 'Mi nombre es Mario Enrique Gómez Jiménez, Procurador 115 Judicial 2 de la Delegatura de Asuntos Penales.'
    candidate = 'Mario Enrique Gómez Jiménez (Procurador)'
    provider = _FakeProvider({text: True}, {text: (candidate, .80)})
    provider.verify_identities = lambda identities: {key: .90 if value['kind'] == 'name' else .55
                                                    for key, value in identities.items()}
    result = speaker_naming.identify_speaker_roles([dict(_segments()[0], text=text)], provider, roles=[candidate])
    assert result[0]['speaker_identity']['name'] == 'Mario Enrique Gómez Jiménez'
    assert result[0]['speaker_identity']['role'] == 'Procurador'


def test_mario_introduction_accepts_both_fields_at_sixty_eight_percent():
    text = 'Mi nombre es Mario Enrique Gómez Jiménez, Procurador 115 Judicial 2 de la Delegatura de Asuntos Penales.'
    candidate = 'Mario Enrique Gómez Jiménez (Procurador)'
    provider = _FakeProvider({text: True}, {text: (candidate, .68)})
    provider.verify_identities = lambda identities: {key: .68 for key in identities}
    result = speaker_naming.identify_speaker_roles([dict(_segments()[0], text=text)], provider, roles=[candidate])
    assert result[0]['speaker_identity']['name'] == 'Mario Enrique Gómez Jiménez'
    assert result[0]['speaker_identity']['role'] == 'Procurador'


def test_second_verifier_no_longer_removes_classified_name_or_role():
    text = 'Soy Mario Gómez, procurador.'
    candidate = 'Mario Gómez (Procurador)'
    provider = _FakeProvider({text: True}, {text: (candidate, .85)})
    provider.verify_identities = lambda identities: {key: .55 if value['kind'] == 'name' else .90
                                                    for key, value in identities.items()}
    result = speaker_naming.identify_speaker_roles([dict(_segments()[0], text=text)], provider, roles=[candidate])
    assert result[0]['speaker_identity']['name'] == 'Mario Gómez'
    assert result[0]['speaker_identity']['role'] == 'Procurador'


def test_procedural_role_does_not_require_an_introduction_flag():
    text = 'Se instala la audiencia. El despacho deniega la solicitud y concede el recurso.'
    provider = _FakeProvider({}, {text: ('Juez', .9)})
    result = speaker_naming.identify_speaker_roles([dict(_segments()[0], text=text)], provider, roles=['Juez'])
    assert result[0]['speaker_identity']['name'] is None
    assert result[0]['speaker_identity']['role'] == 'Juez'


def test_punctuation_does_not_erase_a_literal_name():
    text = 'Mi nombre es Mario Enrique, Gómez Jiménez.'
    name = 'Mario Enrique Gómez Jiménez'
    provider = _FakeProvider({text: True}, {text: (name, .9)})
    result = speaker_naming.identify_speaker_roles([dict(_segments()[0], text=text)], provider, roles=[name])
    assert result[0]['speaker_identity']['name'] == name


def test_explicit_mario_name_survives_role_only_candidate_selection():
    text = 'Mi nombre es Mario Enrique Gómez Jiménez, Procurador 115 Judicial 2 de la Delegatura de Asuntos Penales.'
    provider = _FakeProvider({text: True}, {text: ('Procurador', .8)})
    result = speaker_naming.identify_speaker_roles([dict(_segments()[0], text=text)], provider, roles=['Procurador'])
    assert result[0]['speaker_identity']['name'] == 'Mario Enrique Gómez Jiménez'
    assert result[0]['speaker_identity']['role'] == 'Procurador'


def test_suggestions_never_call_verifier_or_include_embedding_metadata():
    text = 'Mi nombre es Mario Enrique Gómez Jiménez, procurador.'
    provider = _FakeProvider({text: True}, {text: ('Procurador', .8)})
    def verify(_):
        raise AssertionError('Secondary verification must not run')
    provider.verify_identities = verify
    result = speaker_naming.identify_speaker_roles(
        [dict(_segments()[0], text=text, speaker_embedding=object())], provider, roles=['Procurador'])
    identity = result[0]['speaker_identity']
    json.dumps(identity)
    assert identity['status'] == 'suggested'
    assert identity['name'] == 'Mario Enrique Gómez Jiménez'
    assert 'speaker_embedding' not in identity['evidence'][0]


def _mock_identity_response(monkeypatch, entities):
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {'choices': [{'message': {'content': json.dumps(entities)}}]}
    monkeypatch.setattr(speaker_naming.requests, 'post', lambda *a, **kw: Response())


def test_elliptical_defender_recovers_own_name_not_clients(monkeypatch):
    text = ('apoderada de víctimas y demás participantes en la presente vista por la defensa técnica '
            'del señor José Otto Valderrama Castro, representante de su despacho, Néstor Valderrama '
            'Castro, con datos de identificación que ya aparece en registro.')
    _mock_identity_response(monkeypatch, [
        {'name': 'José Otto Valderrama Castro', 'role': 'Acusado', 'introductions': []},
        {'name': 'Néstor Valderrama Castro', 'role': 'Defensor', 'introductions': [
            {'excerpt_index': 0, 'quote': text, 'confidence': .84}]},
        {'name': None, 'role': 'Defensor'},
    ])
    provider = _FakeProvider({text: True}, {text: ('Defensor', .91)})
    result = speaker_naming.identify_speaker_roles([dict(_segments()[0], text=text)], provider,
                                                  openai_api_key='fake')
    identity = result[0]['speaker_identity']
    assert identity['name'] == 'Néstor Valderrama Castro'
    assert identity['role'] == 'Defensor'
    assert identity['name_confidence'] == .84
    assert identity['status'] == 'suggested' and identity['verified'] is False


def test_name_recovery_rejects_ungrounded_or_low_score_metadata(monkeypatch):
    text = 'Por la defensa técnica, Néstor Castro.'
    for intro in [
        {'excerpt_index': 4, 'quote': text, 'confidence': .9},
        {'excerpt_index': 0, 'quote': 'Mi nombre es Néstor Castro', 'confidence': .9},
        {'excerpt_index': 0, 'quote': text, 'confidence': .6},
        {'excerpt_index': 0, 'quote': text, 'confidence': True},
    ]:
        _mock_identity_response(monkeypatch, [{'name': 'Néstor Castro', 'role': 'Defensor',
                                             'introductions': [intro]}])
        candidates = speaker_naming.extract_candidate_identities([text], 'fake')
        assert candidates[0].introductions == []


def test_name_in_next_piece_of_same_turn_is_included(monkeypatch):
    first, second = 'Por la defensa técnica del señor José Otto.', 'Néstor Castro, datos ya registrados.'
    seen = []
    def extract(texts, api_key, model, **kwargs):
        seen.extend(texts)
        candidate = speaker_naming._format_identity('Néstor Castro', 'Defensor')
        candidate.introductions = [{'excerpt_index': 1, 'confidence': .83}]
        return [candidate, 'Defensor']
    monkeypatch.setattr(speaker_naming, 'extract_candidate_identities', extract)
    segments = [dict(_segments()[0], text=first), dict(_segments()[0], start=2, end=4, text=second)]
    provider = _FakeProvider({first: True}, {first + ' ' + second: ('Defensor', .9)})
    result = speaker_naming.identify_speaker_roles(segments, provider, openai_api_key='fake')
    assert seen == [first, second]
    assert result[0]['speaker_name'] == 'Néstor Castro'


def test_competing_anchored_names_do_not_guess(monkeypatch):
    text = 'Néstor Castro y José Otto.'
    _mock_identity_response(monkeypatch, [
        {'name': name, 'role': 'Defensor', 'introductions': [
            {'excerpt_index': 0, 'quote': text, 'confidence': .9}]}
        for name in ['Néstor Castro', 'José Otto']
    ] + [{'name': None, 'role': 'Defensor'}])
    result = speaker_naming.identify_speaker_roles([dict(_segments()[0], text=text)],
        _FakeProvider({text: True}, {text: ('Defensor', .9)}), openai_api_key='fake')
    assert result[0]['speaker_identity']['name'] is None


def test_extractor_receives_adjacent_speakers_as_context_only(monkeypatch):
    captured = {}
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {'choices': [{'message': {'content': '[]'}}]}
    def post(*args, **kwargs):
        captured.update(kwargs['json'])
        return Response()
    monkeypatch.setattr(speaker_naming.requests, 'post', post)
    segments = [
        {'start': 0, 'end': 2, 'speaker': 'SPEAKER_00', 'text': 'Soy la jueza Ana López. Preséntese la defensa.'},
        {'start': 2, 'end': 4, 'speaker': 'SPEAKER_01', 'text': 'Néstor Castro, datos registrados.'},
        {'start': 4, 'end': 6, 'speaker': 'SPEAKER_00', 'text': 'Gracias, doctor. Representa al señor José Otto.'},
    ]
    provider = _FakeProvider({segments[1]['text']: True}, {})
    speaker_naming.identify_speaker_roles(segments, provider, openai_api_key='fake')
    payload = captured['messages'][1]['content'].split('\n\n', 1)[1]
    excerpts = json.loads(payload)
    assert len(excerpts) == 1
    assert excerpts[0]['target']['speaker'] == 'SPEAKER_01'
    assert excerpts[0]['previous']['speaker'] == 'SPEAKER_00'
    assert excerpts[0]['next']['text'] == segments[2]['text']

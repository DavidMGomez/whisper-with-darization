"""Speaker-role/name identification for the opening of a hearing.

Two different tools for two different jobs:

- Jev (confirmation.py's SpeakerConfirmationProvider) for anything that fits
  a closed question: detecting *whether* a segment contains identifying
  content, and *choosing* among a known candidate list who's speaking.
  TypeSafe's real API (docs.typesafe.ai/api.md) only supports three
  question types -- noul (yes/no), choice (closed selection), and score
  (numeric level) -- none of which is open-ended free text, so Jev cannot
  itself extract an arbitrary name or compile a list of "whoever gets
  mentioned" -- it can only ever pick from candidates it's already given.

- OpenAI (extract_candidate_identities, a plain chat-completion call) for
  the one genuinely open-ended job: reading the flagged segments' actual
  text and extracting/deduplicating the literal names and/or roles
  mentioned into a clean candidate list. This is real free-text extraction,
  which is exactly what Jev cannot do.

Pipeline:

1. Detection (Jev, batched, cheap): for every segment in the hearing's
   opening window, ask a yes/no question -- does this segment contain a
   self-introduction, or someone stating/being addressed by a name or role?
   Same "pack many named questions into one request" trick
   confirmation.py's confirm_continuation already uses.

2. Candidate extraction (OpenAI, one request per hearing): feed every
   flagged segment's text to a plain LLM call that extracts and dedupes the
   distinct person names and/or roles actually mentioned -- e.g. "Juez",
   "Maria Fernanda Restrepo", "Demandante". Falls back to the fixed
   DEFAULT_ROLES list if this isn't configured (no API key) or fails, so a
   missing/flaky OpenAI call degrades to generic role classification
   instead of breaking anything.

3. Classification (Jev, batched, one request for every flagged speaker):
   for diarization labels with at least one flagged segment, ask which of
   the extracted candidates each one most likely is -- a closed choice
   question per speaker, all packed into a single request, always
   including an explicit "cannot determine" option.

4. Verification (Jev, batched): require explicit self-identification,
   a choice score and verification score >= 0.70, and a name occurring in
   the speaker's own text. Preserve structured name, role and source evidence.
   These are provider scores, not calibrated probabilities of identity.

A provider/extraction failure at any tier is logged and treated as
"unresolved" for the affected segments/speakers/list -- it never fails the
whole prediction, the same tolerance confirmation.py already applies to
every Jev call.

Known limitation: this only has real introduction content to work with for
the first audio a hearing's speakers appear in. If a hearing is split into
multiple parts upstream, later parts have no visibility into earlier parts'
resolved identities and fall back to unresolved -- carrying them forward
across parts is out of scope for this module.
"""
import json
import logging
import math
import re
import unicodedata
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_INTRO_WINDOW_SECONDS = 1800
MIN_IDENTITY_CONFIDENCE = 0.70

# Fallback only, used when OpenAI extraction isn't configured or fails --
# a deliberately long, general-purpose universe covering civil, criminal,
# family and administrative hearings, since without a real per-hearing
# candidate list there's no way to narrow it further.
DEFAULT_ROLES = [
    "Juez",
    "Magistrado",
    "Fiscal",
    "Procurador",
    "Ministerio Publico",
    "Demandante",
    "Demandado",
    "Apoderado del Demandante",
    "Apoderado del Demandado",
    "Defensor",
    "Defensor Publico",
    "Acusado",
    "Imputado",
    "Victima",
    "Testigo",
    "Perito",
    "Secretario",
    "Conciliador",
    "Interprete o Traductor",
]

CANNOT_DETERMINE = "No determinado"

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
OPENAI_CHAT_ENDPOINT = "https://api.openai.com/v1/chat/completions"


class IdentityLabel(str):
    """Keep extracted name/role fields without breaking legacy string callers."""
    def __new__(cls, label, name, role):
        value = super().__new__(cls, label)
        value.name, value.role = name or None, role or None
        return value


def _format_identity(name: Optional[str], role: Optional[str]) -> Optional[str]:
    """Combines a person's name and role into one display label, preferring
    "Name (Role)" when both are known, degrading to just whichever one is
    available, per the user's explicit preference: a named identity with
    its role beats a bare role, which beats nothing."""
    name = name.strip() if isinstance(name, str) else ""
    role = role.strip() if isinstance(role, str) else ""
    if name and role:
        return IdentityLabel(f"{name} ({role})", name, role)
    return IdentityLabel(name or role, name, role) if name or role else None


def extract_candidate_identities(
    texts: List[str],
    api_key: Optional[str],
    model: str = DEFAULT_OPENAI_MODEL,
    timeout_seconds: float = 15.0,
) -> List[str]:
    """Given segments flagged as containing a name/role/self-introduction,
    extracts a clean, deduplicated list of candidate identities actually
    mentioned, each formatted as "Name (Role)" when both are known, or just
    the name or just the role when only one is. Best-effort: returns [] on
    any failure (no api_key, network error, malformed response) --
    identify_speaker_roles() falls back to DEFAULT_ROLES when this is
    empty, so a flaky/misconfigured extraction degrades gracefully rather
    than failing the whole prediction.
    """
    if not texts or not api_key:
        return []

    try:
        combined = "\n".join(f"- {text}" for text in texts)
        response = requests.post(
            OPENAI_CHAT_ENDPOINT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": 0,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You extract person identities mentioned or self-"
                            "introduced in excerpts from a legal hearing's "
                            "transcript. For each distinct person, extract "
                            "their name if stated, and their formal role "
                            "(e.g. judge, plaintiff, defendant, witness, "
                            "prosecutor, defense counsel, clerk) if it can be "
                            "determined. Respond with ONLY a JSON array of "
                            "objects, each with a \"name\" and a \"role\" key "
                            "(either may be null if not determinable), "
                            "deduplicated by person, and nothing else."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Extract every distinct person mentioned in "
                            f"these transcript excerpts:\n\n{combined}"
                        ),
                    },
                ],
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        entities = json.loads(content)
        labels = []
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            label = _format_identity(entity.get("name"), entity.get("role"))
            if label:
                labels.append(label)
        return labels
    except Exception as exc:  # noqa: BLE001 - a flaky extraction call must not fail the request
        logger.warning("Candidate identity extraction failed, falling back to the default role list: %s", exc)
        return []


def identify_speaker_roles(
    segments: List[dict],
    provider,
    intro_window_seconds: int = DEFAULT_INTRO_WINDOW_SECONDS,
    roles: Optional[List[str]] = None,
    confidence_threshold: float = MIN_IDENTITY_CONFIDENCE,
    openai_api_key: Optional[str] = None,
    openai_model: str = DEFAULT_OPENAI_MODEL,
) -> List[dict]:
    """Attaches `speaker_role` (a resolved name/role, or None if unresolved)
    to every segment, based only on segments within the first
    `intro_window_seconds` of this audio. Mutates and returns `segments` in
    place, matching confirmation.annotate_and_confirm's style.

    `roles`, if given, is used as the fixed candidate list directly
    (skipping OpenAI extraction entirely -- explicit control, no extra
    request). Otherwise candidates are extracted from the flagged segments'
    actual content via OpenAI, falling back to DEFAULT_ROLES if that isn't
    configured or fails.
    """
    if provider is None or not segments:
        return segments

    if not math.isfinite(confidence_threshold) or not MIN_IDENTITY_CONFIDENCE <= confidence_threshold <= 1:
        raise ValueError("Identity confidence must be between 0.70 and 1")
    for seg in segments:
        seg["speaker_identity"] = None
        seg["speaker_name"] = None
        seg["speaker_role"] = None

    intro_segments = [seg for seg in segments if seg.get("start", 0) < intro_window_seconds]
    if not intro_segments:
        for seg in segments:
            seg["speaker_role"] = None
        return segments

    try:
        flags = provider.detect_identifying_content([seg.get("text", "") for seg in intro_segments])
    except Exception as exc:  # noqa: BLE001 - a flaky provider call must not fail the request
        logger.warning("Identifying-content detection failed, leaving speaker roles unresolved: %s", exc)
        for seg in segments:
            seg["speaker_role"] = None
        return segments

    flagged_texts_by_label: Dict[str, List[str]] = {}
    for seg, flagged in zip(intro_segments, flags):
        label = seg.get("speaker")
        if flagged is True and label:
            flagged_texts_by_label.setdefault(label, []).append(seg.get("text", ""))

    if roles:
        candidate_role_names = list(roles)
    else:
        all_flagged_texts = [text for texts in flagged_texts_by_label.values() for text in texts]
        candidate_role_names = extract_candidate_identities(all_flagged_texts, api_key=openai_api_key, model=openai_model)
        if not candidate_role_names:
            candidate_role_names = list(DEFAULT_ROLES)
    if CANNOT_DETERMINE not in candidate_role_names:
        candidate_role_names.append(CANNOT_DETERMINE)

    # classify_roles takes a description per option (used as its selection
    # criteria) rather than a bare name -- see confirmation.py's
    # JevConfirmationProvider.classify_roles for why.
    candidate_roles = {
        role: ("Cannot be determined from this text" if role == CANNOT_DETERMINE
               else f"The speaker is {role}")
        for role in candidate_role_names
    }

    texts_by_label = {label: " ".join(texts) for label, texts in flagged_texts_by_label.items()}
    try:
        results_by_label = provider.classify_roles(texts_by_label, candidate_roles=candidate_roles)
    except Exception as exc:  # noqa: BLE001 - a flaky provider call must not fail the request
        logger.warning("Role classification failed, leaving speaker roles unresolved: %s", exc)
        results_by_label = {}

    # A high choice score alone does not establish that a mentioned person
    # is the speaker. Require a second, explicit self-identification check.
    proposed = {}
    for label, result in results_by_label.items():
        if (label not in texts_by_label or result.speaker == CANNOT_DETERMINE
                or result.speaker not in candidate_roles
                or not math.isfinite(result.confidence)
                or not confidence_threshold <= result.confidence <= 1):
            continue
        candidate = next(candidate for candidate in candidate_role_names if candidate == result.speaker)
        name, role = split_identity(candidate)
        if name and not re.search(r"(?<!\w)" + re.escape(normalize(name)) + r"(?!\w)", normalize(texts_by_label[label])):
            continue
        proposed[label] = result
    try:
        verified = provider.verify_identities({
            label: {"text": texts_by_label[label], "identity": result.speaker}
            for label, result in proposed.items()
        }) if proposed else {}
    except Exception:  # Unsupported providers must fail closed for identity assignment.
        logger.warning("Identity verification unavailable; leaving speakers unresolved")
        verified = {}

    identities = {}
    for label, result in proposed.items():
        score = verified.get(label)
        if (isinstance(score, bool) or not isinstance(score, (float, int))
                or not math.isfinite(score) or not confidence_threshold <= score <= 1):
            continue
        candidate = next(candidate for candidate in candidate_role_names if candidate == result.speaker)
        name, role = split_identity(candidate)
        identities[label] = {
            "name": name, "role": role, "label": result.speaker,
            "confidence": min(result.confidence, score),
            "evidence": [
                {"start": seg.get("start"), "end": seg.get("end"), "text": seg.get("text", "")}
                for seg in intro_segments
                if seg.get("speaker") == label and seg.get("text", "") in flagged_texts_by_label[label]
            ],
        }
    # Conflicting diarization clusters require review, not an automatic merge.
    names = [normalize(identity["name"]) for identity in identities.values() if identity["name"]]
    identities = {label: identity for label, identity in identities.items()
                  if not identity["name"] or names.count(normalize(identity["name"])) == 1}
    for seg in segments:
        identity = identities.get(seg.get("speaker"))
        seg["speaker_identity"] = identity
        seg["speaker_name"] = identity["name"] if identity else None
        # Keep the existing display-label contract for older consumers.
        seg["speaker_role"] = identity["label"] if identity else None
    return segments


def normalize(text):
    text = unicodedata.normalize("NFKD", text.casefold())
    return " ".join("".join(c for c in text if not unicodedata.combining(c)).split())


def split_identity(label):
    if isinstance(label, IdentityLabel):
        return label.name, label.role
    if label in DEFAULT_ROLES:
        return None, label
    match = re.fullmatch(r"(.+?) \(([^()]+)\)", label)
    return (match.group(1), match.group(2)) if match else (label, None)

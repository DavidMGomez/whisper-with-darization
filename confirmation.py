"""Pluggable speaker-confirmation providers.

A confirmation provider takes an ambiguous speaker-turn assignment (low
diarization confidence, or a large time gap since the previous turn) and
asks a fast text-based model to sanity-check it against conversational
context. This is a semantic check, not a re-run of acoustic diarization: it
catches "this sentence doesn't sound like it continues the same speaker"
cases, not acoustic misidentification.

Providers are registered in PROVIDERS below, keyed by name, so switching to a
different vendor later is a matter of adding one class and one registry
entry -- no changes needed anywhere else in predict.py.
"""
import abc
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class ConfirmationResult:
    speaker: str
    confidence: float
    raw: dict


class SpeakerConfirmationProvider(abc.ABC):
    @abc.abstractmethod
    def confirm_speaker(
        self,
        text: str,
        candidate_speakers: List[str],
        context_before: Optional[str],
        context_after: Optional[str],
    ) -> ConfirmationResult:
        ...


class JevConfirmationProvider(SpeakerConfirmationProvider):
    """Uses TypeSafe's Jev "System One" model (https://docs.typesafe.ai) as a
    speaker-confirmation oracle via its `choice` question primitive.

    NOTE: the endpoint below is taken from TypeSafe's published HTTP API
    reference (docs.typesafe.ai/api.md) as of this writing. Verify it against
    your own TypeSafe account/dashboard before relying on this in production
    -- override via the JEV_ENDPOINT env var if it differs.
    """

    DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
    DEFAULT_MODEL = "jev-latest"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        endpoint: Optional[str] = None,
        timeout_seconds: float = 3.0,
    ):
        if not api_key:
            raise ValueError("A Jev API key is required (jev_api_key input or JEV_API_KEY env var)")
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint or os.environ.get("JEV_ENDPOINT", self.DEFAULT_ENDPOINT)
        self.timeout_seconds = timeout_seconds

    def confirm_speaker(self, text, candidate_speakers, context_before=None, context_after=None):
        payload = {
            "model": self.model,
            "state": {
                "previous_utterance": context_before or "",
                "utterance": text,
                "next_utterance": context_after or "",
            },
            "questions": {
                "speaker": {
                    "type": "choice",
                    "instructions": (
                        "This is a fragment of a transcribed multi-speaker conversation. "
                        "Given the previous and next utterances, decide which speaker most "
                        "likely said 'utterance'."
                    ),
                    "criteria": {speaker: None for speaker in candidate_speakers},
                }
            },
        }
        response = requests.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        answer = response.json()["answers"]["speaker"]
        return ConfirmationResult(
            speaker=answer["choice"],
            confidence=float(answer.get("confidence", 0.0)),
            raw=answer,
        )


PROVIDERS: Dict[str, type] = {
    "jev": JevConfirmationProvider,
}


def get_confirmation_provider(name: str, api_key: Optional[str]) -> SpeakerConfirmationProvider:
    if name not in PROVIDERS:
        raise ValueError(f"Unknown confirmation provider '{name}'. Available: {list(PROVIDERS)}")
    resolved_key = api_key or os.environ.get("JEV_API_KEY")
    return PROVIDERS[name](api_key=resolved_key)


def annotate_and_confirm(
    segments: List[dict],
    segment_confidences: List[dict],
    confidence_threshold: float,
    gap_threshold_seconds: float,
    provider: Optional[SpeakerConfirmationProvider],
):
    """Attaches diarization confidence to each transcript segment and,
    when a provider is given, asks it to confirm segments that are either
    low-confidence or follow an unusually large gap from the previous
    segment (a long pause makes a speaker change more likely).

    Mutates and returns `segments` in place. Never raises: a confirmation
    provider failure is logged and skipped so it can't take down the whole
    prediction.
    """
    import nemotron_diarization

    all_speakers = sorted({seg["speaker"] for seg in segments if seg.get("speaker")})
    previous_end = None

    for i, seg in enumerate(segments):
        confidence = nemotron_diarization.confidence_for_range(
            segment_confidences, seg["start"], seg["end"]
        )
        seg["speaker_confidence"] = confidence

        gap = (seg["start"] - previous_end) if previous_end is not None else 0.0
        previous_end = seg["end"]

        needs_confirmation = confidence < confidence_threshold or gap > gap_threshold_seconds
        if not (needs_confirmation and provider is not None):
            continue

        try:
            result = provider.confirm_speaker(
                text=seg.get("text", ""),
                candidate_speakers=all_speakers or [seg.get("speaker", "SPEAKER_00")],
                context_before=segments[i - 1].get("text") if i > 0 else None,
                context_after=segments[i + 1].get("text") if i + 1 < len(segments) else None,
            )
        except Exception as exc:  # noqa: BLE001 - a flaky confirmation call must not fail the request
            logger.warning("Speaker confirmation call failed, keeping diarization result: %s", exc)
            continue

        seg["jev_checked"] = True
        seg["jev_confidence"] = result.confidence
        if result.speaker and result.speaker != seg.get("speaker") and result.confidence >= 0.5:
            original_speaker = seg.get("speaker")
            seg["speaker"] = result.speaker
            seg["speaker_confirmed_by"] = "jev_override"
            for word in seg.get("words", []):
                if word.get("speaker") == original_speaker:
                    word["speaker"] = result.speaker
        else:
            seg["speaker_confirmed_by"] = "jev_confirmed"

    return segments

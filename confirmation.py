"""Pluggable speaker-confirmation providers.

Two-tier design:

1. Continuity check (cheap, batched): for a run of consecutive ambiguous
   segments (low diarization confidence or a big silence gap), ask "is this
   still the same speaker as the established context?" as a yes/no question
   per segment -- but pack all of them into ONE request to the provider
   (multiple named questions against one shared state), since Jev has no
   multi-state batch endpoint but does evaluate many questions per request
   in parallel. This is the common case: most ambiguous segments really are
   just the same speaker continuing.

2. Identification (choice, one request per case): only when the
   continuity check says "no, that's a different speaker" do we ask the
   more expensive "which of the known speakers is this" question, since we
   now know a change happened but not who it changed to.

This mirrors real diarization-review workflow: assume continuity, escalate
to identification only on an actual detected change. It also cuts request
volume a lot compared to asking "who is this" for every single ambiguous
segment.
"""
import abc
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_WINDOW = 3
DEFAULT_BATCH_SIZE = 8


@dataclass
class ConfirmationResult:
    speaker: str
    confidence: float
    raw: dict


@dataclass
class ContinuationResult:
    same_speaker: bool
    confidence: float
    raw: dict


class SpeakerConfirmationProvider(abc.ABC):
    @abc.abstractmethod
    def confirm_speaker(
        self,
        text: str,
        candidate_speakers: List[str],
        context_before: Optional[str] = None,
        context_after: Optional[str] = None,
    ) -> ConfirmationResult:
        """Identifies which of candidate_speakers most likely said text."""
        ...

    @abc.abstractmethod
    def confirm_continuation(
        self,
        established_context: str,
        established_speaker: str,
        candidate_texts: List[str],
    ) -> List[ContinuationResult]:
        """Checks, in as few requests as the provider can manage, whether
        each of candidate_texts (in order) is still established_speaker
        talking, given established_context (their preceding utterances).
        Must return exactly one result per candidate text, in order.
        """
        ...


class JevConfirmationProvider(SpeakerConfirmationProvider):
    """Uses TypeSafe's Jev "System One" model (https://docs.typesafe.ai).

    NOTE: the endpoint below is taken from TypeSafe's published HTTP API
    reference (docs.typesafe.ai/api.md) as of this writing. Verify it against
    your own TypeSafe account/dashboard before relying on this in production
    -- override via the JEV_ENDPOINT env var if it differs.

    Jev has no endpoint for batching multiple *states* in one call, but a
    single request's `questions` map is evaluated in parallel and billed on
    input tokens only (extra questions are cheap) -- so confirm_continuation
    packs one `noul` question per candidate segment into a single request
    instead of making one request per segment.
    """

    DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
    DEFAULT_MODEL = "jev-latest"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        endpoint: Optional[str] = None,
        timeout_seconds: float = 5.0,
    ):
        if not api_key:
            raise ValueError("A Jev API key is required (jev_api_key input or JEV_API_KEY env var)")
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint or os.environ.get("JEV_ENDPOINT", self.DEFAULT_ENDPOINT)
        self.timeout_seconds = timeout_seconds

    def _post(self, state, questions) -> dict:
        payload = {"model": self.model, "state": state, "questions": questions}
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
        return response.json()["answers"]

    def confirm_speaker(self, text, candidate_speakers, context_before=None, context_after=None):
        answers = self._post(
            state={
                "previous_utterance": context_before or "",
                "utterance": text,
                "next_utterance": context_after or "",
            },
            questions={
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
        )
        answer = answers["speaker"]
        return ConfirmationResult(
            speaker=answer["choice"],
            confidence=float(answer.get("confidence", 0.0)),
            raw=answer,
        )

    def confirm_continuation(self, established_context, established_speaker, candidate_texts):
        if not candidate_texts:
            return []

        questions = {
            f"segment_{i}": {
                "type": "noul",
                "instructions": (
                    "'established_context' is a run of consecutive utterances already "
                    f"confirmed to be from the same speaker ({established_speaker}). Is the "
                    "'candidate_utterance' below also that same speaker continuing to talk, "
                    "rather than a different person responding?\n\n"
                    f"candidate_utterance: {text}"
                ),
            }
            for i, text in enumerate(candidate_texts)
        }
        answers = self._post(state={"established_context": established_context}, questions=questions)

        results = []
        for i in range(len(candidate_texts)):
            answer = answers[f"segment_{i}"]
            noul = float(answer.get("noul", 0.0))
            results.append(ContinuationResult(same_speaker=noul >= 0.5, confidence=noul, raw=answer))
        return results


PROVIDERS: Dict[str, type] = {
    "jev": JevConfirmationProvider,
}


def get_confirmation_provider(name: str, api_key: Optional[str]) -> SpeakerConfirmationProvider:
    if name not in PROVIDERS:
        raise ValueError(f"Unknown confirmation provider '{name}'. Available: {list(PROVIDERS)}")
    resolved_key = api_key or os.environ.get("JEV_API_KEY")
    return PROVIDERS[name](api_key=resolved_key)


def _apply_speaker(seg: dict, speaker: str, confirmed_by: str) -> None:
    original_speaker = seg.get("speaker")
    seg["speaker"] = speaker
    seg["speaker_confirmed_by"] = confirmed_by
    for word in seg.get("words", []):
        if word.get("speaker") == original_speaker:
            word["speaker"] = speaker


def _identify_and_apply(
    seg: dict,
    all_speakers: List[str],
    provider: SpeakerConfirmationProvider,
    context_before: Optional[str] = None,
) -> None:
    try:
        result = provider.confirm_speaker(
            text=seg.get("text", ""),
            candidate_speakers=all_speakers or [seg.get("speaker", "SPEAKER_00")],
            context_before=context_before,
        )
    except Exception as exc:  # noqa: BLE001 - a flaky provider call must not fail the request
        logger.warning("Speaker identification call failed, keeping diarization result: %s", exc)
        return

    seg["jev_checked"] = True
    seg["jev_confidence"] = result.confidence
    if result.speaker and result.speaker != seg.get("speaker") and result.confidence >= 0.5:
        _apply_speaker(seg, result.speaker, "jev_override")
    else:
        seg["speaker_confirmed_by"] = "jev_confirmed"


def annotate_and_confirm(
    segments: List[dict],
    segment_confidences: List[dict],
    confidence_threshold: float,
    gap_threshold_seconds: float,
    provider: Optional[SpeakerConfirmationProvider],
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    """Attaches diarization confidence to each transcript segment and, when a
    provider is given, double-checks segments that are either low-confidence
    or follow an unusually large gap from the previous segment.

    Ambiguous segments are checked via a batched continuity question first
    ("is this still the established speaker?"); only a segment where that
    comes back negative gets the more expensive per-segment identification
    call. Mutates and returns `segments` in place. A provider failure is
    logged and skipped -- it never fails the whole prediction.
    """
    import nemotron_diarization

    if not segments:
        return segments

    all_speakers = sorted({seg["speaker"] for seg in segments if seg.get("speaker")})

    previous_end = None
    ambiguous = []
    for seg in segments:
        confidence = nemotron_diarization.confidence_for_range(segment_confidences, seg["start"], seg["end"])
        seg["speaker_confidence"] = confidence
        gap = (seg["start"] - previous_end) if previous_end is not None else 0.0
        previous_end = seg["end"]
        ambiguous.append(confidence < confidence_threshold or gap > gap_threshold_seconds)

    if provider is None:
        return segments

    established_speaker = None
    context_texts: List[str] = []

    def push_context(text: str) -> None:
        context_texts.append(text)
        del context_texts[:-context_window]

    i = 0
    n = len(segments)
    while i < n:
        if not ambiguous[i]:
            seg = segments[i]
            established_speaker = seg.get("speaker", established_speaker)
            push_context(seg.get("text", ""))
            i += 1
            continue

        run_start = i
        while i < n and ambiguous[i] and (i - run_start) < batch_size:
            i += 1
        run = segments[run_start:i]

        if not established_speaker or not context_texts:
            # No established speaker/context yet (e.g. the very first segment
            # is ambiguous) -- identify just the first segment of the run to
            # bootstrap one, then batch-check the rest of the run against it
            # below like normal.
            seg = run[0]
            _identify_and_apply(seg, all_speakers, provider)
            established_speaker = seg.get("speaker", established_speaker)
            push_context(seg.get("text", ""))
            run = run[1:]

        if not run:
            continue

        try:
            results = provider.confirm_continuation(
                established_context=" ".join(context_texts),
                established_speaker=established_speaker,
                candidate_texts=[seg.get("text", "") for seg in run],
            )
        except Exception as exc:  # noqa: BLE001 - a flaky provider call must not fail the request
            logger.warning("Continuation check failed, keeping diarization result: %s", exc)
            for seg in run:
                push_context(seg.get("text", ""))
            continue

        for seg, result in zip(run, results):
            seg["jev_checked"] = True
            seg["jev_continuation_confidence"] = result.confidence
            if result.same_speaker:
                if seg.get("speaker") != established_speaker:
                    _apply_speaker(seg, established_speaker, "jev_override")
                else:
                    seg["speaker_confirmed_by"] = "jev_confirmed"
                push_context(seg.get("text", ""))
            else:
                # A change was detected but not who it changed to -- escalate
                # to identification, then re-anchor context from here so the
                # rest of the run (if any) is compared against the right
                # established speaker.
                _identify_and_apply(seg, all_speakers, provider)
                established_speaker = seg.get("speaker", established_speaker)
                context_texts = [seg.get("text", "")]

    return segments

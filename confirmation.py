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
   continuity check says "no, that's a different speaker" AND it's safe to
   guess do we ask the more expensive "which of the known speakers is this"
   question. "Safe" is deliberately narrow: the change has to be driven by
   a real silence gap (the classic turn-taking cue, not just a low-confidence
   segment), and nothing else may have been picked up during that gap --
   a third voice active in it means this isn't a clean two-party switch, and
   guessing among only the known speakers would be too error-prone. When
   escalation isn't safe, we keep Nemotron's own raw guess rather than force
   a risky multi-way classification.

This mirrors real diarization-review workflow: assume continuity, escalate
to identification only on an actual, low-risk detected change. It also cuts
request volume a lot compared to asking "who is this" for every single
ambiguous segment.
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

    def detect_identifying_content(self, texts: List[str]) -> List[bool]:
        """For each of texts (in order), decides whether it contains a
        self-introduction or someone stating/being addressed by a name or
        role. Batched into as few requests as the provider can manage. Must
        return exactly one bool per text, in order.

        Not an abstract method: providers that don't support speaker-role
        classification (speaker_naming.py's feature) can simply not
        override this -- the default raises, which speaker_naming.py
        already treats the same as any other provider failure (leaves the
        affected segments unresolved rather than failing the request).
        """
        raise NotImplementedError

    def classify_roles(
        self, texts_by_label: Dict[str, str], candidate_roles: Dict[str, str]
    ) -> Dict[str, ConfirmationResult]:
        """Given each flagged speaker's combined utterances (keyed by
        diarization label), decides which of candidate_roles each one most
        likely is, in as few requests as the provider can manage.
        candidate_roles maps each option to a short description used as its
        selection criteria, and should always include an explicit "cannot
        determine" option. Must return exactly one result per key in
        texts_by_label.

        Not abstract for the same reason as detect_identifying_content.
        """
        raise NotImplementedError



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

    def detect_identifying_content(self, texts):
        # Per docs.typesafe.ai's how-to-build-with-system-one guide (the
        # triage_ticket.py reference example): every segment goes into the
        # shared `state` dict under its own key; each question's
        # `instructions` is a structured object (a "question" field
        # referencing its one segment by name in backticks, plus a "focus"
        # field for extra guidance) rather than a bare string or a
        # duplicated copy of the segment text; and `noul` questions spell
        # out explicit true/false criteria with examples instead of leaving
        # the model to infer them purely from the instructions text.
        if not texts:
            return []

        state = {f"seg_{i}": text for i, text in enumerate(texts)}
        questions = {
            f"segment_{i}": {
                "type": "noul",
                "instructions": {
                    "question": (
                        f"Does `seg_{i}` contain a self-introduction, "
                        "someone stating their own name or role, or someone "
                        "else addressing them by name or role?"
                    ),
                    "focus": (
                        "This is a segment from a transcribed legal hearing. "
                        "Roles include e.g. judge, plaintiff, defendant, "
                        "prosecutor, defense counsel, witness, clerk."
                    ),
                },
                "criteria": {
                    "true": {
                        "what": (
                            "A self-introduction, a stated name or role "
                            "(e.g. judge, plaintiff, defendant, prosecutor, "
                            "defense counsel, witness, clerk), or someone "
                            "being addressed by name or role"
                        ),
                        "examples": [
                            "Buenos dias, soy la jueza de este despacho.",
                            "Representa usted al demandado, doctor Gomez?",
                        ],
                    },
                    "false": {
                        "what": "No name or role is mentioned or addressed",
                        "examples": ["Procedamos entonces con la audiencia."],
                    },
                },
            }
            for i in range(len(texts))
        }
        answers = self._post(state=state, questions=questions)
        return [float(answers[f"segment_{i}"].get("noul", 0.0)) >= 0.5 for i in range(len(texts))]

    def classify_roles(self, texts_by_label, candidate_roles):
        # Batched, like confirm_continuation/detect_identifying_content: one
        # "choice" question per speaker in a single request instead of one
        # request per speaker. Same state/instructions-object pattern as
        # detect_identifying_content; each candidate role gets its own
        # criteria description rather than a bare placeholder.
        if not texts_by_label:
            return {}

        labels = list(texts_by_label.keys())
        state = {f"speaker_{i}": texts_by_label[label] for i, label in enumerate(labels)}
        questions = {
            f"speaker_{i}": {
                "type": "choice",
                "instructions": {
                    "question": f"Which role does `speaker_{i}` most likely have?",
                    "focus": (
                        f"`speaker_{i}` lists the combined statements of a "
                        "single speaker from the opening of a legal hearing. "
                        "Choose the cannot-determine option explicitly "
                        "rather than guessing if it isn't clear from this "
                        "text."
                    ),
                },
                "criteria": {
                    role: {"what": description}
                    for role, description in candidate_roles.items()
                },
            }
            for i, label in enumerate(labels)
        }
        answers = self._post(state=state, questions=questions)

        results = {}
        for i, label in enumerate(labels):
            answer = answers[f"speaker_{i}"]
            results[label] = ConfirmationResult(
                speaker=answer["choice"],
                confidence=float(answer.get("confidence", 0.0)),
                raw=answer,
            )
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
    gap_triggered = []
    gap_spans: List[Optional[tuple]] = []
    for seg in segments:
        confidence = nemotron_diarization.confidence_for_range(segment_confidences, seg["start"], seg["end"])
        seg["speaker_confidence"] = confidence
        is_gap = previous_end is not None and (seg["start"] - previous_end) > gap_threshold_seconds
        gap_spans.append((previous_end, seg["start"]) if is_gap else None)
        gap_triggered.append(is_gap)
        ambiguous.append(confidence < confidence_threshold or is_gap)
        previous_end = seg["end"]

    if provider is None:
        return segments

    established_speaker = None
    context_texts: List[str] = []

    def push_context(text: str) -> None:
        context_texts.append(text)
        del context_texts[:-context_window]

    def safe_to_identify(idx: int) -> bool:
        """Only escalate to full multi-way identification for a gap-driven
        change (a real silence, the classic turn-taking cue), and only if
        nothing else was picked up during that gap -- a third voice active
        in it means this isn't a clean two-party switch we can safely guess
        between just the known speakers."""
        if not gap_triggered[idx]:
            return False
        span = gap_spans[idx]
        if span is None or span[0] is None:
            return True
        nearby = nemotron_diarization.distinct_speakers_in_range(segment_confidences, span[0], span[1])
        extra = nearby - {established_speaker, segments[idx].get("speaker")}
        return not extra

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
        run_indices = list(range(run_start, i))

        if not established_speaker or not context_texts:
            # No established speaker/context yet (e.g. the very first segment
            # is ambiguous) -- identify just the first segment of the run to
            # bootstrap one, then batch-check the rest of the run against it
            # below like normal.
            seg = run[0]
            _identify_and_apply(seg, all_speakers, provider)
            established_speaker = seg.get("speaker", established_speaker)
            push_context(seg.get("text", ""))
            run, run_indices = run[1:], run_indices[1:]

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

        for idx, seg, result in zip(run_indices, run, results):
            seg["jev_checked"] = True
            seg["jev_continuation_confidence"] = result.confidence
            if result.same_speaker:
                if seg.get("speaker") != established_speaker:
                    _apply_speaker(seg, established_speaker, "jev_override")
                else:
                    seg["speaker_confirmed_by"] = "jev_confirmed"
                push_context(seg.get("text", ""))
            elif safe_to_identify(idx):
                # A change was detected, it's a real gap, and nothing else
                # was picked up in it -- safe enough to ask who it actually
                # is, then re-anchor context from here.
                _identify_and_apply(seg, all_speakers, provider)
                established_speaker = seg.get("speaker", established_speaker)
                context_texts = [seg.get("text", "")]
            else:
                # A change was detected but identifying who is too risky here
                # (not a clean gap-driven switch, or another voice was picked
                # up in the gap) -- keep Nemotron's own raw guess rather than
                # force a multi-way guess, and anchor onward context on it.
                seg["speaker_confirmed_by"] = "uncertain_not_escalated"
                established_speaker = seg.get("speaker", established_speaker)
                context_texts = [seg.get("text", "")]

    return segments

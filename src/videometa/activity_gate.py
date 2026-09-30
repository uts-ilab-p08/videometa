"""Choose the parts of a video worth annotating by asking the VLM how much happens.

The video is cut into fixed chunks (10 s by default) that overlap their
neighbour (2 s by default), so an action that straddles a boundary is seen
whole by at least one chunk and every chunk opens with a little of what came
before it. Each chunk is written out as a small, low-frame-rate MP4 and shown
to the same vision-language model that later writes the event annotations,
with a prompt asking for one number: how much activity is in the clip, from 0
(nothing moves) to 1 (a busy scene full of interactions), judged on how many
people and vehicles move, how much they move, and how many distinct actions or
interactions occur. The score is kept with the chunk, and the chunks at or
above `ActivityGateConfig.score_threshold` go on to object tracking and event
annotation.

This replaces a pixel-motion gate. A foreground-mask gate cannot tell a person
opening a car door from a tree moving in the wind, and it scored a bus crossing
the foreground far above a distant figure texting; the model is asked about
subjects and actions, which is what the annotation stage will be looking for.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import logging
from pathlib import Path
import re
from tempfile import mkdtemp
from typing import Any, Protocol, Sequence

from videometa.annotation import RelevantWindow, VideoInfo, probe_video
from videometa.local_qwen import DEFAULT_MODEL_ID, generate_from_video, load_local_qwen

logger = logging.getLogger(__name__)


#: The rubric the model scores a chunk against. The bands are written around
#: the MEVA activity taxonomy: a person or vehicle merely passing through is
#: not a MEVA activity, so it sits low; door, trunk, vehicle-entry, meeting,
#: phone and carrying actions are what the taxonomy is made of, so a single
#: clear one lands in the middle of the scale regardless of how large the
#: subject is in the frame.
ACTIVITY_SCORE_GUIDANCE = (
    "Rate how much activity happens in the clip on a scale from 0 to 1, judging "
    "three things together: how many people, bicycles and vehicles are present "
    "and actually moving (not parked vehicles, trees, shadows or lighting "
    "changes), how much they move, and how many distinct actions or "
    "interactions take place.\n"
    "\n"
    "Use this scale:\n"
    "- 0.0: nothing moves. An empty scene, or only parked vehicles, foliage, "
    "shadows or flicker.\n"
    "- 0.1 to 0.2: one person or vehicle passes through and nothing else "
    "happens, or something moves only at the very edge of the frame.\n"
    "- 0.3 to 0.4: one subject does one small thing beyond passing through "
    "(stops, turns back, looks at or talks on a phone, reads something, "
    "carries or puts down an object, sits or stands), or two subjects move "
    "independently of each other.\n"
    "- 0.5 to 0.6: a clear, specific action or interaction: a vehicle door or "
    "trunk opens or closes, someone gets into or out of a vehicle, loads or "
    "unloads it, two people meet, talk, embrace or hand something over, a "
    "vehicle stops, starts, reverses, turns or picks up or drops off someone, "
    "someone opens a building door or goes through a doorway.\n"
    "- 0.7 to 0.8: several such actions, or several subjects interacting, in "
    "the same clip.\n"
    "- 0.9 to 1.0: a busy scene with many subjects and many simultaneous "
    "actions.\n"
    "\n"
    "Count every person, bicycle and vehicle that moves as a subject. Do not "
    "let size decide the score: a small, distant person doing something "
    "specific scores the same as a large one close to the camera. Do not "
    "raise the score for scenery, weather, or a parked vehicle that never "
    "moves."
)

#: The reply shape the gate parses.
ACTIVITY_SCORE_FORMAT = (
    'Return JSON only, with no prose before or after it: {"score": number '
    'between 0 and 1, "subjects": [one short phrase per moving subject, such '
    'as "person in a dark jacket" or "white SUV"], "event_count": whole number '
    'of distinct actions or interactions you can see, "summary": one sentence '
    'saying what happens}. If nothing moves, return {"score": 0.0, '
    '"subjects": [], "event_count": 0, "summary": "No activity."}.'
)


def chunk_spans(
    duration_seconds: float, chunk_seconds: float, overlap_seconds: float
) -> list[tuple[float, float]]:
    """Cut `duration_seconds` into `chunk_seconds` spans that overlap by `overlap_seconds`.

    Spans start every ``chunk_seconds - overlap_seconds``. The last span is
    shortened to the end of the video rather than padded past it, and no span
    is started that would add less than the overlap of new footage: with 10 s
    chunks and 2 s overlap a 21 s video gives ``[0-10, 8-18, 16-21]``, not a
    fourth 24-25 s span that would show nothing the third did not.
    """
    if duration_seconds <= 0:
        return []
    stride = chunk_seconds - overlap_seconds
    spans: list[tuple[float, float]] = []
    start = 0.0
    while True:
        end = min(start + chunk_seconds, duration_seconds)
        spans.append((round(start, 3), round(end, 3)))
        if end >= duration_seconds:
            break
        start += stride
        if duration_seconds - start <= overlap_seconds and spans:
            # The remaining footage is already inside the previous span.
            break
    return spans


@dataclass(frozen=True)
class ActivityGateConfig:
    """Parameters for VLM-scored chunk selection.

    **How the video is cut.** `chunk_seconds` is the length of every chunk and
    `overlap_seconds` how much of the previous chunk each one repeats, so that
    a window carries the context of what came just before it and an action on
    a boundary is whole in at least one chunk.

    **What the model sees.** Each chunk is written as an MP4 of `clip_size`
    at `clip_fps`, whole frame, no overlays; the model is scoring activity, so
    detector boxes would only bias it. A 10 s chunk at 2 FPS is 20 frames.

    **Which chunks survive.** A chunk is relevant when its score is at least
    `score_threshold`. The default of 0.3 keeps anything beyond a lone person
    or vehicle passing through: the MEVA taxonomy has no "walks" or "drives"
    activity but does include reading, texting, carrying and sitting, which
    the rubric places in the 0.3-0.4 band, so a threshold of 0.5 would drop
    them along with the empty car parks. Raise it to 0.5 to keep only chunks
    with a clear door, vehicle-entry, meeting or manoeuvre. `max_windows` and
    `max_total_seconds` then cap what reaches the next stage, highest score
    first.
    """

    chunk_seconds: float = 10.0
    overlap_seconds: float = 2.0
    score_threshold: float = 0.3
    clip_size: tuple[int, int] = (640, 360)
    clip_fps: float = 2.0
    max_windows: int | None = None
    max_total_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive")
        if not 0 <= self.overlap_seconds < self.chunk_seconds:
            raise ValueError("overlap_seconds must be within [0, chunk_seconds)")
        if not 0 <= self.score_threshold <= 1:
            raise ValueError("score_threshold must be within [0, 1]")
        width, height = self.clip_size
        if width <= 0 or height <= 0:
            raise ValueError("clip_size dimensions must be positive")
        if self.clip_fps <= 0:
            raise ValueError("clip_fps must be positive")
        if self.max_windows is not None and self.max_windows < 0:
            raise ValueError("max_windows cannot be negative")
        if self.max_total_seconds is not None and self.max_total_seconds < 0:
            raise ValueError("max_total_seconds cannot be negative")

    @property
    def stride_seconds(self) -> float:
        return self.chunk_seconds - self.overlap_seconds


class ActivityScorer(Protocol):
    """Pluggable model that rates the activity in one clip."""

    def score(
        self, clip_path: str, start_seconds: float, end_seconds: float
    ) -> dict[str, Any]:
        """Return ``{"score": float, "subjects": [str], "event_count": int, "summary": str}``."""


def build_activity_prompt(start_seconds: float, end_seconds: float, clip_fps: float) -> str:
    """The scoring prompt for one chunk."""
    return (
        "You are screening footage from a fixed surveillance camera. This clip "
        f"covers video time {start_seconds:.1f}s to {end_seconds:.1f}s, sampled "
        f"at {clip_fps:g} frames per second, so movement appears in jumps rather "
        "than smoothly.\n\n"
        f"{ACTIVITY_SCORE_GUIDANCE}\n\n"
        f"{ACTIVITY_SCORE_FORMAT}"
    )


def parse_activity_score(text: str) -> dict[str, Any]:
    """Read the model's reply into the four fields, tolerating fences, prose and cut-offs.

    The score is the one field the gate cannot do without, so when the JSON
    does not parse it is recovered from a ``"score": 0.4`` fragment; the other
    fields fall back to empty values. A reply with no score at all raises.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    parsed: Any = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if 0 <= start < end:
            try:
                parsed = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                parsed = None
    if not isinstance(parsed, dict):
        match = re.search(r'"?score"?\s*[:=]\s*([0-9]*\.?[0-9]+)', cleaned)
        if match is None:
            raise ValueError(f"No activity score in model reply: {text[:200]!r}")
        parsed = {"score": float(match.group(1))}
    try:
        score = float(parsed.get("score", 0.0))
    except (TypeError, ValueError):
        raise ValueError(f"Activity score is not a number: {parsed.get('score')!r}") from None
    subjects = parsed.get("subjects", [])
    if isinstance(subjects, str):
        subjects = [subjects]
    if not isinstance(subjects, list):
        subjects = []
    try:
        event_count = max(0, int(parsed.get("event_count", 0) or 0))
    except (TypeError, ValueError):
        event_count = 0
    summary = parsed.get("summary", "")
    return {
        "score": round(max(0.0, min(1.0, score)), 3),
        "subjects": [str(item).strip() for item in subjects if str(item).strip()],
        "event_count": event_count,
        "summary": str(summary).strip() if summary is not None else "",
    }


class LocalQwenActivityScorer:
    """Score chunks with a local MLX Qwen3-VL model.

    Pass `model` and `processor` to reuse weights already loaded for the
    annotation stage (see `from_annotator`), or a `model_id` to load them here.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        model: Any | None = None,
        processor: Any | None = None,
        video_fps: float = 2.0,
        max_tokens: int = 300,
    ) -> None:
        """`video_fps` must match the `clip_fps` the clips were written at.
        `max_tokens` covers a JSON reply with a dozen subjects; the prompt asks
        for one sentence, and a longer answer is cut without losing the score."""
        if (model is None) != (processor is None):
            raise ValueError("pass both model and processor, or neither")
        if model is None:
            model, processor = load_local_qwen(model_id)
        self.model = model
        self.processor = processor
        self.video_fps = video_fps
        self.max_tokens = max_tokens

    @classmethod
    def from_annotator(cls, annotator: Any, **kwargs: Any) -> "LocalQwenActivityScorer":
        """Share the weights of a `LocalQwenEventAnnotator` instead of loading twice."""
        return cls(model=annotator.model, processor=annotator.processor, **kwargs)

    def score(
        self, clip_path: str, start_seconds: float, end_seconds: float
    ) -> dict[str, Any]:
        prompt = build_activity_prompt(start_seconds, end_seconds, self.video_fps)
        reply = generate_from_video(
            self.model,
            self.processor,
            clip_path,
            prompt,
            fps=self.video_fps,
            max_tokens=self.max_tokens,
        )
        return parse_activity_score(reply)


class ActivityWindowFinder:
    """Cut a video into overlapping chunks, score each with the VLM, keep the active ones."""

    def __init__(
        self,
        scorer: ActivityScorer,
        config: ActivityGateConfig | None = None,
        *,
        clip_directory: str | Path | None = None,
    ) -> None:
        """`clip_directory` is where the per-chunk MP4s are written; each video
        gets its own subdirectory. With None they go to a temporary directory
        and are still recorded in `RelevantWindow.clip_path`."""
        self.scorer = scorer
        self.config = config or ActivityGateConfig()
        self.clip_directory = Path(clip_directory) if clip_directory else None

    def probe(self, video_path: str | Path) -> VideoInfo:
        return probe_video(video_path)

    def find(self, video_path: str | Path) -> tuple[VideoInfo, list[RelevantWindow]]:
        """Score every chunk, then flag the ones over the threshold and under the budget."""
        info, scored = self.score_video(video_path)
        return info, self.build_windows(scored)

    def score_video(self, video_path: str | Path) -> tuple[VideoInfo, list[RelevantWindow]]:
        """Write every chunk's clip and ask the scorer about each one.

        Every chunk comes back, in order, with `is_relevant` False; the score,
        subjects, event count and summary are on the window so a caller can
        store or plot them before deciding a threshold. A scorer failure on one
        chunk is recorded in `error` with a score of 0 rather than stopping the
        run.
        """
        info = self.probe(video_path)
        spans = chunk_spans(
            info.duration_seconds, self.config.chunk_seconds, self.config.overlap_seconds
        )
        directory = self._clip_directory_for(info)
        logger.info(
            "Activity gate: %d chunks of %.1fs (overlap %.1fs) for %s, clips in %s",
            len(spans),
            self.config.chunk_seconds,
            self.config.overlap_seconds,
            info.path,
            directory,
        )
        clips = write_chunk_clips(
            info,
            spans,
            directory,
            clip_size=self.config.clip_size,
            clip_fps=self.config.clip_fps,
        )
        windows: list[RelevantWindow] = []
        for index, ((start, end), clip_path) in enumerate(zip(spans, clips), start=1):
            base = RelevantWindow(
                start_seconds=start,
                end_seconds=end,
                start_frame=round(start * info.fps),
                end_frame=min(info.frame_count - 1, round(end * info.fps)),
                score=0.0,
                is_relevant=False,
                clip_path=str(clip_path) if clip_path else None,
            )
            if clip_path is None:
                windows.append(replace(base, error="no frames decoded for this chunk"))
                logger.warning("Chunk %d/%d %.1fs-%.1fs has no frames", index, len(spans), start, end)
                continue
            try:
                result = self.scorer.score(str(clip_path), start, end)
                window = replace(
                    base,
                    score=float(result["score"]),
                    subjects=tuple(result.get("subjects", ())),
                    event_count=int(result.get("event_count", 0)),
                    summary=str(result.get("summary", "")),
                )
            except Exception as error:  # noqa: BLE001 - one bad chunk must not end the run
                window = replace(base, error=str(error))
                logger.warning(
                    "Chunk %d/%d %.1fs-%.1fs failed to score: %s", index, len(spans), start, end, error
                )
            else:
                logger.info(
                    "Chunk %d/%d %.1fs-%.1fs scored %.2f (%d subjects, %d events): %s",
                    index,
                    len(spans),
                    start,
                    end,
                    window.score,
                    len(window.subjects),
                    window.event_count,
                    window.summary,
                )
            windows.append(window)
        return info, windows

    def build_windows(self, scored: Sequence[RelevantWindow]) -> list[RelevantWindow]:
        """Apply the threshold, then the budget, to already-scored chunks.

        Every chunk is returned with `is_relevant` set, so a caller can always
        see what was rejected and why. The budget is spent highest score first,
        ties broken by the earlier chunk.
        """
        config = self.config
        kept = [
            window for window in scored
            if window.error is None and window.score >= config.score_threshold
        ]
        chosen: set[tuple[float, float]] = set()
        total = 0.0
        for window in sorted(kept, key=lambda item: (-item.score, item.start_seconds)):
            if config.max_windows is not None and len(chosen) >= config.max_windows:
                break
            if (
                config.max_total_seconds is not None
                and total + window.duration_seconds > config.max_total_seconds
            ):
                continue
            chosen.add((window.start_seconds, window.end_seconds))
            total += window.duration_seconds
        windows = [
            replace(window, is_relevant=(window.start_seconds, window.end_seconds) in chosen)
            for window in scored
        ]
        logger.info(
            "Activity gate kept %d of %d chunks (%.1fs) at threshold %.2f "
            "(max_windows=%s, max_total_seconds=%s)",
            len(chosen),
            len(windows),
            total,
            config.score_threshold,
            config.max_windows,
            config.max_total_seconds,
        )
        return windows

    def _clip_directory_for(self, info: VideoInfo) -> Path:
        if self.clip_directory is None:
            return Path(mkdtemp(prefix="videometa-activity-"))
        directory = self.clip_directory / Path(info.path).stem
        directory.mkdir(parents=True, exist_ok=True)
        return directory


def write_chunk_clips(
    info: VideoInfo,
    spans: Sequence[tuple[float, float]],
    directory: Path,
    *,
    clip_size: tuple[int, int],
    clip_fps: float,
) -> list[Path | None]:
    """Write one downscaled MP4 per span in a single sequential pass over the video.

    Spans overlap, so a frame can belong to two clips at once; each open
    writer takes the frames in its span at its own `clip_fps` stride, counted
    from the span's first frame. Returns one path per span, None where no
    frame could be decoded for it.
    """
    from videometa.window_annotation import _open_mp4_writer

    cv2 = _import_cv2()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fps = info.fps or 30.0
    clip_fps = min(clip_fps, fps)
    step = max(1, round(fps / clip_fps))
    plans = [
        _ClipPlan(
            start=round(start * fps),
            end=min(info.frame_count - 1, round(end * fps)),
            path=directory / f"chunk_{start:.3f}s_{end:.3f}s.mp4",
        )
        for start, end in spans
    ]
    if not plans:
        return []
    last_frame = max(plan.end for plan in plans)

    capture = cv2.VideoCapture(info.path)
    if not capture.isOpened():
        raise OSError(f"Cannot open video: {info.path}")
    frame_index = 0
    try:
        while frame_index <= last_frame:
            ok, frame = capture.read()
            if not ok:
                break
            resized = None
            for plan in plans:
                if not plan.wants(frame_index, step):
                    continue
                if resized is None:
                    resized = cv2.resize(frame, clip_size, interpolation=cv2.INTER_AREA)
                if plan.writer is None:
                    plan.writer = _open_mp4_writer(plan.path, clip_fps, clip_size)
                    if plan.writer is None:
                        raise OSError(f"Cannot create chunk clip: {plan.path}")
                plan.writer.write(resized)
                plan.written += 1
            # Close writers whose span has passed, so only the overlapping
            # neighbours are ever open at once.
            for plan in plans:
                if frame_index >= plan.end:
                    plan.close()
            frame_index += 1
    finally:
        capture.release()
        for plan in plans:
            plan.close()
    logger.info("Wrote %d chunk clips to %s", sum(1 for plan in plans if plan.written), directory)
    return [plan.path if plan.written else None for plan in plans]


class _ClipPlan:
    """Bookkeeping for one chunk's clip while the source is read once."""

    def __init__(self, start: int, end: int, path: Path) -> None:
        self.start = start
        self.end = end
        self.path = path
        self.writer: Any | None = None
        self.written = 0
        self.closed = False

    def wants(self, frame_index: int, step: int) -> bool:
        return (
            not self.closed
            and self.start <= frame_index <= self.end
            and (frame_index - self.start) % step == 0
        )

    def close(self) -> None:
        if self.writer is not None and not self.closed:
            self.writer.release()
        self.closed = True


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as error:
        raise ImportError("Video processing requires `pip install videometa[vision]`.") from error
    return cv2

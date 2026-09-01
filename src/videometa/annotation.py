"""Reusable video annotation pipeline.

Heavy integrations are imported only when they are used, keeping package import
lightweight and letting callers select their preferred detector and VLM backend.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import logging
from pathlib import Path
from shutil import copyfileobj
from statistics import mean, median, stdev
from tempfile import gettempdir
from typing import Any, Callable, Protocol, Sequence
from urllib.parse import urlparse
from urllib.request import urlopen

_STATISTIC_THRESHOLDS = {
    "avg": "avg",
    "average": "avg",
    "mean": "avg",
    "median": "median",
    "std": "std",
    "mean+std": "std",
    "mean_std": "std",
}
_THRESHOLD_CHOICES = "'avg'/'median'/'std'"
_STD_DIRECTIONS = {"upper", "lower", "both"}


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MotionGateConfig:
    """Parameters controlling motion-based relevant-window selection.

    `motion_threshold` may be a non-negative number or a statistic name:
    ``"avg"`` / ``"median"`` / ``"std"``. ``"std"`` uses one or both of
    ``mean ± motion_std_k * std`` (default k=1.5), controlled by
    `motion_std_direction`. Statistic names are computed from the video's
    sampled motion scores.
    """

    sample_fps: float | None = None
    gate_size: tuple[int, int] = (640, 360)
    window_seconds: float = 60.0
    stride_seconds: float = 50.0
    motion_threshold: float | str = 0.002
    motion_std_k: float = 1.5
    motion_std_direction: str = "upper"
    warmup_seconds: float = 2.0
    mog_history: int = 300
    mog_variance_threshold: float = 24.0

    def __post_init__(self) -> None:
        if (
            (self.sample_fps is not None and self.sample_fps <= 0)
            or self.window_seconds <= 0
            or self.stride_seconds <= 0
        ):
            raise ValueError("sample_fps, window_seconds, and stride_seconds must be positive")
        if self.warmup_seconds < 0:
            raise ValueError("warmup_seconds cannot be negative")
        if self.motion_std_k < 0:
            raise ValueError("motion_std_k cannot be negative")
        if self.motion_std_direction.strip().lower() not in _STD_DIRECTIONS:
            raise ValueError("motion_std_direction must be 'upper', 'lower', or 'both'")
        if isinstance(self.motion_threshold, str):
            if self.motion_threshold.strip().lower() not in _STATISTIC_THRESHOLDS:
                raise ValueError(
                    f"motion_threshold must be a non-negative number or {_THRESHOLD_CHOICES}"
                )
        elif isinstance(self.motion_threshold, bool) or not isinstance(self.motion_threshold, (int, float)):
            raise ValueError(
                f"motion_threshold must be a non-negative number or {_THRESHOLD_CHOICES}"
            )
        elif self.motion_threshold < 0:
            raise ValueError("motion_threshold cannot be negative")

    def resolve_threshold(self, scores: Sequence[float]) -> float:
        """Return the upper numeric cutoff, computing a statistic when requested.

        For a ``"std"`` threshold, use `resolve_std_thresholds()` to obtain
        both bounds when `motion_std_direction` is ``"lower"`` or ``"both"``.
        """
        threshold = self.motion_threshold
        if isinstance(threshold, (int, float)) and not isinstance(threshold, bool):
            return float(threshold)
        if not scores:
            return 0.0
        mode = _STATISTIC_THRESHOLDS[str(threshold).strip().lower()]
        if mode == "avg":
            return float(mean(scores))
        if mode == "median":
            return float(median(scores))
        spread = stdev(scores) if len(scores) > 1 else 0.0
        return float(mean(scores) + self.motion_std_k * spread)

    def resolve_std_thresholds(self, scores: Sequence[float]) -> tuple[float | None, float | None]:
        """Return the active (lower, upper) standard-deviation bounds.

        Non-``"std"`` thresholds return ``(None, resolved_threshold)`` so
        callers can consistently use the upper bound for existing behaviour.
        """
        threshold = self.motion_threshold
        if not isinstance(threshold, str) or _STATISTIC_THRESHOLDS[
            threshold.strip().lower()
        ] != "std":
            return None, self.resolve_threshold(scores)
        if not scores:
            centre = 0.0
            spread = 0.0
        else:
            centre = mean(scores)
            spread = stdev(scores) if len(scores) > 1 else 0.0
        lower = float(centre - self.motion_std_k * spread)
        upper = float(centre + self.motion_std_k * spread)
        direction = self.motion_std_direction.strip().lower()
        return (lower if direction in {"lower", "both"} else None,
                upper if direction in {"upper", "both"} else None)


@dataclass(frozen=True)
class DetectionConfig:
    """Parameters for object tracking within selected windows."""

    model_path: str = "yolo26n.pt"
    tracker: str = "bytetrack.yaml"
    confidence_threshold: float = 0.25
    spatial_grid: int = 3

    def __post_init__(self) -> None:
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.spatial_grid < 2:
            raise ValueError("spatial_grid must be at least 2")


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


@dataclass(frozen=True)
class MotionSample:
    frame_index: int
    timestamp_seconds: float
    score: float


@dataclass(frozen=True)
class RelevantWindow:
    start_seconds: float
    end_seconds: float
    start_frame: int
    end_frame: int
    sample_count: int
    peak_motion: float
    mean_motion: float
    is_relevant: bool


@dataclass(frozen=True)
class BoundingBox:
    """A pixel-space object boundary with a normalized representation."""

    left: float
    top: float
    right: float
    bottom: float

    def normalized(self, width: int, height: int) -> tuple[float, float, float, float]:
        return (
            round(self.left / width, 4),
            round(self.top / height, 4),
            round(self.right / width, 4),
            round(self.bottom / height, 4),
        )


@dataclass(frozen=True)
class ObjectDetection:
    track_id: int
    label: str
    confidence: float
    boundary: BoundingBox
    spatial_description: str


@dataclass(frozen=True)
class FrameAnnotations:
    frame_index: int
    timestamp_seconds: float
    detections: tuple[ObjectDetection, ...]


@dataclass(frozen=True)
class TrackedObject:
    track_id: int
    label: str
    first_frame: int
    last_frame: int
    first_timestamp_seconds: float
    last_timestamp_seconds: float
    average_confidence: float
    spatial_trajectory: tuple[str, ...]


@dataclass(frozen=True)
class ObjectWindowAnnotations:
    window: RelevantWindow
    objects: tuple[TrackedObject, ...]
    frames: tuple[FrameAnnotations, ...]


@dataclass(frozen=True)
class EventObject:
    object_id: str
    label: str
    physical_details: str


@dataclass(frozen=True)
class VideoEvent:
    name: str
    description: str
    involved_objects: tuple[EventObject, ...]


@dataclass(frozen=True)
class VideoAnnotations:
    video: VideoInfo
    windows: tuple[RelevantWindow, ...]
    object_annotations: tuple[ObjectWindowAnnotations, ...]
    events: tuple[VideoEvent, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable annotation document."""
        return asdict(self)


class EventBackend(Protocol):
    """Pluggable VLM or service used to identify semantic events."""

    def analyze(
        self,
        video_path: str,
        start_seconds: float,
        end_seconds: float,
        object_context: Sequence[TrackedObject],
    ) -> Sequence[dict[str, Any]]:
        """Return events in the documented JSON-compatible shape."""


def resolve_video_source(video_path: str | Path) -> Path:
    """Return a local video path, downloading HTTP(S) sources into a cache.

    VideoMeta delegates decoding to OpenCV and YOLO, neither of which reliably
    accepts every remote URL. Remote files are therefore cached by URL hash and
    reused by motion, spatial, and LVLM stages in the same or later runs.
    """
    source = str(video_path)
    parsed = urlparse(source)
    if parsed.scheme not in {"http", "https"}:
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")
        logger.info("Using local video source: %s", path)
        return path

    suffix = Path(parsed.path).suffix or ".mp4"
    cache_dir = Path(gettempdir()) / "videometa-videos"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{sha256(source.encode()).hexdigest()}{suffix}"
    if path.is_file() and path.stat().st_size > 0:
        logger.info("Using cached video URL: %s", path)
        return path

    temporary_path = path.with_suffix(f"{suffix}.part")
    logger.info("Downloading video URL to cache: %s", source)
    try:
        with urlopen(source, timeout=60) as response, temporary_path.open("wb") as output:
            copyfileobj(response, output)
        temporary_path.replace(path)
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        raise OSError(f"Could not download video URL: {source}") from error
    logger.info("Downloaded video URL to: %s", path)
    return path


class RelevantWindowFinder:
    """Scores sampled video frames and groups motion into overlapping windows."""

    def __init__(self, config: MotionGateConfig | None = None) -> None:
        self.config = config or MotionGateConfig()

    def probe(self, video_path: str | Path) -> VideoInfo:
        cv2 = _import_cv2()
        path = str(resolve_video_source(video_path))
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise OSError(f"Cannot open video: {path}")
        try:
            info = VideoInfo(
                path=path,
                fps=float(capture.get(cv2.CAP_PROP_FPS) or 30.0),
                frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            logger.info(
                "Probed video: %s (%d frames, %.2f FPS, %dx%d)",
                info.path,
                info.frame_count,
                info.fps,
                info.width,
                info.height,
            )
            return info
        finally:
            capture.release()

    def sample_motion(self, video_path: str | Path) -> tuple[VideoInfo, list[MotionSample]]:
        """Make one sequential pass through a video and score sampled frames."""
        cv2 = _import_cv2()
        info = self.probe(video_path)
        logger.info(
            "Sampling motion at %s FPS for %s",
            self.config.sample_fps or info.fps,
            info.path,
        )
        capture = cv2.VideoCapture(info.path)
        sample_fps = self.config.sample_fps or info.fps
        step = max(1, round(info.fps / sample_fps))
        warmup_frames = round(self.config.warmup_seconds * info.fps)
        subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.mog_history,
            varThreshold=self.config.mog_variance_threshold,
            detectShadows=False,
        )
        samples: list[MotionSample] = []
        frame_index = 0
        try:
            while capture.grab():
                if frame_index % step == 0:
                    ok, frame = capture.retrieve()
                    if ok:
                        gate = cv2.resize(frame, self.config.gate_size, interpolation=cv2.INTER_AREA)
                        score = float((subtractor.apply(gate) > 0).mean())
                        if frame_index < warmup_frames:
                            score = 0.0
                        samples.append(MotionSample(frame_index, frame_index / info.fps, score))
                frame_index += 1
        finally:
            capture.release()
        logger.info("Collected %d motion samples", len(samples))
        return info, samples

    def find(self, video_path: str | Path) -> tuple[VideoInfo, list[RelevantWindow]]:
        info, samples = self.sample_motion(video_path)
        return info, self.build_windows(samples, info.duration_seconds)

    def resolve_motion_threshold(self, samples: Sequence[MotionSample]) -> float:
        """Return the numeric motion cutoff for these samples."""
        return self.config.resolve_threshold([sample.score for sample in samples])

    def resolve_motion_thresholds(
        self, samples: Sequence[MotionSample]
    ) -> tuple[float | None, float | None]:
        """Return active ``(lower, upper)`` motion-score bounds for these samples."""
        return self.config.resolve_std_thresholds([sample.score for sample in samples])

    def build_windows(
        self, samples: Sequence[MotionSample], duration_seconds: float | None = None
    ) -> list[RelevantWindow]:
        """Build windows from supplied scores, useful for threshold experiments and tests."""
        if not samples:
            return []
        if duration_seconds is None:
            sample_interval = (
                samples[-1].timestamp_seconds - samples[-2].timestamp_seconds
                if len(samples) > 1
                else 0.0
            )
            duration = samples[-1].timestamp_seconds + max(0.0, sample_interval)
        else:
            duration = duration_seconds
        lower_threshold, upper_threshold = self.resolve_motion_thresholds(samples)
        windows: list[RelevantWindow] = []
        start = 0.0
        while start < duration:
            end = min(start + self.config.window_seconds, duration)
            included = [sample for sample in samples if start <= sample.timestamp_seconds < end]
            if included:
                scores = [sample.score for sample in included]
                peak = max(scores)
                is_relevant = (
                    (lower_threshold is not None and peak < lower_threshold)
                    or (upper_threshold is not None and peak > upper_threshold)
                )
                windows.append(
                    RelevantWindow(
                        start_seconds=round(start, 3),
                        end_seconds=round(end, 3),
                        start_frame=included[0].frame_index,
                        end_frame=included[-1].frame_index,
                        sample_count=len(included),
                        peak_motion=round(peak, 6),
                        mean_motion=round(sum(scores) / len(scores), 6),
                        is_relevant=is_relevant,
                    )
                )
            start += self.config.stride_seconds
        logger.info(
            "Built %d motion windows (%d relevant) with bounds (%s, %s)",
            len(windows),
            sum(window.is_relevant for window in windows),
            f"{lower_threshold:.6f}" if lower_threshold is not None else "none",
            f"{upper_threshold:.6f}" if upper_threshold is not None else "none",
        )
        return windows

    def calibrate(
        self, samples: Sequence[MotionSample], thresholds: Sequence[float]
    ) -> dict[float, list[RelevantWindow]]:
        """Evaluate several motion thresholds without re-decoding the video."""
        config = asdict(self.config)
        return {
            threshold: RelevantWindowFinder(
                MotionGateConfig(**{**config, "motion_threshold": threshold})
            ).build_windows(samples)
            for threshold in thresholds
        }


class ObjectBoundaryExtractor:
    """Tracks objects and emits pixel and normalized boundaries per relevant window."""

    def __init__(
        self,
        config: DetectionConfig | None = None,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.config = config or DetectionConfig()
        self._model_factory = model_factory

    def extract(
        self, video_path: str | Path, windows: Sequence[RelevantWindow], fps: float | None = None
    ) -> list[ObjectWindowAnnotations]:
        """Track the full video once, then assign global tracks to relevant windows."""
        cv2 = _import_cv2()
        source_path = resolve_video_source(video_path)
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise OSError(f"Cannot open video: {video_path}")
        source_fps = fps or float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        relevant_windows = [window for window in windows if window.is_relevant]
        logger.info(
            "Tracking full video for spatial features in %d relevant windows: %s",
            len(relevant_windows),
            source_path,
        )
        if not relevant_windows:
            return []

        model = self._new_model()
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise OSError(f"Cannot open video: {video_path}")
        registries: list[dict[int, dict[str, Any]]] = [
            defaultdict(
                lambda: {
                    "label": "",
                    "confidences": [],
                    "first": None,
                    "last": None,
                    "positions": [],
                }
            )
            for _ in relevant_windows
        ]
        frames_by_window: list[list[FrameAnnotations]] = [
            [] for _ in relevant_windows
        ]
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                detections = self._track_frame(model, frame, width, height)
                for index, window in enumerate(relevant_windows):
                    if window.start_frame <= frame_index <= window.end_frame:
                        self._record_detections(
                            registries[index], detections, frame_index
                        )
                        frames_by_window[index].append(
                            FrameAnnotations(
                                frame_index,
                                frame_index / source_fps,
                                tuple(detections),
                            )
                        )
                frame_index += 1
        finally:
            capture.release()

        annotations = [
            self._build_window_annotations(
                window, registry, frames, source_fps
            )
            for window, registry, frames in zip(
                relevant_windows, registries, frames_by_window
            )
        ]
        logger.info("Extracted spatial features for %d windows", len(annotations))
        return annotations

    @staticmethod
    def _record_detections(
        registry: dict[int, dict[str, Any]],
        detections: Sequence[ObjectDetection],
        frame_index: int,
    ) -> None:
        for detection in detections:
            item = registry[detection.track_id]
            item["label"] = detection.label
            item["confidences"].append(detection.confidence)
            item["first"] = frame_index if item["first"] is None else item["first"]
            item["last"] = frame_index
            if detection.spatial_description not in item["positions"]:
                item["positions"].append(detection.spatial_description)

    @staticmethod
    def _build_window_annotations(
        window: RelevantWindow,
        registry: dict[int, dict[str, Any]],
        frames: Sequence[FrameAnnotations],
        fps: float,
    ) -> ObjectWindowAnnotations:
        objects = tuple(
            TrackedObject(
                track_id=track_id,
                label=data["label"],
                first_frame=data["first"],
                last_frame=data["last"],
                first_timestamp_seconds=data["first"] / fps,
                last_timestamp_seconds=data["last"] / fps,
                average_confidence=round(sum(data["confidences"]) / len(data["confidences"]), 4),
                spatial_trajectory=tuple(data["positions"]),
            )
            for track_id, data in registry.items()
        )
        logger.info(
            "Assigned %d global tracks to %d frames in window %.3fs–%.3fs",
            len(objects),
            len(frames),
            window.start_seconds,
            window.end_seconds,
        )
        return ObjectWindowAnnotations(window, objects, tuple(frames))

    def _new_model(self) -> Any:
        if self._model_factory is not None:
            return self._model_factory(self.config.model_path)
        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except ImportError as error:
            raise ImportError(
                "Object extraction requires `pip install videometa[vision]` "
                "or a custom model_factory."
            ) from error
        return YOLO(self.config.model_path)

    def _track_frame(self, model: Any, frame: Any, width: int, height: int) -> list[ObjectDetection]:
        result = model.track(
            frame,
            persist=True,
            tracker=self.config.tracker,
            conf=self.config.confidence_threshold,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        output: list[ObjectDetection] = []
        for track_id, box, confidence, class_id in zip(
            boxes.id.int().cpu().tolist(),
            boxes.xyxy.cpu().tolist(),
            boxes.conf.cpu().tolist(),
            boxes.cls.int().cpu().tolist(),
        ):
            boundary = BoundingBox(*map(float, box))
            output.append(
                ObjectDetection(
                    track_id=int(track_id),
                    label=str(model.names[int(class_id)]),
                    confidence=round(float(confidence), 4),
                    boundary=boundary,
                    spatial_description=describe_spatial_position(
                        boundary, width, height, self.config.spatial_grid
                    ),
                )
            )
        return output


class EventExtractor:
    """Turns window video plus tracked-object context into structured events."""

    def __init__(self, backend: EventBackend) -> None:
        self.backend = backend

    def extract(
        self, video_path: str | Path, object_annotations: Sequence[ObjectWindowAnnotations]
    ) -> list[VideoEvent]:
        source_path = resolve_video_source(video_path)
        logger.info("Extracting semantic events from %d windows", len(object_annotations))
        events: list[VideoEvent] = []
        for annotation in object_annotations:
            analyze_window = getattr(self.backend, "analyze_window", None)
            if callable(analyze_window):
                raw_events = analyze_window(str(source_path), annotation)
            else:
                raw_events = self.backend.analyze(
                    str(source_path),
                    annotation.window.start_seconds,
                    annotation.window.end_seconds,
                    annotation.objects,
                )
            events.extend(_parse_events(raw_events))
        logger.info("Extracted %d semantic events", len(events))
        return events


class VideoAnnotator:
    """Convenience façade for motion gating, spatial boundaries, and semantic events."""

    def __init__(
        self,
        window_finder: RelevantWindowFinder | None = None,
        boundary_extractor: ObjectBoundaryExtractor | None = None,
        event_extractor: EventExtractor | None = None,
    ) -> None:
        self.window_finder = window_finder or RelevantWindowFinder()
        self.boundary_extractor = boundary_extractor or ObjectBoundaryExtractor()
        self.event_extractor = event_extractor

    def annotate(self, video_path: str | Path, include_events: bool = False) -> VideoAnnotations:
        video, windows = self.window_finder.find(video_path)
        object_annotations = self.boundary_extractor.extract(video_path, windows, video.fps)
        if include_events and self.event_extractor is None:
            raise ValueError("include_events=True requires an EventExtractor with an EventBackend")
        events = (
            self.event_extractor.extract(video_path, object_annotations)
            if include_events and self.event_extractor
            else []
        )
        return VideoAnnotations(video, tuple(windows), tuple(object_annotations), tuple(events))


def describe_spatial_position(
    boundary: BoundingBox, width: int, height: int, grid: int = 3
) -> str:
    """Describe the cell containing an object's bounding-box centre."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    x = max(0, min(grid - 1, int(((boundary.left + boundary.right) / 2 / width) * grid)))
    y = max(0, min(grid - 1, int(((boundary.top + boundary.bottom) / 2 / height) * grid)))
    horizontal = ("left", "center", "right") if grid == 3 else tuple(f"column-{i + 1}" for i in range(grid))
    vertical = ("top", "middle", "bottom") if grid == 3 else tuple(f"row-{i + 1}" for i in range(grid))
    return f"{vertical[y]}-{horizontal[x]}"


def _parse_events(raw_events: Sequence[dict[str, Any]]) -> list[VideoEvent]:
    parsed: list[VideoEvent] = []
    for event in raw_events:
        objects = tuple(
            EventObject(
                object_id=str(item.get("id", item.get("object_id", ""))),
                label=str(item.get("label", "")),
                physical_details=str(item.get("physical_details", "")),
            )
            for item in event.get("involved_objects", ())
        )
        parsed.append(
            VideoEvent(
                name=str(event.get("event_name", event.get("name", ""))),
                description=str(event.get("description", "")),
                involved_objects=objects,
            )
        )
    return parsed


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as error:
        raise ImportError("Video processing requires `pip install videometa[vision]`.") from error
    return cv2

"""Reusable video annotation pipeline.

Heavy integrations are imported only when they are used, keeping package import
lightweight and letting callers select their preferred detector and VLM backend.

Stage one, choosing which parts of a video are worth annotating, lives in
`videometa.activity_gate`: the video is cut into fixed overlapping chunks and a
vision-language model scores the activity in each one. This module holds the
shared data types, the YOLO boundary extractor, the event extractor and the
`VideoAnnotator` façade that strings the stages together.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import logging
from pathlib import Path
from shutil import copyfileobj
from statistics import mean
from tempfile import gettempdir
from typing import Any, Callable, Protocol, Sequence
from urllib.parse import urlparse
from urllib.request import urlopen

#: Slack in the identity-stitching distance test, as a fraction of the frame
#: diagonal, covering the jitter between a track's last box and the next one's
#: first box when nothing has actually moved.
_BOX_JITTER = 0.02


logger = logging.getLogger(__name__)
_PACKAGE_LOGGER = "videometa"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def set_verbose(enabled: bool = True) -> None:
    """Show (or hide) videometa's execution details on stderr.

    Verbose output covers each stage: the video probed, the chunks scored and
    kept by the activity gate, the device YOLO runs on, tracks stitched, and
    events extracted. Applications that configure `logging` themselves need not
    call this; the package only logs through the ``videometa`` logger.
    """
    package_logger = logging.getLogger(_PACKAGE_LOGGER)
    for handler in [h for h in package_logger.handlers if getattr(h, "_videometa_verbose", False)]:
        package_logger.removeHandler(handler)
    if not enabled:
        package_logger.setLevel(logging.WARNING)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler._videometa_verbose = True  # type: ignore[attr-defined]
    package_logger.addHandler(handler)
    package_logger.setLevel(logging.INFO)


def _dominant(votes: dict[str, float]) -> str:
    """The highest-scoring key, ties broken by name so the result is stable."""
    return max(sorted(votes), key=lambda key: votes[key]) if votes else ""


def _groups(identity: dict[int, int]) -> dict[int, list[int]]:
    """Invert a track-to-identity map into identity-to-tracks."""
    grouped: dict[int, list[int]] = defaultdict(list)
    for track_id, root in identity.items():
        grouped[root].append(track_id)
    return grouped


def _add_vectors(first: Sequence[float], second: Sequence[float]) -> list[float]:
    """Element-wise sum, spelled out so it does not depend on numpy's operators."""
    return [float(value) + float(other) for value, other in zip(first, second)]


def _correlation(first: Sequence[float], second: Sequence[float]) -> float:
    """Pearson correlation between two histograms, as OpenCV's HISTCMP_CORREL.

    Written out rather than delegated so identity stitching stays testable
    without decoding a video, and because being scale invariant it lets the
    caller compare accumulated histograms without averaging them first.
    """
    left = [float(value) for value in first]
    right = [float(value) for value in second]
    if len(left) != len(right) or not left:
        return 0.0
    left_mean, right_mean = mean(left), mean(right)
    covariance = sum(
        (value - left_mean) * (other - right_mean) for value, other in zip(left, right)
    )
    left_spread = sum((value - left_mean) ** 2 for value in left)
    right_spread = sum((other - right_mean) ** 2 for other in right)
    if left_spread <= 0 or right_spread <= 0:
        return 0.0
    return covariance / ((left_spread * right_spread) ** 0.5)


def _centre_distance(first: BoundingBox, second: BoundingBox) -> float:
    """Pixel distance between two boxes' centres."""
    first_x = (first.left + first.right) / 2
    first_y = (first.top + first.bottom) / 2
    second_x = (second.left + second.right) / 2
    second_y = (second.top + second.bottom) / 2
    return ((first_x - second_x) ** 2 + (first_y - second_y) ** 2) ** 0.5


@dataclass(frozen=True)
class DetectionConfig:
    """Parameters for object tracking within selected windows.

    `classes` is an optional sequence of YOLO class IDs to track. When omitted
    or empty, every class known to the loaded YOLO model is used.

    `device` is the torch device YOLO runs on (``"cuda"``, ``"mps"``, ``"cpu"``,
    ``"cuda:1"``...). When omitted it is auto-detected: CUDA if available, then
    Apple MPS, then CPU.

    **Identity across the video.** The tracker runs once over the whole video,
    so an object keeps its id while it is continuously visible. It cannot keep
    it across a disappearance: ByteTrack matches on position, and once a track
    has been missing longer than the tracker's own buffer it is retired, so the
    same person walking back into shot returns as a new id.

    With `stitch_tracks` those fragments are rejoined after the pass. Two tracks
    are treated as the same object when they never appear at the same moment,
    carry the same label, are separated by at most `stitch_max_gap_seconds`,
    could plausibly have travelled between their last and first positions at
    `stitch_max_speed` (in frame diagonals per second), and their colour
    signatures match to at least `stitch_min_similarity`.

    The defaults allow a tenth of a frame diagonal of travel per second — a
    brisk walk at surveillance framing — and ask for a 0.7 colour correlation.
    On a 60s clip of a car park that rejoined 21 of 23 flickering detections
    while admitting one implausible jump; the looser 0.5/0.5 it replaced
    admitted five, and tightening to 0.85 similarity cost five real rejoins to
    remove the last bad one.

    That last test is a colour histogram over `appearance_samples` crops, not a
    learned re-identification model: it is well suited to a fixed camera and
    distinguishable clothing, and it will confuse two people in similar dark
    coats. Raise `stitch_min_similarity` to merge less, set `stitch_tracks=False`
    to keep the raw tracker ids.
    """

    model_path: str = "yolo26n.pt"
    tracker: str = "bytetrack.yaml"
    confidence_threshold: float = 0.25
    spatial_grid: int = 3
    classes: Sequence[int] | None = None
    device: str | None = None

    # --- identity consistency across the whole video
    stitch_tracks: bool = True
    stitch_max_gap_seconds: float = 30.0
    stitch_min_similarity: float = 0.7
    stitch_max_speed: float = 0.1
    appearance_samples: int = 12
    appearance_bins: tuple[int, int] = (16, 8)

    def __post_init__(self) -> None:
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.spatial_grid < 2:
            raise ValueError("spatial_grid must be at least 2")
        if self.stitch_max_gap_seconds < 0:
            raise ValueError("stitch_max_gap_seconds cannot be negative")
        if not 0 <= self.stitch_min_similarity <= 1:
            raise ValueError("stitch_min_similarity must be within [0, 1]")
        if self.stitch_max_speed <= 0:
            raise ValueError("stitch_max_speed must be positive")
        if self.appearance_samples < 1:
            raise ValueError("appearance_samples must be at least 1")
        if any(bins < 1 for bins in self.appearance_bins):
            raise ValueError("appearance_bins must be positive")
        if self.classes is None:
            return
        try:
            resolved = tuple(dict.fromkeys(int(class_id) for class_id in self.classes))
        except (TypeError, ValueError) as error:
            raise ValueError("classes must be a sequence of non-negative integer YOLO class IDs") from error
        if any(class_id < 0 for class_id in resolved):
            raise ValueError("classes must be a sequence of non-negative integer YOLO class IDs")
        object.__setattr__(self, "classes", resolved or None)

    def resolve_classes(self, model: Any | None = None) -> list[int] | None:
        """Return class IDs to pass to YOLO, or ``None`` to use every model class."""
        if self.classes:
            return list(self.classes)
        names = getattr(model, "names", None)
        if isinstance(names, dict) and names:
            return sorted(int(class_id) for class_id in names)
        if isinstance(names, (list, tuple)) and names:
            return list(range(len(names)))
        return None


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
class RelevantWindow:
    """One fixed chunk of the video and the activity score the VLM gave it.

    `score` is in [0, 1]: how much is happening in the chunk, judged by the
    vision-language model from the number of people and vehicles, how much
    they move, and how many distinct actions or interactions occur (see
    `videometa.activity_gate.ACTIVITY_SCORE_GUIDANCE`). `is_relevant` is that
    score tested against the gate's threshold and budget. `subjects`,
    `event_count` and `summary` are the model's short justification, kept so a
    reviewer can see why a chunk was kept or dropped. `clip_path` is the
    downscaled MP4 the model watched, and `error` records a scoring call that
    failed, in which case `score` is 0 and the chunk is never relevant.
    """

    start_seconds: float
    end_seconds: float
    start_frame: int
    end_frame: int
    score: float
    is_relevant: bool
    subjects: tuple[str, ...] = ()
    event_count: int = 0
    summary: str = ""
    clip_path: str | None = None
    error: str | None = None

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)


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
    reused by the gate, spatial, and LVLM stages in the same or later runs.
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


def probe_video(video_path: str | Path) -> VideoInfo:
    """Read a video's frame rate, frame count and size without decoding it."""
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


class ObjectBoundaryExtractor:
    """Tracks objects and emits pixel and normalized boundaries per relevant window."""

    def __init__(
        self,
        config: DetectionConfig | None = None,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.config = config or DetectionConfig()
        self._model_factory = model_factory
        self._device: str | None = None

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
        self._device = resolve_device(self.config.device)
        logger.info("Running YOLO model %s on device: %s", self.config.model_path, self._device)
        classes = self.config.resolve_classes(model)
        if classes is None:
            logger.info("Tracking all available YOLO classes")
        else:
            logger.info("Tracking %d YOLO classes: %s", len(classes), classes)
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise OSError(f"Cannot open video: {video_path}")
        profiles: dict[int, dict[str, Any]] = {}
        frames_by_window: list[list[FrameAnnotations]] = [[] for _ in relevant_windows]
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                detections = self._track_frame(model, frame, width, height)
                # Profiles are built from the whole video, not only the selected
                # windows, so a track seen between two windows still anchors the
                # identity that links them.
                for detection in detections:
                    self._profile_detection(cv2, profiles, detection, frame, frame_index)
                for index, window in enumerate(relevant_windows):
                    if window.start_frame <= frame_index <= window.end_frame:
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

        identity = self._stitch_tracks(profiles, source_fps, (width ** 2 + height ** 2) ** 0.5)
        labels = {
            track_id: self._settled_label(profiles, group)
            for track_id, group in _groups(identity).items()
        }
        frames_by_window = [
            [self._relabel_frame(frame, identity, labels) for frame in frames]
            for frames in frames_by_window
        ]
        annotations = [
            self._build_window_annotations(window, self._registry_for(frames), frames, source_fps)
            for window, frames in zip(relevant_windows, frames_by_window)
        ]
        logger.info("Extracted spatial features for %d windows", len(annotations))
        return annotations

    def _profile_detection(
        self,
        cv2: Any,
        profiles: dict[int, dict[str, Any]],
        detection: ObjectDetection,
        frame: Any,
        frame_index: int,
    ) -> None:
        """Accumulate what identity stitching needs, one detection at a time."""
        profile = profiles.get(detection.track_id)
        if profile is None:
            profile = profiles[detection.track_id] = {
                "labels": defaultdict(float),
                "first": frame_index,
                "last": frame_index,
                "first_box": detection.boundary,
                "last_box": detection.boundary,
                "appearance": None,
                "samples": 0,
            }
        # A class can flicker between frames, so the label is a vote weighted by
        # confidence rather than whatever the last frame happened to say.
        profile["labels"][detection.label] += detection.confidence
        profile["last"] = frame_index
        profile["last_box"] = detection.boundary
        if self.config.stitch_tracks and profile["samples"] < self.config.appearance_samples:
            signature = self._appearance(cv2, frame, detection.boundary)
            if signature is not None:
                profile["appearance"] = (
                    signature
                    if profile["appearance"] is None
                    else _add_vectors(profile["appearance"], signature)
                )
                profile["samples"] += 1

    def _appearance(self, cv2: Any, frame: Any, boundary: BoundingBox) -> Any:
        """Normalised hue/saturation histogram of one detection's crop."""
        height, width = frame.shape[:2]
        left, top = max(0, int(boundary.left)), max(0, int(boundary.top))
        right, bottom = min(width, int(boundary.right)), min(height, int(boundary.bottom))
        if right - left < 2 or bottom - top < 2:
            return None
        crop = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist(
            [crop], [0, 1], None, list(self.config.appearance_bins), [0, 180, 0, 256]
        )
        return [float(value) for value in cv2.normalize(histogram, histogram).flatten()]

    def _stitch_tracks(
        self, profiles: dict[int, dict[str, Any]], fps: float, diagonal: float
    ) -> dict[int, int]:
        """Map every raw tracker id to a stable identity for the whole video.

        Each track is offered to the clusters that finished before it began.
        Overlapping tracks can never merge — two things visible at once are two
        things — and the winner is the closest colour match that also passes the
        label, gap and travel-distance tests.
        """
        identity = {track_id: track_id for track_id in profiles}
        if not self.config.stitch_tracks or len(profiles) < 2:
            return identity
        order = sorted(profiles, key=lambda track_id: profiles[track_id]["first"])
        clusters = {
            track_id: {**profiles[track_id], "labels": dict(profiles[track_id]["labels"])}
            for track_id in order
        }
        merges = 0
        for track_id in order:
            profile = profiles[track_id]
            if profile["appearance"] is None:
                continue
            best_root, best_score = None, self.config.stitch_min_similarity
            for root, cluster in clusters.items():
                if root == track_id or cluster["last"] >= profile["first"]:
                    continue
                gap_seconds = (profile["first"] - cluster["last"]) / fps if fps else 0.0
                if gap_seconds > self.config.stitch_max_gap_seconds:
                    continue
                if _dominant(cluster["labels"]) != _dominant(profile["labels"]):
                    continue
                # How far the object could have travelled in the gap, plus a
                # little slack for box jitter. Scaling strictly with the gap is
                # what rejects a "reappearance" 200px away 0.1s later, which is
                # a different object moving at an impossible speed.
                reach = diagonal * (self.config.stitch_max_speed * gap_seconds + _BOX_JITTER)
                if _centre_distance(cluster["last_box"], profile["first_box"]) > reach:
                    continue
                if cluster["appearance"] is None:
                    continue
                # Correlation is scale invariant, so the accumulated histograms
                # can be compared without dividing by their sample counts.
                score = _correlation(cluster["appearance"], profile["appearance"])
                if score > best_score:
                    best_root, best_score = root, score
            if best_root is None:
                continue
            cluster = clusters.pop(track_id)
            winner = clusters[best_root]
            winner["last"] = cluster["last"]
            winner["last_box"] = cluster["last_box"]
            winner["samples"] += cluster["samples"]
            winner["appearance"] = _add_vectors(winner["appearance"], cluster["appearance"])
            for label, weight in cluster["labels"].items():
                # a plain dict, so a label the winner has not seen must be seeded
                winner["labels"][label] = winner["labels"].get(label, 0.0) + weight
            identity[track_id] = best_root
            merges += 1
        # collapse chains, so a track merged into a track points at the survivor
        for track_id in order:
            root = identity[track_id]
            while identity[root] != root:
                root = identity[root]
            identity[track_id] = root
        logger.info(
            "Stitched %d of %d tracker ids into %d identities",
            merges,
            len(profiles),
            len(set(identity.values())),
        )
        return identity

    @staticmethod
    def _settled_label(profiles: dict[int, dict[str, Any]], group: Sequence[int]) -> str:
        """One label per identity: the class with the most confidence behind it."""
        votes: dict[str, float] = defaultdict(float)
        for track_id in group:
            for label, weight in profiles[track_id]["labels"].items():
                votes[label] += weight
        return _dominant(votes)

    @staticmethod
    def _relabel_frame(
        frame: FrameAnnotations, identity: dict[int, int], labels: dict[int, str]
    ) -> FrameAnnotations:
        """Rewrite a frame's detections with their stable id and settled label."""
        return replace(
            frame,
            detections=tuple(
                replace(
                    detection,
                    track_id=identity.get(detection.track_id, detection.track_id),
                    label=labels.get(
                        identity.get(detection.track_id, detection.track_id), detection.label
                    ),
                )
                for detection in frame.detections
            ),
        )

    @staticmethod
    def _registry_for(frames: Sequence[FrameAnnotations]) -> dict[int, dict[str, Any]]:
        """Summarise a window's already-relabelled frames, one entry per identity."""
        registry: dict[int, dict[str, Any]] = {}
        for frame in frames:
            for detection in frame.detections:
                item = registry.get(detection.track_id)
                if item is None:
                    item = registry[detection.track_id] = {
                        "label": detection.label,
                        "confidences": [],
                        "first": frame.frame_index,
                        "last": frame.frame_index,
                        "positions": [],
                    }
                item["confidences"].append(detection.confidence)
                item["last"] = frame.frame_index
                if detection.spatial_description not in item["positions"]:
                    item["positions"].append(detection.spatial_description)
        return registry

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
        track_kwargs: dict[str, Any] = {
            "persist": True,
            "tracker": self.config.tracker,
            "conf": self.config.confidence_threshold,
            "verbose": False,
        }
        if self._device is not None:
            track_kwargs["device"] = self._device
        classes = self.config.resolve_classes(model)
        if classes is not None:
            track_kwargs["classes"] = classes
        result = model.track(frame, **track_kwargs)[0]
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


class WindowFinder(Protocol):
    """Anything that can pick the windows of a video worth annotating."""

    def find(self, video_path: str | Path) -> tuple[VideoInfo, list[RelevantWindow]]:
        """Return the probed video and every window, relevant ones flagged."""


class VideoAnnotator:
    """Convenience façade for activity gating, spatial boundaries, and semantic events."""

    def __init__(
        self,
        window_finder: WindowFinder,
        boundary_extractor: ObjectBoundaryExtractor | None = None,
        event_extractor: EventExtractor | None = None,
        verbose: bool = False,
    ) -> None:
        """`window_finder` is normally an `ActivityWindowFinder` from
        `videometa.activity_gate`, built around the same VLM that later writes
        the event annotations."""
        if verbose:
            set_verbose(True)
        self.window_finder = window_finder
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


def resolve_device(requested: str | None = None) -> str:
    """Return the torch device to run YOLO on: `requested`, else CUDA, MPS, then CPU."""
    if requested:
        logger.info("Device: %s (requested)", requested)
        return requested
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        logger.info("Device: cpu (torch not installed)")
        return "cpu"
    if torch.cuda.is_available():
        logger.info("Device: cuda (auto-detected: %s)", torch.cuda.get_device_name(0))
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        logger.info("Device: mps (auto-detected: Apple Silicon GPU)")
        return "mps"
    logger.info("Device: cpu (no CUDA or MPS available)")
    return "cpu"


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as error:
        raise ImportError("Video processing requires `pip install videometa[vision]`.") from error
    return cv2

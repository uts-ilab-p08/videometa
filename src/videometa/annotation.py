"""Reusable video annotation pipeline.

Heavy integrations are imported only when they are used, keeping package import
lightweight and letting callers select their preferred detector and VLM backend.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
from itertools import zip_longest
import logging
from pathlib import Path
from shutil import copyfileobj
from math import ceil
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
    "mad": "mad",
    "median+mad": "mad",
    "auto": "auto",
}
_THRESHOLD_CHOICES = "'avg'/'median'/'std'/'mad'/'auto'"
_STD_DIRECTIONS = {"upper", "lower", "both"}

#: Per-sample motion metrics. See `MotionGateConfig.motion_metric`.
MOTION_METRICS = ("score", "tile_peak", "blob_area", "local")

#: Smallest cutoff each metric may fall to when `motion_threshold="auto"`, in
#: that metric's own units. These are physical floors, not tuned constants: a
#: gate that drops below them is reacting to sensor noise rather than motion.
_METRIC_FLOORS = {
    "score": 0.002,        # fraction of the whole gate frame
    "tile_peak": 0.05,     # fraction of one tile
    "blob_area": 40.0,     # pixels in the largest blob, on the gate frame
    "local": 3.0,          # multiples of the tile's own spread
}
_SEGMENTATIONS = {"events", "grid"}

#: Slack in the identity-stitching distance test, as a fraction of the frame
#: diagonal, covering the jitter between a track's last box and the next one's
#: first box when nothing has actually moved.
_BOX_JITTER = 0.02


logger = logging.getLogger(__name__)


def _sample_interval(samples: Sequence[MotionSample]) -> float:
    """Typical seconds between samples, from the median gap rather than the last one."""
    if len(samples) < 2:
        return 0.0
    gaps = [
        later.timestamp_seconds - earlier.timestamp_seconds
        for earlier, later in zip(samples, samples[1:])
    ]
    return max(0.0, float(median(gaps)))


def _frames_per_second(samples: Sequence[MotionSample]) -> float:
    """Recover the source frame rate from the samples' own frame/time pairing."""
    if len(samples) < 2:
        return 0.0
    span_seconds = samples[-1].timestamp_seconds - samples[0].timestamp_seconds
    span_frames = samples[-1].frame_index - samples[0].frame_index
    return span_frames / span_seconds if span_seconds > 0 else 0.0


def _rolling_median(values: Sequence[float], width: int) -> list[float]:
    """Median filter, which removes lone spikes without blunting a short event.

    A mean would smear a two-sample event across its neighbours and lower its
    peak; a median leaves any run longer than `width`//2 exactly where it is.
    """
    if width <= 1 or len(values) <= width:
        return list(values)
    half = width // 2
    return [
        float(median(values[max(0, index - half):index + half + 1]))
        for index in range(len(values))
    ]


def _merge_spans(
    spans: Sequence[tuple[float, float, float]], gap: float
) -> list[tuple[float, float, float]]:
    """Merge spans separated by less than `gap` seconds, summing their active time."""
    merged: list[tuple[float, float, float]] = []
    for start, end, active in sorted(spans):
        if merged and start - merged[-1][1] <= gap:
            previous_start, previous_end, previous_active = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end), previous_active + active)
        else:
            merged.append((start, end, active))
    return merged


def _at_least(
    span: tuple[float, float, float], minimum: float, duration: float
) -> tuple[float, float, float]:
    """Grow a span symmetrically to `minimum` seconds, staying inside the video.

    Short events are widened rather than dropped: a 0.4s handoff is the kind of
    thing the gate exists to find, and a window has to be long enough for the
    stage behind it to make sense of what it is looking at.
    """
    start, end, active = span
    shortfall = minimum - (end - start)
    if shortfall <= 0:
        return span
    start = max(0.0, start - shortfall / 2)
    end = min(duration, max(end, start + minimum))
    return (max(0.0, min(start, max(0.0, end - minimum))), end, active)


def _split_span(
    span: tuple[float, float, float], maximum: float
) -> list[tuple[float, float, float]]:
    """Cut an over-long span into equal chunks, so no window outgrows its budget."""
    start, end, active = span
    length = end - start
    if length <= maximum:
        return [span]
    parts = ceil(length / maximum)
    size = length / parts
    return [
        (start + index * size, start + (index + 1) * size, active / parts)
        for index in range(parts)
    ]


def _interleave_by_region(
    windows: Sequence[RelevantWindow], tile_grid: tuple[int, int], grid: int
) -> list[RelevantWindow]:
    """Reorder a ranked list so each region's best window comes before any seconds."""
    rows, columns = tile_grid
    groups: dict[tuple[int, int], list[RelevantWindow]] = defaultdict(list)
    for window in windows:
        if window.focus is None:
            region = (-1, -1)
        else:
            row, column = window.focus
            region = (
                min(grid - 1, int(row * grid / max(1, rows))),
                min(grid - 1, int(column * grid / max(1, columns))),
            )
        groups[region].append(window)
    tiers = sorted(groups.values(), key=lambda group: -group[0].relevance)
    return [window for tier in zip_longest(*tiers) for window in tier if window is not None]


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


def _median_and_mad(values: Sequence[float]) -> tuple[float, float]:
    """Median and a standard-deviation-equivalent MAD.

    The 1.4826 factor makes the MAD comparable to a standard deviation for
    normal data, so `motion_std_k` means the same thing for ``"std"`` and
    ``"mad"``. Unlike a standard deviation, it is not dragged upwards by the
    handful of very loud samples that motion scores always contain — which is
    exactly the failure that lets a ``"std"`` threshold sail over every quiet
    event in a video that also holds one bus.
    """
    if not values:
        return 0.0, 0.0
    centre = float(median(values))
    return centre, float(median([abs(value - centre) for value in values]) * 1.4826)


@dataclass(frozen=True)
class MotionGateConfig:
    """Parameters controlling motion-based relevant-window selection.

    **What is measured.** `motion_metric` selects the per-sample number the
    gate thresholds:

    - ``"score"``   — fraction of the whole gate frame flagged as foreground.
      The original metric; it averages a small object away (a 20x20 object on a
      640x360 gate is 0.17% of the frame).
    - ``"tile_peak"`` — the loudest cell of a `tile_grid` grid, as a fraction of
      that cell. Size-relative to a tile instead of the frame, but it saturates
      once one object fills a cell.
    - ``"blob_area"`` — pixels in the largest connected foreground region.
    - ``"local"`` (default) — the loudest cell measured **against that cell's
      own history**: ``(activity - median) / spread`` in units of the cell's own
      spread. A distant person in a quiet corner competes with that corner's
      normal, not with a bus in the foreground, which is what lets small,
      localised motion survive a global cutoff.

    **How it is cut.** `motion_threshold` is a non-negative number or a
    statistic name: ``"avg"`` / ``"median"`` / ``"std"`` / ``"mad"`` /
    ``"auto"``. ``"std"`` uses one or both of ``mean ± motion_std_k * std``,
    controlled by `motion_std_direction`; ``"mad"`` is its outlier-resistant
    twin, ``median + motion_std_k * MAD``. ``"auto"`` (default) is ``"mad"``
    held above a physical floor for the chosen metric, so a video with no motion
    at all does not calibrate its way down into sensor noise. Statistics ignore
    warmup samples, which are zero by construction.

    **How windows are cut.** `segmentation="events"` (default) grows windows
    around runs of motion — hysteresis, padding, merging — so a window's extent
    reflects the motion in it. `segmentation="grid"` restores the original
    fixed `window_seconds`/`stride_seconds` tiling.

    **How windows are filtered.** Every detected window is returned; `max_windows`
    and `max_total_seconds` decide how many are marked `is_relevant`. With
    `spatial_diversity` the budget is spread over distinct regions of the frame
    before it is spent on the loudest region twice.
    """

    sample_fps: float | None = None
    gate_size: tuple[int, int] = (640, 360)
    window_seconds: float = 60.0
    stride_seconds: float = 50.0
    motion_threshold: float | str = "auto"
    motion_std_k: float = 1.5
    motion_std_direction: str = "upper"
    warmup_seconds: float = 2.0
    mog_history: int = 300
    mog_variance_threshold: float = 24.0

    # --- measurement
    motion_metric: str = "local"
    tile_grid: tuple[int, int] = (9, 16)
    tile_floor: float = 0.01
    tile_baseline_quantile: float = 0.9
    diff_levels: int = 12

    # --- segmentation
    segmentation: str = "events"
    hysteresis_ratio: float = 0.5
    smooth_samples: int = 3
    pad_seconds: float = 2.0
    merge_gap_seconds: float = 3.0
    min_window_seconds: float = 2.0
    max_window_seconds: float = 60.0

    # --- selection
    max_windows: int | None = None
    max_total_seconds: float | None = None
    spatial_diversity: bool = False
    diversity_grid: int = 3

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
        if self.motion_metric not in MOTION_METRICS:
            raise ValueError(f"motion_metric must be one of {MOTION_METRICS}")
        if self.segmentation not in _SEGMENTATIONS:
            raise ValueError("segmentation must be 'events' or 'grid'")
        rows, columns = self.tile_grid
        if rows < 1 or columns < 1:
            raise ValueError("tile_grid must be at least 1x1")
        if not 0 < self.tile_floor <= 1:
            raise ValueError("tile_floor must be within (0, 1]")
        if not 0 < self.tile_baseline_quantile < 1:
            raise ValueError("tile_baseline_quantile must be within (0, 1)")
        if not 0 <= self.diff_levels <= 255:
            raise ValueError("diff_levels must be within [0, 255]")
        if not 0 <= self.hysteresis_ratio <= 1:
            raise ValueError("hysteresis_ratio must be within [0, 1]")
        if self.smooth_samples < 1:
            raise ValueError("smooth_samples must be at least 1")
        if self.pad_seconds < 0 or self.merge_gap_seconds < 0:
            raise ValueError("pad_seconds and merge_gap_seconds cannot be negative")
        if self.min_window_seconds < 0:
            raise ValueError("min_window_seconds cannot be negative")
        if self.max_window_seconds <= 0:
            raise ValueError("max_window_seconds must be positive")
        if self.min_window_seconds > self.max_window_seconds:
            raise ValueError("min_window_seconds cannot exceed max_window_seconds")
        if self.max_windows is not None and self.max_windows < 0:
            raise ValueError("max_windows cannot be negative")
        if self.max_total_seconds is not None and self.max_total_seconds < 0:
            raise ValueError("max_total_seconds cannot be negative")
        if self.diversity_grid < 1:
            raise ValueError("diversity_grid must be at least 1")
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

    def metric_floor(self) -> float:
        """Smallest cutoff `"auto"` will use, in the active metric's own units."""
        return _METRIC_FLOORS[self.motion_metric]

    def resolve_threshold(self, scores: Sequence[float]) -> float:
        """Return the upper numeric cutoff, computing a statistic when requested.

        For a ``"std"`` threshold, use `resolve_std_thresholds()` to obtain
        both bounds when `motion_std_direction` is ``"lower"`` or ``"both"``.
        """
        threshold = self.motion_threshold
        if isinstance(threshold, (int, float)) and not isinstance(threshold, bool):
            return float(threshold)
        mode = _STATISTIC_THRESHOLDS[str(threshold).strip().lower()]
        if not scores:
            return self.metric_floor() if mode == "auto" else 0.0
        if mode == "avg":
            return float(mean(scores))
        if mode == "median":
            return float(median(scores))
        if mode == "std":
            spread = stdev(scores) if len(scores) > 1 else 0.0
            return float(mean(scores) + self.motion_std_k * spread)
        centre, spread = _median_and_mad(scores)
        cutoff = centre + self.motion_std_k * spread
        return float(max(cutoff, self.metric_floor()) if mode == "auto" else cutoff)

    def resolve_std_thresholds(self, scores: Sequence[float]) -> tuple[float | None, float | None]:
        """Return the active (lower, upper) standard-deviation bounds.

        Non-``"std"`` thresholds return ``(None, resolved_threshold)`` so
        callers can consistently use the upper bound for existing behaviour.
        """
        threshold = self.motion_threshold
        mode = (
            _STATISTIC_THRESHOLDS[threshold.strip().lower()]
            if isinstance(threshold, str)
            else None
        )
        if mode not in {"std", "mad"}:
            return None, self.resolve_threshold(scores)
        if not scores:
            centre = 0.0
            spread = 0.0
        elif mode == "mad":
            centre, spread = _median_and_mad(scores)
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
    """Parameters for object tracking within selected windows.

    `classes` is an optional sequence of YOLO class IDs to track. When omitted
    or empty, every class known to the loaded YOLO model is used.

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
class MotionSample:
    """One sampled frame, measured four ways from the same foreground mask.

    `score` is the original whole-frame foreground fraction and is left
    untouched so existing thresholds keep their meaning. The rest are computed
    in the same pass; see `MotionGateConfig.motion_metric` for what each one is
    good for. `focus` is the (row, column) of the loudest tile, which is where
    the motion was — used to spread the window budget across the frame.
    """

    frame_index: int
    timestamp_seconds: float
    score: float
    tile_peak: float = 0.0
    blob_area: float = 0.0
    local_score: float = 0.0
    focus: tuple[int, int] | None = None
    is_warmup: bool = False

    def metric(self, name: str) -> float:
        """Return this sample's value for one of `MOTION_METRICS`."""
        if name == "score":
            return self.score
        if name == "tile_peak":
            return self.tile_peak
        if name == "blob_area":
            return self.blob_area
        if name == "local":
            return self.local_score
        raise ValueError(f"unknown motion metric {name!r}; expected one of {MOTION_METRICS}")


@dataclass(frozen=True)
class RelevantWindow:
    """A span of video and the motion evidence that produced it.

    `peak_motion` / `mean_motion` stay in the units of the configured
    `motion_metric`. `focus` names the dominant region, and `relevance` is the
    value windows are ranked by when a budget has to be spent.
    """

    start_seconds: float
    end_seconds: float
    start_frame: int
    end_frame: int
    sample_count: int
    peak_motion: float
    mean_motion: float
    is_relevant: bool
    motion_metric: str = "score"
    relevance: float = 0.0
    focus: tuple[int, int] | None = None
    active_seconds: float = 0.0

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
        """Make one sequential pass through a video and measure sampled frames.

        Two detectors are read from every sampled frame and fused:

        - **MOG2** answers "does this pixel differ from what normally stands
          here", which holds an object up for as long as it is unexpected.
        - **Frame differencing** answers "did this pixel just change", which
          survives MOG2's blind spot: MOG2 absorbs a stationary object into its
          own background within roughly `mog_history`/10 samples, after which a
          person standing still and handling something registers as nothing.

        Per-tile activity is accumulated during the pass and normalised against
        each tile's own history afterwards, which is what `local` scoring needs
        and the reason this method has to see the whole video before it can
        finish a sample.
        """
        cv2 = _import_cv2()
        numpy = _import_numpy()
        config = self.config
        info = self.probe(video_path)
        logger.info(
            "Sampling motion at %s FPS for %s (metric=%s)",
            config.sample_fps or info.fps,
            info.path,
            config.motion_metric,
        )
        capture = cv2.VideoCapture(info.path)
        sample_fps = config.sample_fps or info.fps
        step = max(1, round(info.fps / sample_fps))
        warmup_frames = round(config.warmup_seconds * info.fps)
        subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.mog_history,
            varThreshold=config.mog_variance_threshold,
            detectShadows=False,
        )
        gate_width, gate_height = config.gate_size
        rows = max(1, min(config.tile_grid[0], gate_height))
        columns = max(1, min(config.tile_grid[1], gate_width))
        tile_height, tile_width = gate_height // rows, gate_width // columns
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        measurements: list[tuple[int, float, float, float, tuple[int, int], bool]] = []
        tile_history: list[Any] = []
        previous_gray = None
        frame_index = 0
        try:
            while capture.grab():
                if frame_index % step == 0:
                    ok, frame = capture.retrieve()
                    if ok:
                        gate = cv2.resize(frame, config.gate_size, interpolation=cv2.INTER_AREA)
                        gray = cv2.cvtColor(gate, cv2.COLOR_BGR2GRAY)
                        foreground = subtractor.apply(gate) > 0
                        if previous_gray is None:
                            changed = numpy.zeros_like(foreground)
                        else:
                            changed = cv2.absdiff(gray, previous_gray) > config.diff_levels
                        previous_gray = gray
                        fused = foreground | changed

                        # Tiles are read from the raw fused mask: opening it
                        # first would erase the few-pixel objects this metric
                        # exists to catch. Scattered noise is handled instead by
                        # the per-tile baseline, which rises to meet it.
                        trimmed = fused[: rows * tile_height, : columns * tile_width]
                        tiles = trimmed.reshape(
                            rows, tile_height, columns, tile_width
                        ).mean(axis=(1, 3))
                        hottest = int(tiles.argmax())
                        # The blob metric is the one place an opening belongs:
                        # a "largest connected region" is meaningless if noise
                        # specks are allowed to count as regions.
                        opened = cv2.morphologyEx(
                            fused.astype(numpy.uint8), cv2.MORPH_OPEN, kernel
                        )
                        count, _, stats, _ = cv2.connectedComponentsWithStats(
                            opened, connectivity=8
                        )
                        blob_area = (
                            0.0
                            if count <= 1
                            else float(stats[1:, cv2.CC_STAT_AREA].max())
                        )
                        is_warmup = frame_index < warmup_frames
                        measurements.append((
                            frame_index,
                            0.0 if is_warmup else float(foreground.mean()),
                            0.0 if is_warmup else float(tiles.max()),
                            0.0 if is_warmup else blob_area,
                            (hottest // columns, hottest % columns),
                            is_warmup,
                        ))
                        tile_history.append(
                            numpy.zeros(rows * columns, numpy.float32)
                            if is_warmup
                            else tiles.ravel()
                        )
                frame_index += 1
        finally:
            capture.release()

        local_scores = self._local_scores(
            numpy,
            numpy.asarray(tile_history, dtype=numpy.float32),
            [row[5] for row in measurements],
        )
        samples = [
            MotionSample(
                frame_index=index,
                timestamp_seconds=index / info.fps,
                score=score,
                tile_peak=tile_peak,
                blob_area=blob_area,
                local_score=local,
                focus=focus,
                is_warmup=is_warmup,
            )
            for (index, score, tile_peak, blob_area, focus, is_warmup), local in zip(
                measurements, local_scores
            )
        ]
        logger.info(
            "Collected %d motion samples (%d warmup) over a %dx%d tile grid",
            len(samples),
            sum(sample.is_warmup for sample in samples),
            rows,
            columns,
        )
        return info, samples

    def _local_scores(self, numpy: Any, tiles: Any, warmup: Sequence[bool]) -> list[float]:
        """Score each sample by its loudest tile *relative to that tile's own history*.

        A tile's baseline is its own median and its own spread up to
        `tile_baseline_quantile`, floored at `tile_floor` so a tile that is
        never disturbed cannot divide by nothing and turn a single noisy pixel
        into an enormous score. That floor also caps the metric: a normally
        empty tile that fills completely scores ``1 / tile_floor``, so scores
        saturate at 100 with the default floor and loud windows can tie there.
        The result is in units of "how unusual is this, here" — which puts a
        distant figure in a quiet corner on the same footing as a lorry crossing
        the foreground, instead of three orders of magnitude below it.
        """
        if tiles.size == 0:
            return [0.0] * len(warmup)
        awake = numpy.asarray([not flag for flag in warmup], dtype=bool)
        calibration = tiles[awake] if bool(awake.any()) else tiles
        baseline = numpy.median(calibration, axis=0)
        upper = numpy.quantile(calibration, self.config.tile_baseline_quantile, axis=0)
        spread = numpy.maximum(upper - baseline, self.config.tile_floor)
        excess = ((tiles - baseline) / spread).max(axis=1)
        return [0.0 if flag else max(0.0, float(value)) for flag, value in zip(warmup, excess)]

    def find(self, video_path: str | Path) -> tuple[VideoInfo, list[RelevantWindow]]:
        info, samples = self.sample_motion(video_path)
        return info, self.build_windows(samples, info.duration_seconds)

    def metric_values(self, samples: Sequence[MotionSample]) -> list[float]:
        """The numbers the gate actually thresholds: the configured metric, warmup excluded.

        Warmup samples are zero by construction, so leaving them in would drag
        every statistic towards zero and quietly lower the cutoff.
        """
        awake = [sample for sample in samples if not sample.is_warmup] or list(samples)
        return [sample.metric(self.config.motion_metric) for sample in awake]

    def resolve_motion_threshold(self, samples: Sequence[MotionSample]) -> float:
        """Return the numeric motion cutoff for these samples."""
        return self.config.resolve_threshold(self.metric_values(samples))

    def resolve_motion_thresholds(
        self, samples: Sequence[MotionSample]
    ) -> tuple[float | None, float | None]:
        """Return the active ``(lower, upper)`` bounds, in the configured metric's units."""
        return self.config.resolve_std_thresholds(self.metric_values(samples))

    def build_windows(
        self, samples: Sequence[MotionSample], duration_seconds: float | None = None
    ) -> list[RelevantWindow]:
        """Build windows from supplied samples, then mark the ones worth reading.

        `segmentation="events"` grows each window around a run of motion;
        `segmentation="grid"` restores the original fixed tiling. Either way the
        full set is returned and `is_relevant` marks the selection, so a caller
        can always see what was found and rejected.
        """
        if not samples:
            return []
        duration = (
            duration_seconds
            if duration_seconds is not None
            else samples[-1].timestamp_seconds + max(0.0, _sample_interval(samples))
        )
        windows = (
            self._grid_windows(samples, duration)
            if self.config.segmentation == "grid"
            else self._event_windows(samples, duration)
        )
        return self._select(windows)

    def _grid_windows(
        self, samples: Sequence[MotionSample], duration: float
    ) -> list[RelevantWindow]:
        """Fixed `window_seconds` tiling, thresholded on the window's peak."""
        config = self.config
        metric = config.motion_metric
        lower_threshold, upper_threshold = self.resolve_motion_thresholds(samples)
        timestamps = [sample.timestamp_seconds for sample in samples]
        windows: list[RelevantWindow] = []
        start = 0.0
        while start < duration:
            end = min(start + config.window_seconds, duration)
            included = samples[bisect_left(timestamps, start):bisect_left(timestamps, end)]
            if included:
                values = [sample.metric(metric) for sample in included]
                peak = max(values)
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
                        mean_motion=round(sum(values) / len(values), 6),
                        is_relevant=is_relevant,
                        motion_metric=metric,
                        relevance=round(peak, 6),
                        focus=max(included, key=lambda sample: sample.metric(metric)).focus,
                        active_seconds=round(end - start, 3),
                    )
                )
            start += config.stride_seconds
        logger.info(
            "Built %d grid windows (%d relevant) on '%s' with bounds (%s, %s)",
            len(windows),
            sum(window.is_relevant for window in windows),
            metric,
            f"{lower_threshold:.6f}" if lower_threshold is not None else "none",
            f"{upper_threshold:.6f}" if upper_threshold is not None else "none",
        )
        return windows

    def _event_windows(
        self, samples: Sequence[MotionSample], duration: float
    ) -> list[RelevantWindow]:
        """Grow a window around each run of motion, so its extent means something.

        The original gate cut the video on a fixed 60s grid and asked whether
        any sample inside was loud. That answers "did something happen near
        here" with a window that is mostly not it — a 0.4s object handoff
        arrives wrapped in 59.6s of empty car park, and an event straddling a
        boundary is split across two windows that each look half as interesting.

        Here the signal decides the cut: a run starts when motion crosses the
        threshold, continues while it stays above `hysteresis_ratio` of it (so a
        single quiet frame mid-event does not end the event), and is then padded
        for context and merged with its neighbours.
        """
        config = self.config
        metric = config.motion_metric
        if config.motion_std_direction.strip().lower() != "upper":
            logger.warning(
                "segmentation='events' detects motion, so motion_std_direction=%r is ignored; "
                "use segmentation='grid' to flag unusually still windows",
                config.motion_std_direction,
            )
        interval = _sample_interval(samples)
        enter = self.resolve_motion_threshold(samples)
        leave = enter * config.hysteresis_ratio

        # Segment on a smoothed copy so one noisy sample cannot open a window,
        # but report the real measurements.
        smoothed = _rolling_median(
            [0.0 if sample.is_warmup else sample.metric(metric) for sample in samples],
            config.smooth_samples,
        )
        runs: list[tuple[int, int]] = []
        start_index: int | None = None
        for index, value in enumerate(smoothed):
            if start_index is None:
                if value >= enter:
                    start_index = index
            elif value < leave:
                runs.append((start_index, index - 1))
                start_index = None
        if start_index is not None:
            runs.append((start_index, len(smoothed) - 1))

        spans = [
            (
                max(0.0, samples[first].timestamp_seconds - config.pad_seconds),
                min(duration, samples[last].timestamp_seconds + interval + config.pad_seconds),
                samples[last].timestamp_seconds + interval - samples[first].timestamp_seconds,
            )
            for first, last in runs
        ]
        spans = _merge_spans(spans, config.merge_gap_seconds)
        spans = [_at_least(span, config.min_window_seconds, duration) for span in spans]
        spans = [chunk for span in spans for chunk in _split_span(span, config.max_window_seconds)]

        windows = [
            window
            for span in spans
            if (window := self._window_for(samples, span, metric, duration)) is not None
        ]
        logger.info(
            "Built %d event windows on '%s' covering %.1fs of %.1fs (%.0f%%), threshold %.4f",
            len(windows),
            metric,
            sum(window.duration_seconds for window in windows),
            duration,
            100 * sum(window.duration_seconds for window in windows) / duration if duration else 0,
            enter,
        )
        return windows

    def _window_for(
        self,
        samples: Sequence[MotionSample],
        span: tuple[float, float, float],
        metric: str,
        duration: float,
    ) -> RelevantWindow | None:
        """Summarise the samples inside one span."""
        start, end, active = span
        timestamps = [sample.timestamp_seconds for sample in samples]
        included = samples[bisect_left(timestamps, start):bisect_left(timestamps, end)]
        if not included:
            return None
        values = [sample.metric(metric) for sample in included]
        peak = max(values)
        loudest = max(included, key=lambda sample: sample.metric(metric))
        fps = _frames_per_second(samples)
        return RelevantWindow(
            start_seconds=round(start, 3),
            end_seconds=round(end, 3),
            start_frame=round(start * fps) if fps else included[0].frame_index,
            end_frame=round(end * fps) if fps else included[-1].frame_index,
            sample_count=len(included),
            peak_motion=round(peak, 6),
            mean_motion=round(sum(values) / len(values), 6),
            is_relevant=True,
            motion_metric=metric,
            relevance=round(peak, 6),
            focus=loudest.focus,
            active_seconds=round(min(active, end - start), 3),
        )

    def _select(self, windows: Sequence[RelevantWindow]) -> list[RelevantWindow]:
        """Spend the window budget, widest coverage first.

        Windows are ranked by `relevance` — the peak, not the total — so a brief
        intense event is not outranked by a long tepid one. With
        `spatial_diversity` the ranking is then interleaved by region, so the
        budget buys the best window from each part of the frame before it buys a
        second one from the busiest part. That is what keeps a corner of the car
        park from being crowded out by the road.
        """
        config = self.config
        windows = list(windows)
        if config.max_windows is None and config.max_total_seconds is None:
            return windows
        # `local` saturates at 1/tile_floor once a normally-empty cell fills
        # completely, so several windows can share the top score. Sustained
        # motion breaks the tie ahead of whichever one happens to come first.
        ranked = sorted(
            (window for window in windows if window.is_relevant),
            key=lambda window: (-window.relevance, -window.active_seconds, window.start_seconds),
        )
        if config.spatial_diversity:
            ranked = _interleave_by_region(ranked, config.tile_grid, config.diversity_grid)

        chosen: set[int] = set()
        total = 0.0
        for window in ranked:
            if config.max_windows is not None and len(chosen) >= config.max_windows:
                break
            if (
                config.max_total_seconds is not None
                and total + window.duration_seconds > config.max_total_seconds
            ):
                continue
            chosen.add(id(window))
            total += window.duration_seconds
        selected = [
            window if id(window) in chosen or not window.is_relevant
            else replace(window, is_relevant=False)
            for window in windows
        ]
        logger.info(
            "Selected %d of %d windows (%.1fs) under budget (max_windows=%s, max_total_seconds=%s)",
            sum(window.is_relevant for window in selected),
            len(selected),
            total,
            config.max_windows,
            config.max_total_seconds,
        )
        return selected

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


def _import_numpy() -> Any:
    try:
        import numpy  # type: ignore[import-not-found]
    except ImportError as error:
        raise ImportError("Video processing requires `pip install videometa[vision]`.") from error
    return numpy


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as error:
        raise ImportError("Video processing requires `pip install videometa[vision]`.") from error
    return cv2

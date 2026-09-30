"""Public API for videometa."""

from importlib.metadata import PackageNotFoundError, version

from videometa.activity_gate import (
    ACTIVITY_SCORE_GUIDANCE,
    ActivityGateConfig,
    ActivityScorer,
    ActivityWindowFinder,
    LocalQwenActivityScorer,
    build_activity_prompt,
    chunk_spans,
    parse_activity_score,
    write_chunk_clips,
)
from videometa.annotation import (
    BoundingBox,
    DetectionConfig,
    EventBackend,
    EventExtractor,
    EventObject,
    FrameAnnotations,
    ObjectBoundaryExtractor,
    ObjectDetection,
    ObjectWindowAnnotations,
    RelevantWindow,
    TrackedObject,
    VideoAnnotations,
    VideoAnnotator,
    VideoEvent,
    VideoInfo,
    WindowFinder,
    describe_spatial_position,
    probe_video,
    resolve_device,
    resolve_video_source,
    set_verbose,
)
from videometa.local_qwen import DEFAULT_MODEL_ID, load_local_qwen
from videometa.window_annotation import (
    LVLMEventAnnotator,
    LocalQwenEventAnnotator,
    PreparedWindowInput,
    WindowSpatialFeatureJoiner,
)

try:
    __version__ = version("videometa")
except PackageNotFoundError:
    __version__ = "0.0.1"

__all__ = [
    "ACTIVITY_SCORE_GUIDANCE",
    "ActivityGateConfig",
    "ActivityScorer",
    "ActivityWindowFinder",
    "BoundingBox",
    "DEFAULT_MODEL_ID",
    "DetectionConfig",
    "EventBackend",
    "EventExtractor",
    "EventObject",
    "FrameAnnotations",
    "LocalQwenActivityScorer",
    "ObjectBoundaryExtractor",
    "ObjectDetection",
    "ObjectWindowAnnotations",
    "RelevantWindow",
    "TrackedObject",
    "VideoAnnotations",
    "VideoAnnotator",
    "VideoEvent",
    "VideoInfo",
    "WindowFinder",
    "LVLMEventAnnotator",
    "LocalQwenEventAnnotator",
    "PreparedWindowInput",
    "WindowSpatialFeatureJoiner",
    "build_activity_prompt",
    "chunk_spans",
    "describe_spatial_position",
    "load_local_qwen",
    "parse_activity_score",
    "probe_video",
    "resolve_device",
    "resolve_video_source",
    "set_verbose",
    "write_chunk_clips",
]

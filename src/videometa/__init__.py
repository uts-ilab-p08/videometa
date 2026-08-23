"""Public API for videometa."""

from importlib.metadata import PackageNotFoundError, version

from videometa.annotation import (
    BoundingBox,
    DetectionConfig,
    EventBackend,
    EventExtractor,
    EventObject,
    FrameAnnotations,
    MotionGateConfig,
    MotionSample,
    ObjectBoundaryExtractor,
    ObjectDetection,
    ObjectWindowAnnotations,
    RelevantWindow,
    RelevantWindowFinder,
    TrackedObject,
    VideoAnnotations,
    VideoAnnotator,
    VideoEvent,
    VideoInfo,
    describe_spatial_position,
)

try:
    __version__ = version("videometa")
except PackageNotFoundError:
    __version__ = "0.0.1"

__all__ = [
    "BoundingBox",
    "DetectionConfig",
    "EventBackend",
    "EventExtractor",
    "EventObject",
    "FrameAnnotations",
    "MotionGateConfig",
    "MotionSample",
    "ObjectBoundaryExtractor",
    "ObjectDetection",
    "ObjectWindowAnnotations",
    "RelevantWindow",
    "RelevantWindowFinder",
    "TrackedObject",
    "VideoAnnotations",
    "VideoAnnotator",
    "VideoEvent",
    "VideoInfo",
    "describe_spatial_position",
]
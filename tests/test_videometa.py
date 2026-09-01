from pathlib import Path

from videometa import (
    BoundingBox,
    EventExtractor,
    MotionGateConfig,
    MotionSample,
    ObjectWindowAnnotations,
    RelevantWindow,
    RelevantWindowFinder,
    TrackedObject,
    WindowSpatialFeatureJoiner,
    describe_spatial_position,
    resolve_video_source,
)
from videometa.window_annotation import _sample_frame_features


def test_window_builder_uses_peak_motion_and_overlap() -> None:
    finder = RelevantWindowFinder(
        MotionGateConfig(
            window_seconds=5,
            stride_seconds=4,
            motion_threshold=0.1,
        )
    )
    samples = [
        MotionSample(frame_index=index * 30, timestamp_seconds=float(index * 2), score=score)
        for index, score in enumerate((0.01, 0.02, 0.3, 0.01, 0.02, 0.01))
    ]

    windows = finder.build_windows(samples, duration_seconds=11)

    assert [(window.start_seconds, window.end_seconds) for window in windows] == [
        (0.0, 5.0),
        (4.0, 9.0),
        (8.0, 11),
    ]
    assert [window.is_relevant for window in windows] == [True, True, False]
    assert windows[0].peak_motion == 0.3


def test_motion_sampling_uses_source_fps_by_default() -> None:
    assert MotionGateConfig().sample_fps is None
    assert MotionGateConfig(sample_fps=2).sample_fps == 2


def test_resolve_video_source_accepts_local_paths() -> None:
    video = Path(__file__).with_name("2018-03-05.13-15-00.13-20-00.bus.G340.r13.avi")

    assert resolve_video_source(video) == video


def test_calibration_does_not_mutate_finder_configuration() -> None:
    finder = RelevantWindowFinder(MotionGateConfig(motion_threshold=0.2))
    samples = [MotionSample(0, 0.0, 0.3), MotionSample(1, 1.0, 0.1)]

    results = finder.calibrate(samples, thresholds=(0.05, 0.4))

    assert results[0.05][0].is_relevant
    assert not results[0.4][0].is_relevant
    assert finder.config.motion_threshold == 0.2


def test_motion_threshold_avg_uses_sample_mean() -> None:
    finder = RelevantWindowFinder(
        MotionGateConfig(window_seconds=5, stride_seconds=5, motion_threshold="avg")
    )
    samples = [
        MotionSample(0, 0.0, 0.1),
        MotionSample(1, 1.0, 0.3),
        MotionSample(2, 2.0, 0.2),
    ]

    threshold = finder.resolve_motion_threshold(samples)
    windows = finder.build_windows(samples, duration_seconds=5)

    assert threshold == 0.2
    assert windows[0].is_relevant
    assert finder.config.motion_threshold == "avg"


def test_motion_threshold_median_uses_sample_median() -> None:
    finder = RelevantWindowFinder(
        MotionGateConfig(window_seconds=5, stride_seconds=5, motion_threshold="median")
    )
    samples = [
        MotionSample(0, 0.0, 0.1),
        MotionSample(1, 1.0, 0.9),
        MotionSample(2, 2.0, 0.2),
    ]

    windows = finder.build_windows(samples, duration_seconds=5)

    assert finder.resolve_motion_threshold(samples) == 0.2
    assert windows[0].is_relevant


def test_motion_threshold_std_uses_mean_plus_scaled_stdev() -> None:
    finder = RelevantWindowFinder(
        MotionGateConfig(window_seconds=5, stride_seconds=5, motion_threshold="std")
    )
    samples = [
        MotionSample(0, 0.0, 0.1),
        MotionSample(1, 1.0, 0.2),
        MotionSample(2, 2.0, 0.3),
    ]

    threshold = finder.resolve_motion_threshold(samples)
    windows = finder.build_windows(samples, duration_seconds=5)

    assert threshold == 0.35
    assert not windows[0].is_relevant


def test_motion_threshold_std_can_select_lower_or_both_bounds() -> None:
    samples = [
        MotionSample(0, 0.0, 0.1),
        MotionSample(1, 1.0, 0.1),
        MotionSample(2, 2.0, 0.9),
        MotionSample(3, 3.0, 0.9),
    ]
    lower_finder = RelevantWindowFinder(
        MotionGateConfig(
            window_seconds=2,
            stride_seconds=2,
            motion_threshold="std",
            motion_std_k=0.5,
            motion_std_direction="lower",
        )
    )
    both_finder = RelevantWindowFinder(
        MotionGateConfig(
            window_seconds=2,
            stride_seconds=2,
            motion_threshold="std",
            motion_std_k=0.5,
            motion_std_direction="both",
        )
    )

    lower_threshold, upper_threshold = lower_finder.resolve_motion_thresholds(samples)
    assert round(lower_threshold or 0.0, 6) == 0.26906
    assert upper_threshold is None
    assert [window.is_relevant for window in lower_finder.build_windows(samples, 4)] == [True, False]
    assert [window.is_relevant for window in both_finder.build_windows(samples, 4)] == [True, True]


def test_motion_threshold_rejects_unknown_std_direction() -> None:
    try:
        MotionGateConfig(motion_threshold="std", motion_std_direction="sideways")
    except ValueError as error:
        assert "upper" in str(error)
    else:
        raise AssertionError("expected ValueError for unknown motion_std_direction")


def test_motion_threshold_rejects_unknown_statistic() -> None:
    try:
        MotionGateConfig(motion_threshold="p90")
    except ValueError as error:
        assert "avg" in str(error)
        assert "median" in str(error)
        assert "std" in str(error)
    else:
        raise AssertionError("expected ValueError for unknown motion_threshold")


def test_spatial_description_uses_boundary_centre() -> None:
    boundary = BoundingBox(0, 0, 100, 100)
    assert describe_spatial_position(boundary, width=300, height=300) == "top-left"
    assert describe_spatial_position(boundary, width=300, height=300, grid=4) == "row-1-column-1"


def test_event_extractor_normalizes_backend_event_shape() -> None:
    class FakeBackend:
        def analyze(self, video_path, start_seconds, end_seconds, object_context):
            assert video_path == "video.mp4"
            assert len(object_context) == 1
            return [
                {
                    "event_name": "Vehicle enters",
                    "description": "A sedan enters from the left.",
                    "involved_objects": [
                        {"id": 7, "label": "car", "physical_details": "white sedan"}
                    ],
                }
            ]

    window = RelevantWindow(0, 5, 0, 149, 3, 0.5, 0.2, True)
    tracked = TrackedObject(7, "car", 0, 149, 0, 4.9, 0.9, ("middle-left",))
    annotations = ObjectWindowAnnotations(window, (tracked,), ())

    events = EventExtractor(FakeBackend()).extract("video.mp4", [annotations])

    assert events[0].name == "Vehicle enters"
    assert events[0].involved_objects[0].object_id == "7"


def test_joiner_defaults_to_qwen_safe_video_shape() -> None:
    joiner = WindowSpatialFeatureJoiner()

    assert joiner.annotated_video_size == (640, 360)
    assert joiner.annotated_video_fps == 2.0


def test_qwen_frame_features_are_sampled() -> None:
    frames = [{"frame_index": index} for index in range(150)]

    sampled = _sample_frame_features(frames, max_frames=12)

    assert len(sampled) == 12
    assert sampled[0]["frame_index"] == 0

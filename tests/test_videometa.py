from videometa import (
    BoundingBox,
    EventExtractor,
    MotionGateConfig,
    MotionSample,
    ObjectWindowAnnotations,
    RelevantWindow,
    RelevantWindowFinder,
    TrackedObject,
    describe_spatial_position,
)


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


def test_calibration_does_not_mutate_finder_configuration() -> None:
    finder = RelevantWindowFinder(MotionGateConfig(motion_threshold=0.2))
    samples = [MotionSample(0, 0.0, 0.3), MotionSample(1, 1.0, 0.1)]

    results = finder.calibrate(samples, thresholds=(0.05, 0.4))

    assert results[0.05][0].is_relevant
    assert not results[0.4][0].is_relevant
    assert finder.config.motion_threshold == 0.2


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

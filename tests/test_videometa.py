from dataclasses import replace
from pathlib import Path

from videometa import (
    BoundingBox,
    DetectionConfig,
    EventExtractor,
    MotionGateConfig,
    MotionSample,
    FrameAnnotations,
    ObjectBoundaryExtractor,
    ObjectDetection,
    ObjectWindowAnnotations,
    RelevantWindow,
    RelevantWindowFinder,
    TrackedObject,
    PreparedWindowInput,
    WindowSpatialFeatureJoiner,
    describe_spatial_position,
    resolve_video_source,
)
from videometa.window_annotation import _sample_frame_features


def legacy_config(**overrides: object) -> MotionGateConfig:
    """The fixed-grid, frame-mean gate that shipped before event segmentation."""
    return MotionGateConfig(segmentation="grid", motion_metric="score", **overrides)  # type: ignore[arg-type]


def test_window_builder_uses_peak_motion_and_overlap() -> None:
    finder = RelevantWindowFinder(
        legacy_config(
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
    finder = RelevantWindowFinder(legacy_config(motion_threshold=0.2))
    samples = [MotionSample(0, 0.0, 0.3), MotionSample(1, 1.0, 0.1)]

    results = finder.calibrate(samples, thresholds=(0.05, 0.4))

    assert results[0.05][0].is_relevant
    assert not results[0.4][0].is_relevant
    assert finder.config.motion_threshold == 0.2


def test_motion_threshold_avg_uses_sample_mean() -> None:
    finder = RelevantWindowFinder(
        legacy_config(window_seconds=5, stride_seconds=5, motion_threshold="avg")
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
        legacy_config(window_seconds=5, stride_seconds=5, motion_threshold="median")
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
        legacy_config(window_seconds=5, stride_seconds=5, motion_threshold="std")
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
        legacy_config(
            window_seconds=2,
            stride_seconds=2,
            motion_threshold="std",
            motion_std_k=0.5,
            motion_std_direction="lower",
        )
    )
    both_finder = RelevantWindowFinder(
        legacy_config(
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


def test_detection_config_defaults_to_all_yolo_classes() -> None:
    config = DetectionConfig()

    assert config.classes is None
    assert config.resolve_classes() is None
    assert config.resolve_classes(type("Model", (), {"names": {0: "person", 2: "car"}})()) == [0, 2]


def test_detection_config_stores_unique_class_ids() -> None:
    config = DetectionConfig(classes=[0, 2, 0, 5])

    assert config.classes == (0, 2, 5)
    assert config.resolve_classes() == [0, 2, 5]


def test_detection_config_empty_classes_means_all() -> None:
    assert DetectionConfig(classes=[]).classes is None


def test_detection_config_rejects_invalid_class_ids() -> None:
    try:
        DetectionConfig(classes=[-1])
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("expected ValueError for negative YOLO class IDs")


def test_track_frame_passes_resolved_yolo_classes() -> None:
    class FakeResult:
        boxes = None

    class FakeModel:
        names = {0: "person", 1: "bicycle", 2: "car"}

        def track(self, frame, **kwargs):
            captured.append(kwargs)
            return [FakeResult()]

    captured: list[dict] = []
    extractor = ObjectBoundaryExtractor(DetectionConfig(classes=[0, 2, 5]))
    extractor._track_frame(FakeModel(), frame=object(), width=100, height=100)

    assert captured[0]["classes"] == [0, 2, 5]
    assert captured[0]["conf"] == 0.25

    captured.clear()
    ObjectBoundaryExtractor()._track_frame(FakeModel(), frame=object(), width=100, height=100)
    assert captured[0]["classes"] == [0, 1, 2]


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


def motion_samples(values: list[float], fps: float = 30.0, sample_fps: float = 10.0,
                   focus: tuple[int, int] | None = (4, 8)) -> list[MotionSample]:
    """Samples carrying `values` as the local score, spaced at `sample_fps`."""
    step = int(round(fps / sample_fps))
    return [
        MotionSample(
            frame_index=index * step,
            timestamp_seconds=index * step / fps,
            score=0.001,
            local_score=value,
            focus=focus,
            is_warmup=index < 2,
        )
        for index, value in enumerate(values)
    ]


def test_event_windows_hug_the_motion_instead_of_a_fixed_grid() -> None:
    # 40s of quiet with a 1s burst at 20s: the old grid returned a 60s window,
    # the event gate should return something the length of the burst plus padding.
    values = [0.0] * 400
    for index in range(200, 210):
        values[index] = 50.0
    finder = RelevantWindowFinder(
        MotionGateConfig(motion_threshold=5.0, pad_seconds=2.0, min_window_seconds=2.0)
    )

    windows = finder.build_windows(motion_samples(values), duration_seconds=40.0)

    assert len(windows) == 1
    window = windows[0]
    assert window.is_relevant
    assert (window.start_seconds, window.end_seconds) == (18.0, 23.0)
    assert window.peak_motion == 50.0
    assert round(window.active_seconds, 1) == 1.0
    assert window.motion_metric == "local"


def test_hysteresis_keeps_one_quiet_sample_from_splitting_an_event() -> None:
    values = [0.0] * 200
    for index in range(100, 120):
        values[index] = 40.0
    values[110] = 12.0  # a dip that clears the threshold but not the hysteresis floor
    finder = RelevantWindowFinder(
        MotionGateConfig(motion_threshold=20.0, hysteresis_ratio=0.5, pad_seconds=0.0,
                         merge_gap_seconds=0.0, min_window_seconds=0.0)
    )

    windows = finder.build_windows(motion_samples(values), duration_seconds=20.0)

    assert len(windows) == 1
    assert (windows[0].start_seconds, windows[0].end_seconds) == (10.0, 12.0)


def test_short_events_are_widened_and_long_ones_are_split() -> None:
    values = [0.0] * 600
    for index in range(100, 103):
        values[index] = 50.0                # a 0.3s event
    for index in range(300, 500):
        values[index] = 50.0                # a 20s block
    finder = RelevantWindowFinder(
        MotionGateConfig(motion_threshold=5.0, pad_seconds=0.0, merge_gap_seconds=0.0,
                         min_window_seconds=4.0, max_window_seconds=8.0)
    )

    windows = finder.build_windows(motion_samples(values), duration_seconds=60.0)

    durations = [round(window.duration_seconds, 3) for window in windows]
    assert durations[0] == 4.0              # widened from 0.3s, not dropped
    assert all(duration <= 8.0 for duration in durations)
    assert round(sum(durations[1:]), 1) == 20.0


def test_a_single_sample_spike_is_treated_as_noise() -> None:
    values = [0.0] * 600
    values[100] = 50.0                      # one sample, with quiet on both sides
    finder = RelevantWindowFinder(
        MotionGateConfig(motion_threshold=5.0, smooth_samples=3, min_window_seconds=0.0)
    )

    assert finder.build_windows(motion_samples(values), duration_seconds=60.0) == []
    # ...unless the caller turns the median filter off
    unsmoothed = RelevantWindowFinder(
        MotionGateConfig(motion_threshold=5.0, smooth_samples=1, min_window_seconds=0.0)
    )
    assert len(unsmoothed.build_windows(motion_samples(values), duration_seconds=60.0)) == 1


def test_windows_are_ranked_by_peak_and_capped_by_budget() -> None:
    values = [0.0] * 600
    for index in range(100, 120):
        values[index] = 10.0                # quieter, earlier
    for index in range(300, 320):
        values[index] = 90.0                # louder, later
    finder = RelevantWindowFinder(
        MotionGateConfig(motion_threshold=5.0, pad_seconds=0.0, merge_gap_seconds=0.0,
                         min_window_seconds=0.0, max_windows=1)
    )

    windows = finder.build_windows(motion_samples(values), duration_seconds=60.0)

    assert [window.is_relevant for window in windows] == [False, True]
    assert windows[1].peak_motion == 90.0
    # nothing is thrown away, so a caller can still see what was rejected
    assert windows[0].peak_motion == 10.0


def test_spatial_diversity_spends_the_budget_across_regions() -> None:
    quiet_corner = motion_samples([0.0] * 300, focus=(0, 15))
    for index in range(100, 120):
        quiet_corner[index] = replace(quiet_corner[index], local_score=20.0)
    busy_centre = motion_samples([0.0] * 300, focus=(4, 8))
    for index in range(200, 220):
        busy_centre[index] = replace(busy_centre[index], local_score=90.0)
    merged = [
        corner if corner.local_score >= centre.local_score else centre
        for corner, centre in zip(quiet_corner, busy_centre)
    ]
    # a second, even louder burst in the busy centre
    for index in range(250, 270):
        merged[index] = replace(merged[index], local_score=95.0, focus=(4, 8))

    def relevant_focuses(diversity: bool) -> list[tuple[int, int] | None]:
        finder = RelevantWindowFinder(
            MotionGateConfig(motion_threshold=5.0, pad_seconds=0.0, merge_gap_seconds=0.0,
                             min_window_seconds=0.0, max_windows=2,
                             spatial_diversity=diversity)
        )
        return [w.focus for w in finder.build_windows(merged, 30.0) if w.is_relevant]

    assert relevant_focuses(False) == [(4, 8), (4, 8)]     # both go to the loudest region
    assert sorted(relevant_focuses(True)) == [(0, 15), (4, 8)]


def test_a_still_video_produces_no_windows() -> None:
    finder = RelevantWindowFinder(MotionGateConfig())          # motion_threshold="auto"

    assert finder.build_windows(motion_samples([0.0] * 300), duration_seconds=30.0) == []


def test_auto_threshold_never_falls_below_the_metric_floor() -> None:
    # A video of pure noise must not calibrate its way down into that noise.
    noise = [0.1, 0.2, 0.15, 0.05, 0.2, 0.1] * 20
    config = MotionGateConfig(motion_metric="local")

    assert config.resolve_threshold(noise) == config.metric_floor() == 3.0


def test_mad_threshold_is_not_dragged_up_by_one_loud_sample() -> None:
    samples = [0.1] * 20 + [50.0]
    quiet = MotionGateConfig(motion_metric="score", motion_threshold="mad", motion_std_k=1.5)
    fragile = MotionGateConfig(motion_metric="score", motion_threshold="std", motion_std_k=1.5)

    assert quiet.resolve_threshold(samples) == 0.1          # median + 1.5 * 0 MAD
    assert fragile.resolve_threshold(samples) > 18.0        # mean + 1.5 * a huge stdev


def test_warmup_samples_are_excluded_from_the_threshold() -> None:
    values = [0.0] * 50 + [30.0] * 50
    finder = RelevantWindowFinder(MotionGateConfig(motion_threshold="median", pad_seconds=0.0,
                                                   min_window_seconds=0.0))
    samples = [
        replace(sample, is_warmup=index < 50)
        for index, sample in enumerate(motion_samples(values))
    ]

    # with warmup zeros counted the median would be 0 and everything would flag;
    # ignoring them puts the cut at 30 and only the moving half survives
    windows = finder.build_windows(samples, duration_seconds=10.0)
    assert all(window.start_seconds >= 5.0 for window in windows)


def test_motion_metric_selects_the_field_that_is_thresholded() -> None:
    sample = MotionSample(0, 0.0, score=0.5, tile_peak=0.8, blob_area=120.0, local_score=9.0)

    assert sample.metric("score") == 0.5
    assert sample.metric("tile_peak") == 0.8
    assert sample.metric("blob_area") == 120.0
    assert sample.metric("local") == 9.0
    try:
        sample.metric("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unknown metric")


def test_invalid_segmentation_parameters_are_rejected() -> None:
    for kwargs in (
        {"motion_metric": "loudness"},
        {"segmentation": "sliding"},
        {"hysteresis_ratio": 1.5},
        {"smooth_samples": 0},
        {"pad_seconds": -1.0},
        {"min_window_seconds": 30.0, "max_window_seconds": 10.0},
        {"tile_floor": 0.0},
    ):
        try:
            MotionGateConfig(**kwargs)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


def test_reported_threshold_matches_the_metric_the_gate_cuts_on() -> None:
    # the helper used to report a cutoff computed from `score` whatever the
    # configured metric was, so a caller plotting it drew the wrong line
    samples = [
        MotionSample(index, index / 10, score=0.5, local_score=20.0)
        for index in range(20)
    ]
    finder = RelevantWindowFinder(MotionGateConfig(motion_metric="local",
                                                   motion_threshold="median"))

    assert finder.resolve_motion_threshold(samples) == 20.0
    assert finder.resolve_motion_thresholds(samples) == (None, 20.0)


def track_profile(first, last, label="person", appearance=(1.0, 0.2, 0.05),
                  box=BoundingBox(100, 100, 140, 200), confidence=0.9):
    """One entry of the extractor's internal per-track profile."""
    return {
        "labels": {label: confidence},
        "first": first,
        "last": last,
        "first_box": box,
        "last_box": box,
        "appearance": list(appearance),
        "samples": 1,
    }


def stitch(profiles, **overrides):
    extractor = ObjectBoundaryExtractor(DetectionConfig(**overrides))
    return extractor._stitch_tracks(profiles, fps=30.0, diagonal=2200.0)


def test_a_reappearing_object_keeps_its_original_identity() -> None:
    # the same person leaves at frame 100 and returns at frame 200 as a new
    # tracker id, because ByteTrack retires a track it cannot see
    profiles = {
        7: track_profile(0, 100),
        9: track_profile(200, 300, appearance=(1.0, 0.21, 0.04)),
    }

    assert stitch(profiles) == {7: 7, 9: 7}


def test_objects_visible_at_the_same_time_are_never_merged() -> None:
    # identical appearance, but both on screen at once: two people, not one
    profiles = {
        1: track_profile(0, 300),
        2: track_profile(100, 400),
    }

    assert stitch(profiles) == {1: 1, 2: 2}


def test_stitching_requires_label_gap_and_distance_to_agree() -> None:
    near = BoundingBox(100, 100, 140, 200)
    far = BoundingBox(1800, 900, 1840, 1000)

    different_label = {1: track_profile(0, 100), 2: track_profile(200, 300, label="car")}
    assert stitch(different_label) == {1: 1, 2: 2}

    long_gap = {1: track_profile(0, 100), 2: track_profile(2000, 2100)}
    assert stitch(long_gap, stitch_max_gap_seconds=30.0) == {1: 1, 2: 2}

    too_far = {1: track_profile(0, 100, box=near), 2: track_profile(103, 200, box=far)}
    assert stitch(too_far, stitch_max_speed=0.01) == {1: 1, 2: 2}

    unlike = {1: track_profile(0, 100, appearance=(1.0, 0.0, 0.0)),
              2: track_profile(200, 300, appearance=(0.0, 0.0, 1.0))}
    assert stitch(unlike) == {1: 1, 2: 2}


def test_three_fragments_of_one_object_collapse_to_a_single_identity() -> None:
    profiles = {
        1: track_profile(0, 100),
        2: track_profile(200, 300, appearance=(1.0, 0.19, 0.06)),
        3: track_profile(400, 500, appearance=(1.0, 0.22, 0.05)),
    }

    assert stitch(profiles) == {1: 1, 2: 1, 3: 1}


def test_stitching_can_be_turned_off() -> None:
    profiles = {7: track_profile(0, 100), 9: track_profile(200, 300)}

    assert stitch(profiles, stitch_tracks=False) == {7: 7, 9: 9}


def test_label_is_a_confidence_weighted_vote_over_the_whole_identity() -> None:
    # YOLO called it a truck once, with low confidence, and a car everywhere else
    profiles = {
        1: {"labels": {"car": 4.2, "truck": 0.3}, "first": 0, "last": 100,
            "first_box": BoundingBox(0, 0, 10, 10), "last_box": BoundingBox(0, 0, 10, 10),
            "appearance": [1.0], "samples": 1},
        2: {"labels": {"truck": 0.9}, "first": 200, "last": 300,
            "first_box": BoundingBox(0, 0, 10, 10), "last_box": BoundingBox(0, 0, 10, 10),
            "appearance": [1.0], "samples": 1},
    }

    assert ObjectBoundaryExtractor._settled_label(profiles, [1]) == "car"
    assert ObjectBoundaryExtractor._settled_label(profiles, [1, 2]) == "car"
    assert ObjectBoundaryExtractor._settled_label(profiles, [2]) == "truck"


def test_frames_are_rewritten_with_the_stable_id_and_settled_label() -> None:
    def detection(track_id, label):
        return ObjectDetection(
            track_id=track_id, label=label, confidence=0.9,
            boundary=BoundingBox(0, 0, 10, 10), spatial_description="top-left",
        )

    frames = [
        FrameAnnotations(0, 0.0, (detection(7, "car"),)),
        FrameAnnotations(210, 7.0, (detection(9, "truck"),)),
    ]
    identity = {7: 7, 9: 7}
    labels = {7: "car"}

    rewritten = [ObjectBoundaryExtractor._relabel_frame(f, identity, labels) for f in frames]

    assert [d.track_id for f in rewritten for d in f.detections] == [7, 7]
    assert [d.label for f in rewritten for d in f.detections] == ["car", "car"]

    # and the window summary then holds one object, not two
    registry = ObjectBoundaryExtractor._registry_for(rewritten)
    assert list(registry) == [7]
    assert registry[7]["first"] == 0 and registry[7]["last"] == 210


def test_correlation_matches_opencvs_histogram_comparison() -> None:
    from videometa.annotation import _correlation

    assert _correlation([1, 2, 3, 4], [2, 4, 6, 8]) == 1.0      # scale invariant
    assert _correlation([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert _correlation([1, 1, 1], [1, 2, 3]) == 0.0            # no spread, no signal
    assert _correlation([1, 2], [1, 2, 3]) == 0.0               # mismatched lengths


def test_merging_pools_label_votes_the_winner_has_not_seen() -> None:
    # both tracks read mostly as "car", but each carries a stray class of its own
    profiles = {
        1: {"labels": {"car": 5.0, "truck": 0.4}, "first": 0, "last": 100,
            "first_box": BoundingBox(0, 0, 10, 10), "last_box": BoundingBox(0, 0, 10, 10),
            "appearance": [1.0, 0.2, 0.05], "samples": 1},
        2: {"labels": {"car": 4.0, "bus": 0.6}, "first": 200, "last": 300,
            "first_box": BoundingBox(0, 0, 10, 10), "last_box": BoundingBox(0, 0, 10, 10),
            "appearance": [1.0, 0.21, 0.04], "samples": 1},
    }

    assert stitch(profiles) == {1: 1, 2: 1}
    assert ObjectBoundaryExtractor._settled_label(profiles, [1, 2]) == "car"


def test_travel_allowance_scales_with_the_gap_not_a_flat_second() -> None:
    here = BoundingBox(100, 100, 140, 200)        # centre (120, 150)
    there = BoundingBox(320, 100, 360, 200)       # centre (340, 150) — 220px away

    # 220px in 0.1s is ~2200px/s across a 2200px diagonal: a different object
    sprinting = {1: track_profile(0, 100, box=here), 2: track_profile(103, 200, box=there)}
    assert stitch(sprinting) == {1: 1, 2: 2}

    # the same 220px with 3s to cover it is an ordinary walk
    strolling = {1: track_profile(0, 100, box=here), 2: track_profile(190, 300, box=there)}
    assert stitch(strolling) == {1: 1, 2: 1}

    # and a track that never moved rejoins across a short gap, jitter included
    jittering = {
        1: track_profile(0, 100, box=here),
        2: track_profile(103, 200, box=BoundingBox(112, 100, 152, 200)),
    }
    assert stitch(jittering) == {1: 1, 2: 1}


def prepared_window() -> PreparedWindowInput:
    return PreparedWindowInput(
        start_seconds=30.0,
        end_seconds=42.0,
        object_features=(
            {"track_id": 201, "label": "car", "average_confidence": 0.8,
             "spatial_trajectory": ("top-left", "middle-left")},
        ),
        frame_features=(
            {"timestamp_seconds": 30.0,
             "detections": [{"track_id": 201, "label": "car",
                             "spatial_position": "top-left"}]},
        ),
        image_messages=(),
        artifact_directory=None,
        annotated_video_path="clip.mp4",
    )


def test_both_prompts_carry_the_same_description_rules() -> None:
    from types import SimpleNamespace

    from videometa.window_annotation import (
        DESCRIPTION_GUIDANCE,
        LVLMEventAnnotator,
        LocalQwenEventAnnotator,
    )

    window = prepared_window()
    hosted = LVLMEventAnnotator._build_prompt(SimpleNamespace(feature_frames=4), window)
    local = LocalQwenEventAnnotator._build_window_prompt(None, window, 4)

    for prompt in (hosted, local):
        assert DESCRIPTION_GUIDANCE in prompt
        # the rules that answer the complaint about frame-relative wording
        assert "Never write a track id" in prompt
        assert "frame coordinates provided to help you locate the subject" in prompt
        assert "never 'moves from top to bottom'" in prompt
        # the grid vocabulary is still supplied, but framed as a lookup hint
        assert "frame-grid cells" in prompt
        assert "top-left" in prompt


def test_track_ids_are_stripped_from_prose_but_kept_as_fields() -> None:
    from videometa.window_annotation import _clean_events, _strip_track_ids

    assert _strip_track_ids("car #201 moves to the left") == "car moves to the left"
    assert _strip_track_ids("The white SUV (track 201) reverses.") == "The white SUV reverses."
    assert _strip_track_ids("Vehicle track id 42 stops") == "Vehicle stops"
    # wording that is not an id survives untouched
    assert _strip_track_ids("the red car drives east") == "the red car drives east"

    cleaned = _clean_events([
        {
            "event_name": "car #201 departs",
            "description": "car #201 pulls away along the access road",
            "involved_objects": [
                {"id": "201", "label": "car", "physical_details": "white SUV #201"}
            ],
        }
    ])

    assert cleaned[0]["event_name"] == "car departs"
    assert cleaned[0]["description"] == "car pulls away along the access road"
    assert cleaned[0]["involved_objects"][0]["physical_details"] == "white SUV"
    assert cleaned[0]["involved_objects"][0]["id"] == "201"     # the id itself is untouched


def test_cleaning_events_tolerates_odd_model_output() -> None:
    from videometa.window_annotation import _clean_events

    assert _clean_events([]) == []
    assert _clean_events(["not a dict"]) == ["not a dict"]
    assert _clean_events([{"description": None}]) == [{"description": None}]
    assert _clean_events([{"involved_objects": "nope"}]) == [{"involved_objects": "nope"}]
    assert _clean_events([{"involved_objects": [{"id": "1"}]}]) == [
        {"involved_objects": [{"id": "1"}]}
    ]


def test_prompt_asks_for_natural_names_and_detailed_descriptions() -> None:
    from videometa.window_annotation import DESCRIPTION_GUIDANCE

    # event_name: plain English, not dataset vocabulary
    assert "snake_case" in DESCRIPTION_GUIDANCE
    assert "person_opens_vehicle_door" in DESCRIPTION_GUIDANCE     # named as a bad example
    assert "sentence case" in DESCRIPTION_GUIDANCE
    # description: detail, with a stated length
    assert "two to four complete sentences" in DESCRIPTION_GUIDANCE
    assert "The end state" in DESCRIPTION_GUIDANCE
    # ordinary movement counts, so a busy window does not come back empty...
    assert "including ordinary movement" in DESCRIPTION_GUIDANCE
    # ...but an empty window is still allowed to return nothing
    assert "return an empty list rather than" in DESCRIPTION_GUIDANCE


def test_prompts_no_longer_gate_on_the_word_relevant() -> None:
    from types import SimpleNamespace

    from videometa.window_annotation import LVLMEventAnnotator, LocalQwenEventAnnotator

    window = prepared_window()
    hosted = LVLMEventAnnotator._build_prompt(SimpleNamespace(feature_frames=4), window)
    local = LocalQwenEventAnnotator._build_window_prompt(None, window, 4)

    for prompt in (hosted, local):
        assert "observable activity" in prompt
        assert "relevant event" not in prompt


def test_both_prompts_carry_the_action_checklist() -> None:
    from types import SimpleNamespace

    from videometa.window_annotation import (
        ACTION_GUIDANCE,
        ACTION_VOCABULARY,
        LVLMEventAnnotator,
        LocalQwenEventAnnotator,
    )

    window = prepared_window()
    hosted = LVLMEventAnnotator._build_prompt(SimpleNamespace(feature_frames=4), window)
    local = LocalQwenEventAnnotator._build_window_prompt(None, window, 4)

    for prompt in (hosted, local):
        assert ACTION_GUIDANCE in prompt
        assert '"actions": [str]' in prompt
        assert "one event per subject per action" in prompt.lower()
        assert "cropped" not in prompt          # this window shows the whole frame
    # the classes a MEVA comparison most often misses are on the checklist
    for action in ("opens a vehicle door", "talks to another person", "vehicle reverses",
                   "vehicle turns right", "texts or looks at a phone",
                   "shakes hands with or touches another person", "buys something or pays at a counter"):
        assert action in ACTION_VOCABULARY


def test_prompts_say_when_the_video_is_a_crop() -> None:
    from types import SimpleNamespace

    from videometa.window_annotation import LVLMEventAnnotator, LocalQwenEventAnnotator

    window = replace(prepared_window(), crop_box=(100, 50, 740, 410))
    hosted = LVLMEventAnnotator._build_prompt(SimpleNamespace(feature_frames=4), window)
    local = LocalQwenEventAnnotator._build_window_prompt(None, window, 4)

    for prompt in (hosted, local):
        assert "cropped to the part of the camera frame" in prompt
        assert "still refer to the full frame" in prompt


def test_actions_are_normalised_to_the_vocabulary() -> None:
    from videometa.window_annotation import _clean_events

    cleaned = _clean_events([
        {
            "event_name": "Driver leaves",
            "description": "The driver opens the door and gets out.",
            "actions": [
                "Opens a vehicle door.",          # case and punctuation
                "the driver gets out of a vehicle",  # phrase embedded in a sentence
                "waves at the camera",            # not on the checklist
                "opens a vehicle door",           # duplicate
                7,                                # not a string
            ],
        }
    ])

    assert cleaned[0]["actions"] == ["opens a vehicle door", "gets out of a vehicle"]
    assert cleaned[0]["other_actions"] == ["waves at the camera"]
    # a single string, a wrong type, and a missing key are all tolerated
    assert _clean_events([{"actions": "vehicle stops"}])[0]["actions"] == ["vehicle stops"]
    assert _clean_events([{"actions": 42}])[0]["actions"] == []
    assert "actions" not in _clean_events([{"description": "no actions key"}])[0]
    assert "other_actions" not in _clean_events([{"actions": ["sits down"]}])[0]


def test_truncated_output_keeps_the_complete_events() -> None:
    import json

    import pytest

    from videometa.window_annotation import _parse_event_list

    cut_off = (
        '```json\n[{"event_name": "A", "description": "one", "actions": []},\n'
        ' {"event_name": "B", "description": "two", "actions": ["sits down"]},\n'
        ' {"event_name": "C", "description": "the model ran out of tok'
    )
    assert [event["event_name"] for event in _parse_event_list(cut_off)] == ["A", "B"]
    # the shapes both annotators receive when nothing went wrong
    assert _parse_event_list('{"events": [{"event_name": "A"}]}') == [{"event_name": "A"}]
    assert _parse_event_list('[{"event_name": "A"}]') == [{"event_name": "A"}]
    # nothing complete before the cut: the original error still surfaces
    with pytest.raises(json.JSONDecodeError):
        _parse_event_list('[{"event_name": "brok')
    with pytest.raises(ValueError):
        _parse_event_list('{"no_events": 1}')


def test_activity_crop_follows_the_actors_not_the_parked_cars() -> None:
    from types import SimpleNamespace

    from videometa import BoundingBox
    from videometa.window_annotation import _activity_crop

    def frame(index, *boxes):
        return SimpleNamespace(
            frame_index=index,
            detections=[
                SimpleNamespace(track_id=tid, label=label, boundary=BoundingBox(*box))
                for tid, label, box in boxes
            ],
        )

    # A car park: parked cars in every corner, one person walking past a car near the bottom.
    parked = [(10, "car", (0, 0, 300, 150)), (11, "car", (1600, 0, 1920, 150)),
              (12, "car", (0, 900, 300, 1080)), (13, "car", (1600, 900, 1920, 1080))]
    frames = [
        frame(0, *parked, (1, "person", (900, 500, 960, 700)), (2, "car", (700, 550, 1000, 720))),
        frame(30, *parked, (1, "person", (1100, 500, 1160, 700)), (2, "car", (700, 550, 1000, 720))),
    ]
    crop = _activity_crop(frames, (1920, 1080), 16 / 9, padding=0.2)

    assert crop is not None
    left, top, right, bottom = crop
    assert left <= 900 and right >= 1160 and top <= 500 and bottom >= 700   # the walking person
    assert right - left < 1920 * 0.8                                         # not the whole car park
    assert abs((right - left) / (bottom - top) - 16 / 9) < 0.05
    assert 0 <= left < right <= 1920 and 0 <= top < bottom <= 1080

    # A single moving car and no people: crop follows the car.
    driving = [frame(0, *parked, (3, "car", (200, 400, 500, 600))),
               frame(30, *parked, (3, "car", (900, 400, 1200, 600)))]
    left, top, right, bottom = _activity_crop(driving, (1920, 1080), 16 / 9)
    assert left <= 200 and right >= 1200 and top <= 400 and bottom >= 600

    # A lone distant person: the crop is never smaller than min_size, so it is not upscaled into a blur.
    tiny = [frame(0, (4, "person", (1000, 300, 1020, 350)))]
    left, top, right, bottom = _activity_crop(tiny, (1920, 1080), 16 / 9, min_size=(640, 360))
    assert right - left >= 640 and bottom - top >= 360
    assert left <= 1000 and right >= 1020 and top <= 300 and bottom >= 350
    assert abs((right - left) / (bottom - top) - 16 / 9) < 0.05

    # Nothing moves and nobody is there: the whole frame.
    assert _activity_crop([frame(0, *parked)], (1920, 1080), 16 / 9) is None
    # People at opposite corners: too wide to help, so the whole frame.
    spread = [frame(0, (1, "person", (10, 10, 100, 200)), (2, "person", (1800, 850, 1900, 1070)))]
    assert _activity_crop(spread, (1920, 1080), 16 / 9) is None
    assert _activity_crop([], (1920, 1080), 16 / 9) is None


def test_overlay_boxes_are_shifted_by_the_crop_offset() -> None:
    from types import SimpleNamespace

    import numpy as np

    from videometa import BoundingBox
    from videometa.window_annotation import _draw_detections

    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    detection = SimpleNamespace(
        label="person", track_id=1, confidence=0.9,
        boundary=BoundingBox(300, 200, 400, 300), spatial_description="middle-center",
    )
    # crop starts at (200, 100) in the source and is scaled by 0.5: the box lands at (50, 50)-(100, 100)
    annotated, features = _draw_detections(
        frame, [detection], scale_x=0.5, scale_y=0.5, offset_x=200, offset_y=100
    )

    assert tuple(annotated[100, 75]) == (255, 255, 255)   # bottom edge of the shifted box
    assert tuple(annotated[250, 300]) == (0, 0, 0)        # where the unshifted box would have been
    assert features[0]["box_xyxy"] == [300, 200, 400, 300]  # features keep source coordinates


def test_joiner_validates_crop_settings() -> None:
    import pytest

    joiner = WindowSpatialFeatureJoiner(crop_to_activity=True)
    assert joiner.crop_to_activity and joiner.crop_padding == 0.2 and joiner.crop_max_area == 0.6
    assert WindowSpatialFeatureJoiner().crop_to_activity is False
    with pytest.raises(ValueError):
        WindowSpatialFeatureJoiner(crop_padding=3)
    with pytest.raises(ValueError):
        WindowSpatialFeatureJoiner(crop_max_area=0)


def test_duplicate_events_are_dropped() -> None:
    from videometa.window_annotation import _clean_events

    event = {"event_name": "Person walks across the car park", "description": "A person walks.",
             "actions": [], "involved_objects": []}
    cleaned = _clean_events([event, dict(event), {**event, "description": "A person runs."}])

    assert len(cleaned) == 2
    assert [item["description"] for item in cleaned] == ["A person walks.", "A person runs."]
    # events with no text at all are never treated as duplicates of each other
    assert len(_clean_events([{"involved_objects": []}, {"involved_objects": []}])) == 2


def test_action_vocabulary_is_a_parameter() -> None:
    from types import SimpleNamespace

    from videometa.window_annotation import (
        ACTION_GUIDANCE,
        LVLMEventAnnotator,
        LocalQwenEventAnnotator,
        _clean_events,
        action_guidance,
    )

    window = prepared_window()
    # no vocabulary: no checklist block in either prompt
    free = SimpleNamespace(feature_frames=4, action_vocabulary=None)
    assert "checklist" not in LVLMEventAnnotator._build_prompt(free, window)
    assert "checklist" not in LocalQwenEventAnnotator._build_window_prompt(free, window, 4)
    assert action_guidance(None) == ""
    # a dataset's own vocabulary replaces the default one
    custom = SimpleNamespace(feature_frames=4, action_vocabulary=("kicks a ball", "scores a goal"))
    prompt = LVLMEventAnnotator._build_prompt(custom, window)
    assert "kicks a ball" in prompt and "opens a vehicle door" not in prompt
    cleaned = _clean_events([{"actions": ["Kicks a ball", "opens a vehicle door"]}],
                            vocabulary=custom.action_vocabulary)
    assert cleaned[0]["actions"] == ["kicks a ball"]
    assert cleaned[0]["other_actions"] == ["opens a vehicle door"]
    # objects without the attribute (older callers, tests) keep the default checklist
    assert ACTION_GUIDANCE in LocalQwenEventAnnotator._build_window_prompt(None, window, 4)

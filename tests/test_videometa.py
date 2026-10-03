from dataclasses import replace
from pathlib import Path

import pytest

from videometa import (
    ActivityGateConfig,
    ActivityWindowFinder,
    BoundingBox,
    DetectionConfig,
    EventExtractor,
    FrameAnnotations,
    ObjectBoundaryExtractor,
    ObjectDetection,
    ObjectWindowAnnotations,
    RelevantWindow,
    TrackedObject,
    PreparedWindowInput,
    VideoInfo,
    WindowSpatialFeatureJoiner,
    chunk_spans,
    describe_spatial_position,
    parse_activity_score,
    resolve_video_source,
    write_chunk_clips,
)
from videometa.window_annotation import _sample_frame_features


class FakeScorer:
    """Scores chunks from a table keyed on their start second."""

    def __init__(self, scores: dict[float, float], fail_at: float | None = None) -> None:
        self.scores = scores
        self.fail_at = fail_at
        self.calls: list[tuple[str, float, float]] = []

    def score(self, clip_path, start_seconds, end_seconds):
        self.calls.append((clip_path, start_seconds, end_seconds))
        if self.fail_at == start_seconds:
            raise RuntimeError("model blew up")
        return {
            "score": self.scores.get(start_seconds, 0.0),
            "subjects": ["person in a dark jacket"],
            "event_count": 1,
            "summary": "A person walks to a car.",
        }


def scored_windows(scores: dict[float, float], duration: float = 30.0):
    """Chunks of a `duration` s video carrying `scores`, keyed on their start second."""
    return [
        RelevantWindow(start, end, round(start * 30), round(end * 30), scores[start], False)
        for start, end in chunk_spans(duration, 10.0, 2.0)
    ]


# ---------------------------------------------------------------- activity gate


def test_chunks_are_ten_seconds_with_two_seconds_of_overlap() -> None:
    assert chunk_spans(30.0, 10.0, 2.0) == [(0.0, 10.0), (8.0, 18.0), (16.0, 26.0), (24.0, 30.0)]
    # the tail is shortened, never padded past the end
    assert chunk_spans(21.0, 10.0, 2.0) == [(0.0, 10.0), (8.0, 18.0), (16.0, 21.0)]
    # a span that would only repeat the overlap is not started
    assert chunk_spans(18.0, 10.0, 2.0) == [(0.0, 10.0), (8.0, 18.0)]
    assert chunk_spans(19.0, 10.0, 2.0) == [(0.0, 10.0), (8.0, 18.0), (16.0, 19.0)]
    # short videos are one chunk; nothing is one span of nothing
    assert chunk_spans(7.0, 10.0, 2.0) == [(0.0, 7.0)]
    assert chunk_spans(0.0, 10.0, 2.0) == []
    assert chunk_spans(20.0, 10.0, 0.0) == [(0.0, 10.0), (10.0, 20.0)]


def test_gate_config_validates_its_parameters() -> None:
    config = ActivityGateConfig()
    assert (config.chunk_seconds, config.overlap_seconds, config.stride_seconds) == (10.0, 2.0, 8.0)
    assert config.score_threshold == 0.3
    for kwargs in (
        {"chunk_seconds": 0},
        {"overlap_seconds": 10.0},
        {"overlap_seconds": -1.0},
        {"score_threshold": 1.5},
        {"clip_fps": 0},
        {"clip_size": (0, 360)},
        {"max_windows": -1},
    ):
        with pytest.raises(ValueError):
            ActivityGateConfig(**kwargs)  # type: ignore[arg-type]


def test_activity_score_is_parsed_from_fenced_partial_or_bare_replies() -> None:
    clean = parse_activity_score(
        '{"score": 0.55, "subjects": ["white SUV", " person "], "event_count": 2, '
        '"summary": "A person gets into a white SUV."}'
    )
    assert clean == {
        "score": 0.55,
        "subjects": ["white SUV", "person"],
        "event_count": 2,
        "summary": "A person gets into a white SUV.",
    }
    fenced = parse_activity_score('```json\n{"score": 0.2, "subjects": [], "event_count": 0}\n```')
    assert (fenced["score"], fenced["summary"]) == (0.2, "")
    prose = parse_activity_score('Sure, here is the rating: {"score": 0.7, "event_count": 3}. Done.')
    assert (prose["score"], prose["event_count"]) == (0.7, 3)
    # cut off mid-string: the score is still recovered
    cut = parse_activity_score('{"score": 0.4, "subjects": ["person in a red')
    assert cut["score"] == 0.4 and cut["subjects"] == []
    # out-of-range and odd types are clamped or tolerated
    assert parse_activity_score('{"score": 1.4}')["score"] == 1.0
    assert parse_activity_score('{"score": "0.3", "subjects": "a car", "event_count": "x"}') == {
        "score": 0.3, "subjects": ["a car"], "event_count": 0, "summary": "",
    }
    with pytest.raises(ValueError):
        parse_activity_score("I cannot rate this video.")
    with pytest.raises(ValueError):
        parse_activity_score('{"score": "high"}')


def test_threshold_keeps_chunks_at_or_above_it_and_marks_the_rest() -> None:
    scored = scored_windows({0.0: 0.0, 8.0: 0.3, 16.0: 0.6, 24.0: 0.29})
    finder = ActivityWindowFinder(FakeScorer({}), ActivityGateConfig(score_threshold=0.3))

    windows = finder.build_windows(scored)

    assert [window.is_relevant for window in windows] == [False, True, True, False]
    # nothing is thrown away, so the rejected chunks and their scores stay visible
    assert [window.score for window in windows] == [0.0, 0.3, 0.6, 0.29]
    assert [window.start_seconds for window in windows] == [0.0, 8.0, 16.0, 24.0]


def test_budget_is_spent_highest_score_first() -> None:
    scored = scored_windows({0.0: 0.5, 8.0: 0.9, 16.0: 0.7, 24.0: 0.6})
    by_count = ActivityWindowFinder(FakeScorer({}), ActivityGateConfig(max_windows=2))
    by_seconds = ActivityWindowFinder(FakeScorer({}), ActivityGateConfig(max_total_seconds=26.0))

    assert [w.is_relevant for w in by_count.build_windows(scored)] == [False, True, True, False]
    # 26 s buys the two best 10 s chunks; the 0.6 tail (24-30 s, 6 s) still fits, the 0.5 one does not
    assert [w.is_relevant for w in by_seconds.build_windows(scored)] == [False, True, True, True]


def test_a_failed_chunk_is_recorded_and_never_relevant() -> None:
    scored = scored_windows({0.0: 0.9, 8.0: 0.9}, duration=18.0)
    scored[0] = replace(scored[0], error="model blew up", score=0.0)
    windows = ActivityWindowFinder(FakeScorer({})).build_windows(scored)

    assert [w.is_relevant for w in windows] == [False, True]
    assert windows[0].error == "model blew up"


def synthetic_video(path: Path, seconds: float = 21.0, fps: float = 30.0) -> VideoInfo:
    """A tiny MP4 whose frame brightness encodes the frame index."""
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 36))
    count = int(seconds * fps)
    for index in range(count):
        frame = np.full((36, 64, 3), (index * 255 // count), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return VideoInfo(str(path), fps, count, 64, 36)


def test_chunk_clips_are_written_in_one_pass_at_the_clip_frame_rate(tmp_path: Path) -> None:
    import cv2

    info = synthetic_video(tmp_path / "source.mp4")
    spans = chunk_spans(info.duration_seconds, 10.0, 2.0)

    clips = write_chunk_clips(info, spans, tmp_path / "clips", clip_size=(32, 18), clip_fps=2.0)

    assert [clip.name for clip in clips] == [
        "chunk_0.000s_10.000s.mp4", "chunk_8.000s_18.000s.mp4", "chunk_16.000s_21.000s.mp4",
    ]
    lengths = []
    for clip in clips:
        capture = cv2.VideoCapture(str(clip))
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 32
        frames = 0
        while capture.read()[0]:
            frames += 1
        capture.release()
        lengths.append(frames)
    # 10 s at 2 FPS is 20 frames (21 with the end frame), the 5 s tail about 10
    assert lengths[0] >= 20 and lengths[1] >= 20 and 10 <= lengths[2] <= 12


def test_finder_scores_every_chunk_keeps_the_active_ones_and_survives_a_failure(tmp_path: Path) -> None:
    info = synthetic_video(tmp_path / "source.mp4")
    scorer = FakeScorer({0.0: 0.1, 8.0: 0.6, 16.0: 0.4}, fail_at=16.0)
    finder = ActivityWindowFinder(
        scorer, ActivityGateConfig(score_threshold=0.3, clip_size=(32, 18)),
        clip_directory=tmp_path / "clips",
    )

    video, windows = finder.find(info.path)

    assert video.frame_count == info.frame_count
    assert [(w.start_seconds, w.end_seconds) for w in windows] == [(0.0, 10.0), (8.0, 18.0), (16.0, 21.0)]
    assert [w.score for w in windows] == [0.1, 0.6, 0.0]
    assert [w.is_relevant for w in windows] == [False, True, False]
    assert windows[1].subjects == ("person in a dark jacket",)
    assert windows[1].event_count == 1 and windows[1].summary.startswith("A person")
    assert windows[2].error == "model blew up"
    # frame ranges line up with the source so the tracker assigns frames correctly
    assert (windows[1].start_frame, windows[1].end_frame) == (240, 540)
    assert windows[2].end_frame == info.frame_count - 1
    # every chunk was shown to the model from its own clip file
    assert [Path(call[0]).name for call in scorer.calls] == [
        "chunk_0.000s_10.000s.mp4", "chunk_8.000s_18.000s.mp4", "chunk_16.000s_21.000s.mp4",
    ]
    assert all(Path(w.clip_path).is_file() for w in windows)
    assert Path(windows[0].clip_path).parent == tmp_path / "clips" / "source"


def test_activity_prompt_carries_the_rubric_and_asks_for_json_only() -> None:
    from videometa import ACTIVITY_SCORE_GUIDANCE, build_activity_prompt

    prompt = build_activity_prompt(8.0, 18.0, 2.0)

    assert ACTIVITY_SCORE_GUIDANCE in prompt
    assert "8.0s to 18.0s" in prompt and "2 frames per second" in prompt
    assert "Return JSON only" in prompt
    assert '"score"' in prompt and '"subjects"' in prompt and '"event_count"' in prompt
    # the MEVA actions the gate must not score away are named in the rubric
    for phrase in ("talks on a phone", "trunk opens or closes", "gets into or out of a vehicle",
                   "reverses", "distant person"):
        assert phrase in prompt


def test_scorer_constructor_needs_model_and_processor_together() -> None:
    from videometa import LocalQwenActivityScorer

    with pytest.raises(ValueError):
        LocalQwenActivityScorer(model=object())


# ---------------------------------------------------------------- shared


def test_resolve_video_source_accepts_local_paths() -> None:
    video = Path(__file__).with_name("2018-03-05.13-15-00.13-20-00.bus.G340.r13.avi")

    assert resolve_video_source(video) == video


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

    window = RelevantWindow(0, 5, 0, 149, 0.5, True)
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
    frames = [
        {"frame_index": index, "timestamp_seconds": index / 30, "detections": []}
        for index in range(150)
    ]

    sampled = _sample_frame_features(frames, max_frames=12)

    assert len(sampled) == 12
    assert sampled[0]["t"] == 0.0


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
        assert "one event per subject per continuous scene" in prompt.lower()
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


def test_involved_object_ids_are_reduced_to_the_bare_track_id() -> None:
    from videometa.window_annotation import _clean_events, _normalise_object_id

    # the overlay reads "person #20"; a model that copies it must still join to track "20"
    assert _normalise_object_id("person #20") == "20"
    assert _normalise_object_id("#20") == "20"
    assert _normalise_object_id(20) == "20"
    assert _normalise_object_id("20") == "20"
    assert _normalise_object_id(None) is None
    # nothing numeric to extract: left alone rather than guessed
    assert _normalise_object_id("the walker") == "the walker"

    cleaned = _clean_events([
        {
            "event_name": "Person walks toward the counter",
            "description": "A person in a dark jacket walks toward the counter.",
            "involved_objects": [
                {"id": "person #20", "label": "person", "physical_details": "dark jacket"},
                {"id": 33, "label": "person"},
            ],
        }
    ])
    assert [obj["id"] for obj in cleaned[0]["involved_objects"]] == ["20", "33"]
    assert "physical_details" not in cleaned[0]["involved_objects"][1]


def test_prompt_keeps_each_event_on_its_own_subjects() -> None:
    from videometa.window_annotation import DESCRIPTION_GUIDANCE

    # the description may not wander to bystanders or background objects...
    assert "One event, one set of subjects" in DESCRIPTION_GUIDANCE
    assert "never described or given actions of its own" in DESCRIPTION_GUIDANCE
    # ...and the ids must belong to the subjects the description is about
    assert "must point at the same subjects" in DESCRIPTION_GUIDANCE
    assert "not the box around the person already standing at the counter" in DESCRIPTION_GUIDANCE


def test_checklist_asks_for_one_event_per_scene_with_all_its_actions() -> None:
    from videometa.window_annotation import DESCRIPTION_GUIDANCE, action_guidance

    guidance = action_guidance()
    # a subject's consecutive actions are one event carrying several actions...
    assert "one event per subject per continuous scene, not one per action" in guidance
    assert "ONE event with three actions" in guidance
    assert "never repeat the same subject's movement as a second event" in guidance
    # ...and the old per-action splitting rule is gone
    assert "three events, not one" not in guidance
    # involved objects cover every person and object that is part of the action
    assert "check the overlay boxes one by one" in DESCRIPTION_GUIDANCE
    assert "people and objects alike" in DESCRIPTION_GUIDANCE
    assert "never given an invented id" in DESCRIPTION_GUIDANCE


def test_prompt_demands_the_shape_of_every_movement() -> None:
    from videometa.window_annotation import DESCRIPTION_GUIDANCE

    assert "Movement rule" in DESCRIPTION_GUIDANCE
    for phrase in (
        "keeps straight on",
        "turns left",
        "turns right",
        "makes a U-turn",
        "reverses",
        "goes round in a circle or loop",
        "walks back and forth",
        "pulls into or out of a parking bay",
    ):
        assert phrase in DESCRIPTION_GUIDANCE, phrase
    # left and right are the subject's, not the camera's
    assert "from the subject's own direction of travel" in DESCRIPTION_GUIDANCE
    assert "not from the camera's point of view" in DESCRIPTION_GUIDANCE


def test_both_prompts_make_the_model_review_its_own_events() -> None:
    from types import SimpleNamespace

    from videometa.window_annotation import (
        REVIEW_GUIDANCE,
        LVLMEventAnnotator,
        LocalQwenEventAnnotator,
    )

    window = prepared_window()
    remote = LVLMEventAnnotator._build_prompt(SimpleNamespace(feature_frames=4, action_vocabulary=None), window)
    local = LocalQwenEventAnnotator._build_window_prompt(SimpleNamespace(action_vocabulary=None), window, 4)
    for prompt in (remote, local):
        assert "review your draft critically, one event at a time" in prompt
        assert "thinking it through step by step" in prompt
        assert "Did it actually happen?" in prompt
        assert "Is it worth recording?" in prompt
        assert "Does it make sense?" in prompt
        # the review happens before the JSON instruction, and the answer is still JSON only
        assert prompt.index(REVIEW_GUIDANCE) < prompt.index("Return JSON only")
        assert "the output stays JSON only" in prompt


def test_prompt_asks_for_apparent_gender_and_forbids_same_subject_duplicates() -> None:
    from videometa.window_annotation import DESCRIPTION_GUIDANCE, PHYSICAL_DETAILS_GUIDANCE, REVIEW_GUIDANCE

    assert "apparent gender and age group" in DESCRIPTION_GUIDANCE
    assert "say 'a person' only when it cannot be judged" in DESCRIPTION_GUIDANCE
    assert "apparent gender and age group" in PHYSICAL_DETAILS_GUIDANCE
    assert "Each event is a different scene" in DESCRIPTION_GUIDANCE
    assert "never three 'walks toward' events" in DESCRIPTION_GUIDANCE
    # the same subject may still have several events when the scenes are separate
    assert "only when the scenes are genuinely separate" in DESCRIPTION_GUIDANCE
    assert "point at the same scene" in REVIEW_GUIDANCE


def test_prompt_groups_subjects_into_an_event_only_when_they_interact() -> None:
    from videometa.window_annotation import DESCRIPTION_GUIDANCE, REVIEW_GUIDANCE

    assert "Group by interaction" in DESCRIPTION_GUIDANCE
    assert "Being in the same frame at the same time is not an interaction" in DESCRIPTION_GUIDANCE
    assert "two people walking separately through the room are two events" in DESCRIPTION_GUIDANCE
    assert "If a subject acts on its own, it gets its own event with only itself listed" in DESCRIPTION_GUIDANCE
    assert "every listed subject interacts with the others" in REVIEW_GUIDANCE

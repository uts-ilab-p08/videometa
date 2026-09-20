# videometa

`videometa` turns a video into structured annotations in three independent stages:

1. Find relevant, motion-based time windows.
2. Track objects and extract their spatial boundaries.
3. Use a pluggable vision-language backend to identify events, descriptions, and involved objects.

The package does not import OpenCV, Ultralytics, or a VLM at import time. Install and
configure only the integrations used by your application.

## Installation

```bash
pip install videometa opencv-python ultralytics
```

## Usage

### Find relevant windows

```python
from videometa import MotionGateConfig, RelevantWindowFinder

finder = RelevantWindowFinder(MotionGateConfig())      # sensible defaults
video, windows = finder.find("camera.mp4")
relevant_windows = [window for window in windows if window.is_relevant]
```

The gate makes three decisions, each configurable.

**1. What counts as motion — `motion_metric`.**

| metric | what it measures | use it when |
|---|---|---|
| `"score"` | fraction of the whole frame flagged as foreground | you need the pre-0.1 behaviour |
| `"tile_peak"` | the loudest cell of `tile_grid` | objects are small but the camera is static and clean |
| `"blob_area"` | pixels in the largest connected region | you care about one coherent object, not scattered change |
| `"local"` *(default)* | the loudest cell **against that cell's own history** | mixed scenes, where a distant figure and a passing lorry must both register |

`"score"` averages over the whole frame, so a 20x20 object on a 640x360 gate is
0.17% of it and sits below any threshold that also rejects noise. `"local"`
divides each cell's activity by that cell's own spread, so motion is scored
against what is normal *there* — which is what lets a small, distant, or
peripheral event clear the same bar as a large central one.

Every sample is measured from two fused detectors: MOG2, and frame differencing
to cover MOG2's blind spot (it absorbs a stationary object into its background
within roughly `mog_history`/10 samples).

**2. Where the windows are cut — `segmentation`.**

`"events"` (default) grows each window around a run of motion: a run opens at
the threshold, stays open while it holds above `hysteresis_ratio` of it, and is
then padded by `pad_seconds`, merged with neighbours closer than
`merge_gap_seconds`, widened to `min_window_seconds` and split at
`max_window_seconds`. Short events are widened, never dropped.

`"grid"` restores the original fixed `window_seconds`/`stride_seconds` tiling,
which is the only mode that supports `motion_std_direction` for finding
unusually *still* windows.

**3. Which windows survive — the threshold and the budget.**

`motion_threshold` accepts a fixed number or a statistic name computed from that
video's own samples (warmup samples excluded):

- `"auto"` *(default)* — `median + motion_std_k * MAD`, held above a physical
  floor for the chosen metric so a still video cannot calibrate its way down
  into sensor noise.
- `"mad"` — the same robust statistic with no floor.
- `"std"` — `mean ± motion_std_k * std`. A standard deviation is inflated by the
  few very loud samples every motion trace contains, which can push the cut
  above every quiet event in a video that also holds one lorry; prefer `"mad"`.
- `"avg"` / `"median"` — the plain statistic.

Set `max_windows` or `max_total_seconds` to cap what the next stage has to read.
Windows are ranked by peak, not total, so a brief intense event is not outranked
by a long tepid one; every window is still returned, with `is_relevant` marking
the selection. `spatial_diversity=True` spreads the budget across regions of the
frame before spending it twice on the busiest one — useful when one area
dominates the motion statistics, wasteful when it does not.

```python
MotionGateConfig(
    motion_metric="local",
    segmentation="events",
    motion_threshold="auto",
    pad_seconds=2.0,          # context around each event; the main cost dial
    max_total_seconds=120,    # optional cap on what reaches the next stage
)
```

A gate can only keep less than the events themselves occupy by dropping some. On
a 5-minute MEVA clip whose 24 labelled events plus 2s padding occupy 38% of the
running time, the defaults keep 53% with every event caught, against 87-100% for
the fixed-grid gate.

When `sample_fps` is omitted (the default), every video frame is evaluated: its value
is automatically set to the source video FPS. Set `sample_fps=2` or another positive
value only when you deliberately want to sample less frequently.

Use `finder.sample_motion()` once, then `finder.calibrate(samples, thresholds)` to
compare threshold values without decoding the source video again.
`finder.resolve_motion_thresholds(samples)` returns the active lower and upper bounds.

Because `"local"` scores motion against each cell's own history, it finds what is
*unusual for this video*. On footage that is busy from start to finish, the
baseline rises to meet it — use `"tile_peak"` or `"score"` with an absolute
threshold there.

### Extract spatial object boundaries

```python
from videometa import DetectionConfig, ObjectBoundaryExtractor

extractor = ObjectBoundaryExtractor(
    DetectionConfig(
        model_path="yolo26n.pt",
        tracker="bytetrack.yaml",
        confidence_threshold=0.25,
        classes=[0, 1, 2, 3, 5, 7],  # omit or pass [] to track every YOLO class
    )
)
object_windows = extractor.extract("camera.mp4", relevant_windows, video.fps)
```

`classes` accepts YOLO class IDs such as COCO `0` for person. When omitted or
empty, tracking uses every class known to the loaded model.

Each `ObjectDetection` has pixel coordinates, normalized coordinates via
`detection.boundary.normalized(video.width, video.height)`, and a spatial description
such as `"top-left"`.

#### Identity across the video

The tracker runs once over the whole video, not once per window, so an object keeps
one `track_id` for as long as it stays visible — including across window boundaries.
What it cannot do by itself is survive a disappearance: ByteTrack matches on position,
so once an object has been out of shot longer than the tracker's buffer its track is
retired, and the same person walking back in returns under a new id.

`stitch_tracks` (on by default) rejoins those fragments after the pass. Two tracks
become one identity when they are never on screen at the same moment, share a label,
are separated by at most `stitch_max_gap_seconds`, could plausibly have travelled
between their last and first positions at `stitch_max_speed` frame-diagonals per
second, and their colour signatures correlate at least `stitch_min_similarity`.

```python
DetectionConfig(
    stitch_tracks=True,
    stitch_max_gap_seconds=30.0,   # never join things further apart than this
    stitch_min_similarity=0.7,     # colour-histogram correlation, 0..1
    stitch_max_speed=0.1,          # frame diagonals per second
    appearance_samples=12,         # crops sampled per track to build its signature
)
```

Labels are settled per identity by a confidence-weighted vote across every frame of
the track, so an object that YOLO reads as `truck` in one frame and `car` in forty
others is reported as `car` everywhere, rather than taking whichever class the last
frame happened to produce.

The appearance test is a colour histogram, not a learned re-identification model. It
suits a fixed camera and distinguishable clothing; it will merge two people in similar
dark coats, and strong lighting changes will stop it merging one person with
themselves. Raise `stitch_min_similarity` to merge less, or set `stitch_tracks=False`
to keep the raw tracker ids. The count of merges is logged at INFO.

### Add events with your VLM

Provide a small adapter around the VLM or API of your choice. It receives the source
video, window time range, and tracked-object context; it returns JSON-compatible event
records.

```python
from videometa import EventExtractor, VideoAnnotator

class MyEventBackend:
    def analyze(self, video_path, start_seconds, end_seconds, object_context):
        # Send the selected clip and object_context to your model.
        return [{
            "event_name": "Vehicle enters",
            "description": "A white sedan enters from the left.",
            "involved_objects": [{
                "id": "7",
                "label": "car",
                "physical_details": "white sedan",
            }],
        }]

annotator = VideoAnnotator(event_extractor=EventExtractor(MyEventBackend()))
annotations = annotator.annotate("camera.mp4", include_events=True)
document = annotations.to_dict()  # JSON-serializable
```

`VideoAnnotator` is the one-call façade. The three stage classes can also be used
separately, which supports parameter experiments, alternative object detectors, and
different event models.

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`videometa` was created by Juan Vargas. It is licensed under the terms of the MIT license.

## Credits

`videometa` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

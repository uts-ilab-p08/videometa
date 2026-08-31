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

finder = RelevantWindowFinder(
    MotionGateConfig(
        gate_size=(640, 360),
        window_seconds=60,
        stride_seconds=50,
        motion_threshold=0.002,  # or "avg" / "median"
    )
)
video, windows = finder.find("camera.mp4")
relevant_windows = [window for window in windows if window.is_relevant]
```

`motion_threshold` accepts a fixed number or a statistic name. `"avg"` and `"median"`
are computed from that video's sampled motion scores and used as the cutoff.
`finder.resolve_motion_threshold(samples)` returns the numeric value actually applied.

When `sample_fps` is omitted (the default), every video frame is evaluated: its value
is automatically set to the source video FPS. Set `sample_fps=2` or another positive
value only when you deliberately want to sample less frequently.

Use `finder.sample_motion()` once, then `finder.calibrate(samples, thresholds)` to
compare threshold values without decoding the source video again.

### Extract spatial object boundaries

```python
from videometa import DetectionConfig, ObjectBoundaryExtractor

extractor = ObjectBoundaryExtractor(
    DetectionConfig(
        model_path="yolo26n.pt",
        tracker="bytetrack.yaml",
        confidence_threshold=0.25,
    )
)
object_windows = extractor.extract("camera.mp4", relevant_windows, video.fps)
```

Each `ObjectDetection` has pixel coordinates, normalized coordinates via
`detection.boundary.normalized(video.width, video.height)`, and a spatial description
such as `"top-left"`. Object identities intentionally restart in each window, so the
result is reliable within a window and does not claim unsupported cross-window identity.

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

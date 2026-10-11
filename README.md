<h1 align="center">VideoMeta</h1>

<p align="center">
  <strong>Turn raw surveillance video into structured, reviewable event annotations.</strong>
</p>

<p align="center">
  <a href="https://test.pypi.org/project/videometa/"><img alt="Test PyPI" src="https://img.shields.io/pypi/v/videometa?pypiBaseUrl=https%3A%2F%2Ftest.pypi.org&label=test%20pypi&color=4c8bf5"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python 3.14+" src="https://img.shields.io/badge/python-3.14%2B-3776ab?logo=python&logoColor=white"></a>
  <a href="https://github.com/uts-ilab-p08/videometa/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-2ea44f"></a>
  <a href="https://github.com/uts-ilab-p08/videometa"><img alt="Source on GitHub" src="https://img.shields.io/badge/source-GitHub-181717?logo=github"></a>
</p>

<p align="center">
  <img alt="VideoMeta annotation pipeline: a VLM window-significance filter selects clips, a YOLO spatial feature model tracks objects, and a VLM annotator writes event annotations with element coordinates" src="https://raw.githubusercontent.com/uts-ilab-p08/videometa/main/docs/_static/annotation_pipeline.png" width="100%">
</p>

<!-- docs-body-start -->

VideoMeta takes a long, mostly uneventful video and produces a JSON document of
*what happened, when, where in the frame, and who or what was involved*. It does
so in three independent stages, shown left to right above:

| Stage | Component | Model | What it does |
|---|---|---|---|
| 1 | **Window significance filter** | Qwen3-VL (local, 4-bit) | Cuts the video into overlapping chunks and scores how much happens in each. Only chunks above a threshold go on. |
| 2 | **Spatial feature model** | YOLO26 + ByteTrack | Tracks people and vehicles through the selected windows and extracts their bounding boxes, with one identity per object across the whole video. |
| 3 | **VLM annotator** | Qwen3-VL or any hosted VLM | Reads the window clip together with the tracked-object context and writes plain-English events, descriptions and the objects involved. |

The output pairs each **event annotation** with the **element coordinates** of
the objects that took part, so a reviewer can jump from a sentence to the exact
region of the frame it describes.

VideoMeta does not import OpenCV, Ultralytics or a VLM at import time. Install
only the integrations your application uses.

---

## Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Stage 1: find relevant windows](#stage-1-find-relevant-windows)
- [Stage 2: extract spatial object boundaries](#stage-2-extract-spatial-object-boundaries)
- [Stage 3: annotate events with a VLM](#stage-3-annotate-events-with-a-vlm)
- [Output](#output)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

```bash
pip install videometa
```

Add the extras you need:

| Extra | Installs | Needed for |
|---|---|---|
| `vision` | `numpy`, `opencv-python`, `ultralytics` | Video decoding, the window filter clips, and YOLO tracking (stage 2) |
| `local` | `mlx-vlm` | The local Qwen3-VL gate and annotator on Apple silicon (stages 1 and 3) |
| `lvlm` | `openai` | Any OpenAI-compatible hosted VLM as the annotator (stage 3) |

```bash
pip install "videometa[vision,local]"   # Apple silicon, fully local pipeline
pip install "videometa[vision,lvlm]"    # YOLO locally, hosted VLM for annotation
```

To try the latest development build from Test PyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ videometa
```

## Quickstart

The three stages compose into one call through `VideoAnnotator`.

```python
from videometa import (
    ActivityGateConfig,
    ActivityWindowFinder,
    DetectionConfig,
    EventExtractor,
    LocalQwenActivityScorer,
    ObjectBoundaryExtractor,
    VideoAnnotator,
)

# Stage 1: VLM window-significance filter
scorer = LocalQwenActivityScorer("mlx-community/Qwen3-VL-8B-Instruct-4bit")
finder = ActivityWindowFinder(scorer, ActivityGateConfig(), clip_directory="clips")

# Stage 2: YOLO spatial feature model
extractor = ObjectBoundaryExtractor(DetectionConfig(model_path="yolo26l.pt"))

# Stage 3: any backend that returns event records (see below)
events = EventExtractor(MyEventBackend())

annotator = VideoAnnotator(finder, boundary_extractor=extractor, event_extractor=events)
annotations = annotator.annotate("camera.mp4", include_events=True)
document = annotations.to_dict()  # JSON-serializable
```

Each stage class also works on its own, which makes it easy to run parameter
experiments, swap in a different detector, or try another event model.

---

## Stage 1: find relevant windows

```python
from videometa import ActivityGateConfig, ActivityWindowFinder, LocalQwenActivityScorer

scorer = LocalQwenActivityScorer("mlx-community/Qwen3-VL-8B-Instruct-4bit")
finder = ActivityWindowFinder(scorer, ActivityGateConfig(), clip_directory="clips")
video, windows = finder.find("camera.mp4")
relevant_windows = [window for window in windows if window.is_relevant]
```

The gate does three things.

**1. Cut the video into fixed, overlapping chunks.** `chunk_seconds` (10) and
`overlap_seconds` (2) give chunks starting every 8 s: `0-10, 8-18, 16-26, ...`.
Each chunk begins with the last two seconds of the previous one, so an action
that straddles a boundary is whole in at least one chunk and the model sees
what led into the window. Every chunk is written as a small MP4 (`clip_size`
640x360, `clip_fps` 2, so 20 frames for 10 s) in a single pass over the source.

**2. Ask the VLM how much happens in each chunk.** The same Qwen3-VL model that
later writes the event annotations is shown the clip with a rubric
(`ACTIVITY_SCORE_GUIDANCE`) and returns a 0-1 `score` judged on how many people
and vehicles move, how much they move, and how many distinct actions or
interactions occur, plus `subjects`, an `event_count` and a one-sentence
`summary`. All four are stored on the `RelevantWindow`, so a reviewer can see
why a chunk was kept or dropped. The rubric's bands:

| score | what the model sees |
|---|---|
| 0.0 | nothing moves: empty scene, parked cars, foliage, flicker |
| 0.1-0.2 | one person or vehicle passes through, nothing else |
| 0.3-0.4 | one subject does one small thing (phone, reading, carrying, sits/stands), or two subjects move independently |
| 0.5-0.6 | a clear action or interaction: door/trunk, gets in/out, loads, two people meet or talk, vehicle stops/reverses/turns/picks up |
| 0.7-0.8 | several such actions or several interacting subjects |
| 0.9-1.0 | a busy scene with many simultaneous actions |

A pixel-motion gate cannot tell a person opening a car door from a tree in the
wind, and scores a bus in the foreground far above a distant figure texting.
The model is asked about subjects and actions, which is what the annotation
stage is looking for, and told not to let size decide the score.

**3. Keep the chunks at or above `score_threshold`.** The default 0.3 keeps
everything beyond a lone subject passing through. That is deliberate for MEVA:
its taxonomy has no "walks" or "drives" activity, but does include
`person_texts_on_phone`, `person_reads_document`, `person_heavy_carry` and
`person_sits_down`, which the rubric places at 0.3-0.4, so a 0.5 cut would drop
them along with the empty car parks. Raise it to 0.5 to keep only chunks with
a clear door, vehicle-entry, meeting or manoeuvre. `max_windows` and
`max_total_seconds` then cap the selection, highest score first. Every chunk is
still returned with `is_relevant` marking the selection, and a chunk whose
scoring call failed is kept with `error` set and `score=0`.

```python
ActivityGateConfig(
    chunk_seconds=10.0,
    overlap_seconds=2.0,
    score_threshold=0.3,      # 0.5 for clear actions only
    clip_size=(640, 360),
    clip_fps=2.0,             # must match LocalQwenActivityScorer(video_fps=...)
    max_total_seconds=None,   # optional cap on what reaches the next stage
)
```

Use `finder.score_video(path)` once to get every chunk scored, then
`finder.build_windows(scored)` under different `ActivityGateConfig` thresholds
to compare cuts without running the model again.

`ActivityScorer` is a protocol: anything with
`score(clip_path, start_seconds, end_seconds) -> {"score", "subjects",
"event_count", "summary"}` can stand in for the local model, and
`parse_activity_score()` turns a model's raw reply into that shape.
`LocalQwenActivityScorer.from_annotator(annotator)` shares weights with a
`LocalQwenEventAnnotator` in the same process instead of loading them twice.

---

## Stage 2: extract spatial object boundaries

```python
from videometa import DetectionConfig, ObjectBoundaryExtractor

extractor = ObjectBoundaryExtractor(
    DetectionConfig(
        model_path="yolo26l.pt",
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

### Identity across the video

The tracker runs once over the whole video, not once per window, so an object keeps
one `track_id` for as long as it stays visible, including across window boundaries.
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

---

## Stage 3: annotate events with a VLM

The annotator receives the window clip, its time range and the tracked-object
context from stage 2, and returns JSON-compatible event records. Two annotators
ship with the package, and a small adapter lets you plug in anything else.

| Annotator | Runs | Install extra |
|---|---|---|
| `LocalQwenEventAnnotator` | Qwen3-VL through MLX on Apple silicon | `local` |
| `LVLMEventAnnotator` | Any OpenAI-compatible vision endpoint | `lvlm` |
| Your own `EventBackend` | Anything with an `analyze(...)` method | none |

### Bring your own backend

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

annotator = VideoAnnotator(finder, event_extractor=EventExtractor(MyEventBackend()))
annotations = annotator.annotate("camera.mp4", include_events=True)
```

### How descriptions are worded

`LVLMEventAnnotator` and `LocalQwenEventAnnotator` share one instruction block,
`DESCRIPTION_GUIDANCE`, which decides how events read. It exists because of what
the model is handed: spatial features arrive as frame-grid cells (`top-left`,
`middle-center`), and a model given that vocabulary writes *"car #201 moves from
top-left to middle-left"*, a description of the picture rather than of the
scene. The block tells the model to

- write `event_name` as a plain-English phrase in sentence case (`Person loads a
  suitcase into a white SUV`), never dataset vocabulary like
  `person_opens_vehicle_door`;
- write `description` as two to four full sentences covering who is involved and
  how they look, where it happens, what occurs step by step, and how the scene
  is left afterwards;
- name things by appearance (`the white SUV`, `a person in a dark jacket`), never
  by track id;
- locate action against what is in the scene, such as a parking bay, the kerb or a
  doorway, and treat the grid cells as a hint for where to look, not as words
  to repeat;
- give direction of travel by destination or landmark, and use compass
  directions only where the scene makes them certain;
- report ordinary movement too, not just noteworthy events, while still
  returning an empty list when genuinely nothing moves.

Track ids remain available in `involved_objects[].id`. They are also stripped
from `event_name`, `description` and `physical_details` after the model replies,
so an id cannot reach a reader even if the model ignores the instruction. Ids
come from the tracker and change between runs, which makes them meaningless in
prose.

To change the house style, edit `DESCRIPTION_GUIDANCE`; both annotators follow it.

### Actions, not just narration

Left to itself, a video model answers "describe every activity" with one event
per window about the most visible movement: *Person walks across the car park*.
The door that was opened, the phone call and the reverse out of the bay in the
same ten seconds never get a sentence, so any comparison against an activity
taxonomy such as MEVA scores them as missed. `ACTION_GUIDANCE` therefore hands
both annotators a checklist, `ACTION_VOCABULARY`, of plain-English actions
(`opens a vehicle door`, `talks to another person`, `vehicle reverses`, ...),
asks for one event per subject per action, and adds an `actions` list to every
event. After the reply, checklist phrases are kept in `actions` (normalised to
the vocabulary) and anything else the model wrote goes to `other_actions`.

The checklist is a parameter, not a fixed part of the package: pass
`action_vocabulary=(...)` to either annotator with the actions your own dataset
or evaluation cares about, or `action_vocabulary=None` to drop the checklist and
let the model name actions freely. Exact duplicate events in one reply are
dropped.

Two related guards: descriptions are asked for in two to four sentences and
`physical_details` in one phrase, so the output budget goes to more events
rather than longer ones, and a reply cut off by the token limit is salvaged
(every complete event before the cut is kept) instead of costing the whole window.

### Crop the video to the action

`WindowSpatialFeatureJoiner(crop_to_activity=True)` writes the annotated MP4
from the region around the window's actors, the people plus every vehicle that
moves (a parked car is scenery), padded by `crop_padding` (20%), instead of
shrinking the whole frame. On a 1080p camera resized to 640x360 a 200 px person
becomes 70 px and a door or phone is no longer readable; the crop keeps them
near full size for the same number of video tokens. When the region would
exceed `crop_max_area` (60%) of the frame the full frame is used. The crop is
recorded in `PreparedWindowInput.crop_box` and both prompts tell the model that
the grid cells still refer to the full frame.

---

## Output

`annotations.to_dict()` returns one JSON-serializable document with four keys:

- **`video`**: path, fps, frame count and resolution;
- **`windows`**: every chunk with its activity `score`, `subjects`, `event_count`,
  `summary` and whether it was selected (`is_relevant`);
- **`object_annotations`**: per selected window, each tracked identity
  (`track_id`, voted `label`, first and last frame, spatial trajectory) and the
  per-frame bounding boxes with their pixel coordinates and grid position;
- **`events`**: the VLM's event records, each with a `name`, a `description` and
  the `involved_objects` whose ids map back to the tracked identities above.

---

## Contributing

Interested in contributing? Check out the
[contributing guidelines](https://github.com/uts-ilab-p08/videometa/blob/main/CONTRIBUTING.md).
Please note that this project is released with a
[Code of Conduct](https://github.com/uts-ilab-p08/videometa/blob/main/CONDUCT.md).
By contributing to this project, you agree to abide by its terms.

## License

VideoMeta is licensed under the terms of the
[MIT license](https://github.com/uts-ilab-p08/videometa/blob/main/LICENSE).

## Credits

VideoMeta was built by the UTS iLab P08 team. It was scaffolded with
[`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the
`py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

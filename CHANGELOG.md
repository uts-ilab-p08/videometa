# Changelog

<!--next-version-placeholder-->

## v0.0.19 (30/09/2026)

The OpenCV motion gate is replaced by a VLM activity gate. **Breaking**:
`MotionGateConfig`, `MotionSample` and `RelevantWindowFinder` are gone, and
`RelevantWindow` has new fields.

- `videometa.activity_gate`: the video is cut into fixed chunks
  (`chunk_seconds=10`) that overlap their neighbour (`overlap_seconds=2`), so
  each window carries the end of the previous one as context. Every chunk is
  written as a small MP4 (`clip_size`, `clip_fps`) in one sequential pass and
  shown to the same Qwen3-VL model that later writes the annotations, with a
  rubric (`ACTIVITY_SCORE_GUIDANCE`) asking for a 0-1 activity score from the
  number of moving subjects, how much they move, and how many distinct actions
  or interactions occur. The score, the subjects, an event count and a
  one-sentence summary are stored on every chunk.
- `ActivityGateConfig.score_threshold` (default 0.3) decides which chunks go
  on to tracking and annotation. The rubric places a lone person or vehicle
  passing through at 0.1-0.2 and one subject doing one small thing (phone,
  reading, carrying, sitting) at 0.3-0.4, so 0.3 keeps the subtle MEVA
  activities while dropping empty scenes; 0.5 keeps only clear door,
  vehicle-entry, meeting and manoeuvre chunks. `max_windows` and
  `max_total_seconds` cap the selection, highest score first.
- `ActivityWindowFinder(scorer, config, clip_directory=...)` exposes
  `score_video()` (every chunk scored, none selected) and `build_windows()`
  (threshold + budget) separately, so a threshold can be re-evaluated without
  re-running the model. A chunk whose scoring call fails is kept with
  `error` set and `score=0` instead of aborting the video.
- `LocalQwenActivityScorer` wraps the MLX model for scoring;
  `LocalQwenActivityScorer.from_annotator(annotator)` shares weights with a
  `LocalQwenEventAnnotator`, which now also accepts `model=`/`processor=`.
  `ActivityScorer` is a protocol, so any other VLM can be plugged in.
- `videometa.local_qwen` holds the one MLX loading and video-generation path
  both stages use (`load_local_qwen`, `generate_from_video`, `count_tokens`).
- `RelevantWindow` is now `(start_seconds, end_seconds, start_frame,
  end_frame, score, is_relevant, subjects, event_count, summary, clip_path,
  error)`. `peak_motion`, `mean_motion`, `motion_metric`, `relevance`,
  `focus`, `active_seconds` and `sample_count` are removed.
- `VideoAnnotator` takes a required `window_finder` (any `WindowFinder`);
  `probe_video()` replaces `RelevantWindowFinder.probe()`.

## v0.0.18 (28/09/2026)

Follow-up to 0.0.17 after a pilot on one MEVA camera: the crop never engaged
in a car park and the model padded its answer with repeated events.

- `crop_to_activity` now crops around the window's *actors*: the people plus
  every vehicle that moves, then moving tracks only, then people only, taking
  the first set that fits under `crop_max_area`. Parked cars are scenery, and
  the union of all tracked vehicles was the whole frame on every window.
- `action_vocabulary` is a constructor parameter of `LVLMEventAnnotator` and
  `LocalQwenEventAnnotator`: pass your dataset's own action list, or None to
  drop the checklist. `action_guidance(vocabulary)` builds the prompt block;
  `ACTION_VOCABULARY` stays as the default. The annotators are not tied to
  any one evaluation dataset.
- `_clean_events` drops exact duplicate events (same name and description)
  within one reply.


## v0.0.17 (28/09/2026)

Window annotation now enumerates actions instead of narrating the window, and
no longer loses a window when the model runs out of output tokens. Driven by a
MEVA coverage check where 70% of windows came back as one "person walks" event
and 10% of windows were discarded on a truncated JSON string.

- `ACTION_VOCABULARY` / `ACTION_GUIDANCE`: both prompts now carry a
  plain-English checklist of actions derived from the MEVA taxonomy (door
  open/close, gets into/out of a vehicle, talks to a person, on a phone, picks
  up / puts down, vehicle turns / stops / starts / reverses, ...), ask for one
  event per subject per action, and add an `actions: [str]` field to each
  event. `_clean_events` keeps checklist phrases in `actions` and moves anything
  else the model wrote to `other_actions`.
- `DESCRIPTION_GUIDANCE` asks for two to four sentences instead of four to
  seven, and `PHYSICAL_DETAILS_GUIDANCE` for one phrase per object, so the
  output budget is spent on more events rather than longer ones.
- Truncated model output is salvaged: `_parse_event_list` keeps every complete
  event before the cut instead of raising on the partial one. Both annotators
  use it.
- `WindowSpatialFeatureJoiner(crop_to_activity=True)` crops the annotated MP4
  to the region holding the window's tracked people and vehicles (padded by
  `crop_padding`, falling back to the full frame above `crop_max_area`) before
  resizing, so a 1080p person keeps near full size at 640x360 for the same
  video tokens. The crop is recorded in `PreparedWindowInput.crop_box`, both
  prompts tell the model the video is a crop, and detection overlays are
  offset accordingly. `_write_annotated_window_video` now returns
  `(path, crop_box)`.


## v0.0.10 (20/09/2026)

Motion gate rework. The defaults change, and `build_windows()` now returns
event-shaped windows instead of a fixed grid; pass
`MotionGateConfig(segmentation="grid", motion_metric="score", motion_threshold=0.002)`
for the previous behaviour.

- `motion_metric` selects what is thresholded: `"score"` (the old whole-frame
  mean), `"tile_peak"`, `"blob_area"`, or `"local"` (new default), which scores
  each grid cell against its own history so small or distant motion is not
  averaged away.
- `sample_motion()` fuses MOG2 with frame differencing, so an object that stops
  moving no longer disappears when MOG2 absorbs it, and records `tile_peak`,
  `blob_area`, `local_score`, `focus` and `is_warmup` on every `MotionSample`.
- `segmentation="events"` (new default) grows windows around runs of motion with
  hysteresis, padding, merging, a minimum duration and a maximum duration,
  replacing the fixed `window_seconds`/`stride_seconds` tiling.
- `motion_threshold` gains `"mad"` and `"auto"` (the default), which resist the
  inflation that one loud passage causes in a standard deviation. Threshold
  statistics now ignore warmup samples, which are zero by construction.
- `max_windows` / `max_total_seconds` cap what reaches the next stage, ranking
  windows by peak; `spatial_diversity` spreads that budget across the frame.
- `RelevantWindow` gains `motion_metric`, `relevance`, `focus`, `active_seconds`
  and a `duration_seconds` property.

Annotation descriptions read as scene descriptions rather than detector logs.

- Both annotators now share one `DESCRIPTION_GUIDANCE` block: plain-English
  event names in sentence case rather than dataset vocabulary, descriptions of
  two to four sentences covering appearance, location, sequence and outcome,
  objects named by appearance instead of track id, action located against scene
  features rather than frame edges, and no detail invented beyond what is
  visible. The frame-grid cells are still supplied but are labelled as a lookup
  hint, not description vocabulary.
- Both prompts ask for every *observable activity* rather than every *relevant
  event*, which was setting the bar high enough that busy windows came back with
  an empty event list.
- Track ids are stripped from `event_name`, `description` and
  `physical_details` after the model replies; they stay in
  `involved_objects[].id`.
- `LVLMEventAnnotator._build_prompt()` was split out of `annotate()` so the
  prompt can be inspected and tested without an API client.

Object identity is now consistent across a whole video.

- `DetectionConfig.stitch_tracks` (on by default) rejoins tracker ids that belong
  to the same object seen at different times. Candidates must never overlap in
  time, share a label, fall within `stitch_max_gap_seconds`, be reachable at
  `stitch_max_speed`, and match on a colour signature to
  `stitch_min_similarity`. Set it to `False` for the raw tracker ids.
  Defaults were chosen against real tracker output: a tenth of a frame diagonal
  of travel per second and a 0.7 colour correlation rejoined 21 of 23 flickering
  detections on a car-park clip while admitting one implausible jump.
- Each identity's label is a confidence-weighted vote over every frame of the
  track instead of whatever the last frame reported, so a class that flickers
  between `car` and `truck` settles on one answer for the whole video.
- Track ids and labels are rewritten consistently in both `TrackedObject` and
  the `ObjectDetection`s inside `FrameAnnotations`, so overlays, window
  summaries and LVLM prompts all agree.

## v0.0.9 (02/09/2026)

- `DetectionConfig.classes` can restrict YOLO tracking to selected class IDs. When omitted or empty, every class known to the loaded model is used.

## v0.0.1 (22/08/2026)

- First release of `videometa`!
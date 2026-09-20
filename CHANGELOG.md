# Changelog

<!--next-version-placeholder-->

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

- Both annotators now share one `DESCRIPTION_GUIDANCE` block: name objects by
  appearance instead of track id, locate action against scene features rather
  than frame edges, give direction of travel by destination or landmark, and
  avoid inventing detail that is not visible. The frame-grid cells are still
  supplied but are labelled as a lookup hint, not description vocabulary.
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
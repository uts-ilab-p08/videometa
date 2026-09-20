# Changelog

<!--next-version-placeholder-->

## Unreleased

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

## v0.0.9 (02/09/2026)

- `DetectionConfig.classes` can restrict YOLO tracking to selected class IDs. When omitted or empty, every class known to the loaded model is used.

## v0.0.1 (22/08/2026)

- First release of `videometa`!
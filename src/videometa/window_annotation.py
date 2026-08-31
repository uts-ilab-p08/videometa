"""Separate preparation and LVLM annotation stages for VideoMeta windows."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Sequence

from videometa.annotation import ObjectWindowAnnotations, resolve_video_source


logger = logging.getLogger(__name__)


# Prefill KV cache for Qwen3-VL-4B costs roughly 145 KB per token (36 layers,
# 8 KV heads, head_dim 128, fp16). A 66k-token prompt is ~10 GB of cache alone,
# which exceeds the Metal budget on a 24 GB Mac. Keep prompts well under this.
DEFAULT_PROMPT_TOKEN_BUDGET = 6000

# How many sampled frames of spatial JSON to send alongside the video. The model
# already sees the overlays drawn on the MP4, so this is supporting evidence
# only, not a replacement for the pixels.
DEFAULT_FEATURE_FRAMES = 8


@dataclass(frozen=True)
class PreparedWindowInput:
    """All frames and spatial features joined for one relevant video window."""

    start_seconds: float
    end_seconds: float
    object_features: tuple[dict[str, Any], ...]
    frame_features: tuple[dict[str, Any], ...]
    image_messages: tuple[dict[str, Any], ...]
    artifact_directory: str | None
    annotated_video_path: str | None


class WindowSpatialFeatureJoiner:
    """Join every window frame and its detections into an LVLM-ready payload."""

    def __init__(
        self,
        *,
        output_directory: str | Path | None = None,
        annotated_video_size: tuple[int, int] = (640, 360),
        annotated_video_fps: float = 2.0,
        annotated_video_max_frames: int = 32,
        encode_frame_images: bool = False,
    ) -> None:
        """Configure artifact output and the Qwen MP4 size and frame rate.

        ``encode_frame_images`` controls whether every window frame is also
        base64-encoded into ``image_messages``. Only the remote OpenAI-style
        path (``LVLMEventAnnotator``) needs those. The local MLX path reads the
        annotated MP4 instead, and holding ~20 MB of base64 per window for 40+
        windows wastes close to a gigabyte of host RAM for nothing.
        """
        width, height = annotated_video_size
        if width <= 0 or height <= 0:
            raise ValueError("annotated_video_size dimensions must be positive.")
        if annotated_video_fps <= 0:
            raise ValueError("annotated_video_fps must be positive.")
        if annotated_video_max_frames <= 0:
            raise ValueError("annotated_video_max_frames must be positive.")
        self.output_directory = Path(output_directory) if output_directory else None
        self.annotated_video_size = annotated_video_size
        self.annotated_video_fps = annotated_video_fps
        self.annotated_video_max_frames = annotated_video_max_frames
        self.encode_frame_images = encode_frame_images

    def prepare(
        self, video_path: str | Path, annotation: ObjectWindowAnnotations
    ) -> PreparedWindowInput:
        """Label every frame, then join the window with its spatial object features."""
        try:
            import cv2
        except ImportError as error:
            raise ImportError(
                "Preparing LVLM window inputs requires `pip install opencv-python numpy`."
            ) from error

        frame_annotations = annotation.frames
        window = annotation.window
        artifact_directory = self._artifact_directory(annotation)
        logger.info(
            "Preparing window %.3fs-%.3fs with %d frames",
            window.start_seconds,
            window.end_seconds,
            len(frame_annotations),
        )
        object_features = tuple(
            {
                "track_id": item.track_id,
                "label": item.label,
                "average_confidence": round(item.average_confidence, 2),
                "spatial_trajectory": list(item.spatial_trajectory),
            }
            for item in annotation.objects
        )
        if not frame_annotations:
            return PreparedWindowInput(
                window.start_seconds,
                window.end_seconds,
                object_features,
                (),
                (),
                str(artifact_directory) if artifact_directory else None,
                None,
            )

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise OSError(f"Cannot open video: {video_path}")

        image_messages: list[dict[str, Any]] = []
        frame_features: list[dict[str, Any]] = []
        try:
            for frame_annotation, frame in _iter_source_frames(capture, frame_annotations):
                annotated_frame, detections = _draw_detections(
                    frame, frame_annotation.detections
                )
                frame_features.append(
                    {
                        "frame_index": frame_annotation.frame_index,
                        "timestamp_seconds": round(frame_annotation.timestamp_seconds, 2),
                        "detections": detections,
                    }
                )
                if not (artifact_directory or self.encode_frame_images):
                    continue
                ok, encoded = cv2.imencode(".jpg", annotated_frame)
                if not ok:
                    continue
                image_bytes = encoded.tobytes()
                if artifact_directory:
                    (
                        artifact_directory
                        / f"frame_{frame_annotation.frame_index:06d}.jpg"
                    ).write_bytes(image_bytes)
                if self.encode_frame_images:
                    image_messages.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,"
                                + base64.b64encode(image_bytes).decode("ascii")
                            },
                        }
                    )
        finally:
            capture.release()

        if artifact_directory:
            (artifact_directory / "spatial_features.json").write_text(
                json.dumps(frame_features, indent=2), encoding="utf-8"
            )
        annotated_video_path = _write_annotated_window_video(
            str(video_path),
            frame_annotations,
            artifact_directory,
            target_size=self.annotated_video_size,
            target_fps=self.annotated_video_fps,
            max_frames=self.annotated_video_max_frames,
        )
        prepared = PreparedWindowInput(
            window.start_seconds,
            window.end_seconds,
            object_features,
            tuple(frame_features),
            tuple(image_messages),
            str(artifact_directory) if artifact_directory else None,
            str(annotated_video_path) if annotated_video_path else None,
        )
        logger.info(
            "Prepared window %.3fs-%.3fs (%d frames, artifacts: %s)",
            window.start_seconds,
            window.end_seconds,
            len(frame_features),
            prepared.artifact_directory or "disabled",
        )
        return prepared

    def _artifact_directory(self, annotation: ObjectWindowAnnotations) -> Path | None:
        if self.output_directory is None:
            return None
        window = annotation.window
        directory = self.output_directory / (
            f"window_{window.start_seconds:.3f}s_{window.end_seconds:.3f}s"
        )
        directory.mkdir(parents=True, exist_ok=True)
        return directory


class LVLMEventAnnotator:
    """Return event annotations from a pre-joined ``PreparedWindowInput``."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        feature_frames: int = DEFAULT_FEATURE_FRAMES,
    ) -> None:
        if not base_url or not api_key:
            raise ValueError("base_url and api_key are required")
        try:
            from openai import OpenAI
        except ImportError as error:
            raise ImportError("LVLM annotation requires `pip install openai`.") from error
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.feature_frames = feature_frames

    def annotate(self, prepared_input: PreparedWindowInput) -> list[dict[str, Any]]:
        """Send a prepared window's images and spatial context to the LVLM."""
        prompt = (
            f"These ordered images cover video time {prepared_input.start_seconds:.2f}s "
            f"to {prepared_input.end_seconds:.2f}s. The overlays show detector "
            "boundaries labelled as `class #track_id`. "
            "Use the images and the spatial features to identify visible events. "
            "Detector labels are supporting evidence, not certain visual facts. "
            'Return JSON only: {"events": [{"event_name": str, "description": str, '
            '"involved_objects": [{"id": str, "label": str, "physical_details": str}]}]}.\n\n'
            "Tracked-object features:\n"
            f"{_compact_json(_compact_object_features(prepared_input.object_features))}\n\n"
            "Sampled per-frame features:\n"
            f"{_compact_json(_sample_frame_features(prepared_input.frame_features, self.feature_frames))}"
        )
        content = list(prepared_input.image_messages)
        content.append({"type": "text", "text": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=800,
        )
        parsed = json.loads(_strip_json_fence(response.choices[0].message.content))
        if not isinstance(parsed, dict) or not isinstance(parsed.get("events"), list):
            raise ValueError("LVLM response must be a JSON object containing an events list.")
        return parsed["events"]


class LocalQwenEventAnnotator:
    """Annotate a prepared, boundary-annotated window using local MLX Qwen3-VL."""

    def __init__(
        self,
        model_id: str = "mlx-community/Qwen3-VL-4B-Instruct-4bit",
        *,
        video_fps: float = 2.0,
        max_tokens: int = 600,
        prompt_token_budget: int = DEFAULT_PROMPT_TOKEN_BUDGET,
        feature_frames: int = DEFAULT_FEATURE_FRAMES,
    ) -> None:
        try:
            from mlx_vlm import load
        except ImportError as error:
            raise ImportError(
                "Local Qwen annotation requires `pip install mlx-vlm`."
            ) from error
        self.model, self.processor = load(model_id)
        self.video_fps = video_fps
        self.max_tokens = max_tokens
        self.prompt_token_budget = prompt_token_budget
        self.feature_frames = feature_frames
        logger.info("Loaded local Qwen model: %s", model_id)

    def annotate(
        self, prepared_input: PreparedWindowInput | str | Path
    ) -> list[dict[str, Any]]:
        """Annotate a prepared window, local video path, or public HTTP(S) video URL."""
        if isinstance(prepared_input, PreparedWindowInput):
            return self._annotate_prepared_window(prepared_input)
        return self._annotate_video_source(prepared_input)

    def _annotate_prepared_window(
        self, prepared_input: PreparedWindowInput
    ) -> list[dict[str, Any]]:
        """Pass a persisted annotated window video and its spatial context to Qwen3-VL."""
        if prepared_input.annotated_video_path is None:
            raise ValueError(
                "Local Qwen requires an annotated window video. Configure "
                "WindowSpatialFeatureJoiner with output_directory."
            )
        logger.info(
            "Annotating prepared window %.3fs-%.3fs with local Qwen",
            prepared_input.start_seconds,
            prepared_input.end_seconds,
        )
        prompt = self._fit_window_prompt(prepared_input)
        return self._generate_events(prepared_input.annotated_video_path, prompt)

    def _annotate_video_source(self, video_source: str | Path) -> list[dict[str, Any]]:
        """Download a URL if needed, then annotate a source video without spatial context."""
        video_path = resolve_video_source(video_source)
        logger.info("Annotating video source with local Qwen: %s", video_path)
        return self._generate_events(
            str(video_path),
            (
                "Analyze this video and identify every relevant event. For each event, "
                "provide a concise event name and description, then list the involved "
                "objects with a label and short visual description. "
                'Return JSON only as a list of events: [{"event_name": str, '
                '"description": str, "involved_objects": [{"id": str, "label": str, '
                '"physical_details": str}]}].'
            ),
        )

    def _fit_window_prompt(self, prepared_input: PreparedWindowInput) -> str:
        """Shrink the spatial-feature sample until the prompt fits the token budget.

        This is the guard that prevents the Metal prefill OOM. Serialising all
        150 frames of a 5-second window at 30 FPS produces ~66,000 tokens, and
        the KV cache for that alone exceeds the GPU budget on a 24 GB Mac.
        """
        frame_count = self.feature_frames
        while True:
            prompt = self._build_window_prompt(prepared_input, frame_count)
            token_count = self._count_tokens(prompt)
            if token_count <= self.prompt_token_budget:
                logger.info(
                    "Window prompt: %d tokens from %d sampled frames",
                    token_count,
                    frame_count,
                )
                return prompt
            if frame_count <= 1:
                raise ValueError(
                    f"Window prompt is {token_count} tokens even with a single sampled "
                    f"frame, over the {self.prompt_token_budget} budget. Reduce the "
                    "number of tracked objects or raise prompt_token_budget."
                )
            frame_count = max(1, frame_count // 2)
            logger.warning(
                "Window prompt was %d tokens; retrying with %d sampled frames",
                token_count,
                frame_count,
            )

    def _build_window_prompt(
        self, prepared_input: PreparedWindowInput, frame_count: int
    ) -> str:
        return (
            "Analyze this one annotated video window independently. Identify every "
            "relevant event that occurs within this window only. Ignore static "
            "background objects and do not infer events outside the displayed time. "
            "For each event, provide a concise event name and description, then list "
            "only the objects involved. Each involved object must include its "
            "detector track ID when available, object label, and a short description "
            "of visible physical details or its role in the event. The overlays show "
            "detector boundaries labelled as `class #track_id`. "
            "Use the video and the spatial features as supporting evidence; do not "
            "treat detector labels as certain visual facts. "
            'Return JSON only as a list of events: [{"event_name": str, '
            '"description": str, "involved_objects": [{"id": str, "label": str, '
            '"physical_details": str}]}].\n\n'
            "Tracked-object features (id, label, mean confidence, positions visited):\n"
            f"{_compact_json(_compact_object_features(prepared_input.object_features))}\n\n"
            "Sampled per-frame features (t = seconds, d = detections as id/label/position):\n"
            f"{_compact_json(_sample_frame_features(prepared_input.frame_features, frame_count))}"
        )

    def _count_tokens(self, prompt: str) -> int:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        return len(tokenizer(prompt).input_ids)

    def _generate_events(self, video_path: str, prompt: str) -> list[dict[str, Any]]:
        """Run local Qwen3-VL over a local video path and parse its JSON event list."""
        try:
            import mlx.core as mx
            from mlx_vlm import generate
        except ImportError as error:
            raise ImportError(
                "Local Qwen annotation requires `pip install mlx-vlm`."
            ) from error

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        formatted_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        try:
            output = generate(
                self.model,
                self.processor,
                formatted_prompt,
                video=[video_path],
                fps=self.video_fps,
                max_tokens=self.max_tokens,
                verbose=False,
            )
        finally:
            # MLX pools freed Metal buffers. Across dozens of windows in one
            # loop that pool grows until allocation fails, so release it after
            # every call whether or not generation succeeded.
            mx.clear_cache()
        output_text = output.text if hasattr(output, "text") else str(output)
        parsed = json.loads(_strip_json_fence(output_text))
        if isinstance(parsed, dict):
            parsed = parsed.get("events")
        if not isinstance(parsed, list):
            raise ValueError("Local Qwen response must be a JSON event list.")
        logger.info("Local Qwen returned %d events for %s", len(parsed), video_path)
        return parsed


def _compact_json(value: Any) -> str:
    """Serialize without the whitespace that inflates the prompt token count."""
    return json.dumps(value, separators=(",", ":"))


def _compact_object_features(
    object_features: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop long keys and full-precision floats from the tracked-object summary."""
    return [
        {
            "id": item["track_id"],
            "l": item["label"],
            "c": round(float(item["average_confidence"]), 2),
            "p": list(item["spatial_trajectory"]),
        }
        for item in object_features
    ]


def _sample_frame_features(
    frame_features: Sequence[dict[str, Any]],
    max_frames: int = DEFAULT_FEATURE_FRAMES,
) -> list[dict[str, Any]]:
    """Return a short, compact timeline sample of the per-frame detections.

    Two things matter here. First the frame count: a 5-second window at 30 FPS
    holds 150 frames, and sending them all is what blows up the prefill. Second
    the per-record size: the pixel boxes are already drawn on the video the
    model is watching, so repeating them as full-precision floats costs
    thousands of tokens and adds nothing.
    """
    if max_frames <= 0:
        return []
    sampled = list(frame_features)
    if len(sampled) > max_frames:
        step = max(1, len(sampled) // max_frames)
        sampled = sampled[::step][:max_frames]
    return [
        {
            "t": round(float(frame["timestamp_seconds"]), 2),
            "d": [
                {
                    "id": detection["track_id"],
                    "l": detection["label"],
                    "p": detection["spatial_position"],
                }
                for detection in frame["detections"]
            ],
        }
        for frame in sampled
    ]


def _iter_source_frames(capture: Any, frame_annotations: Sequence[Any]):
    """Yield ``(annotation, frame)`` pairs, seeking once and then reading forward.

    ``capture.set(CAP_PROP_POS_FRAMES, ...)`` per frame forces a keyframe seek
    and decode on every iteration, which on a long AVI is far slower than
    reading sequentially through a contiguous window.
    """
    import cv2

    wanted = sorted(frame_annotations, key=lambda item: item.frame_index)
    if not wanted:
        return
    position = wanted[0].frame_index
    capture.set(cv2.CAP_PROP_POS_FRAMES, position)
    for annotation in wanted:
        while position < annotation.frame_index:
            if not capture.grab():
                return
            position += 1
        ok, frame = capture.read()
        position += 1
        if not ok:
            return
        yield annotation, frame


def _select_frames_by_time(
    frame_annotations: Sequence[Any], target_fps: float, max_frames: int
) -> list[Any]:
    """Pick annotations spaced by wall-clock time rather than by list index.

    Index striding assumes ``frame_annotations`` are consecutive source frames.
    That happens to be true today, but it breaks silently the moment the motion
    or detection stage subsamples. Selecting on ``timestamp_seconds`` is correct
    either way.
    """
    if not frame_annotations:
        return []
    interval = 1.0 / target_fps
    selected: list[Any] = []
    next_timestamp: float | None = None
    for annotation in frame_annotations:
        timestamp = float(annotation.timestamp_seconds)
        if next_timestamp is None or timestamp >= next_timestamp:
            selected.append(annotation)
            next_timestamp = timestamp + interval
        if len(selected) >= max_frames:
            break
    return selected


def _write_annotated_window_video(
    video_path: str,
    frame_annotations: Sequence[Any],
    artifact_directory: Path | None,
    *,
    target_size: tuple[int, int] | None = None,
    target_fps: float | None = None,
    max_frames: int = 32,
) -> Path | None:
    """Persist an annotated MP4 for a window when artifact output is enabled."""
    if artifact_directory is None or not frame_annotations:
        return None
    import cv2

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise OSError(f"Cannot open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    fps = min(source_fps, target_fps) if target_fps else source_fps
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width, height = target_size or (source_width, source_height)
    scale_x = width / source_width
    scale_y = height / source_height
    output_path = artifact_directory / "annotated_window.mp4"
    selected = _select_frames_by_time(frame_annotations, fps, max_frames)
    logger.info(
        "Writing annotated window video at %dx%d, %.2f FPS, %d frames: %s",
        width,
        height,
        fps,
        len(selected),
        output_path,
    )
    writer = _open_mp4_writer(output_path, fps, (width, height))
    if writer is None:
        capture.release()
        raise OSError(f"Cannot create annotated window video: {output_path}")
    written = 0
    try:
        for frame_annotation, frame in _iter_source_frames(capture, selected):
            if (width, height) != (source_width, source_height):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            annotated, _ = _draw_detections(
                frame,
                frame_annotation.detections,
                scale_x=scale_x,
                scale_y=scale_y,
            )
            writer.write(annotated)
            written += 1
    finally:
        writer.release()
        capture.release()
    if written == 0:
        raise OSError(f"Annotated window video is empty: {output_path}")
    logger.info("Wrote annotated window video (%d frames): %s", written, output_path)
    return output_path


def _open_mp4_writer(output_path: Path, fps: float, size: tuple[int, int]):
    """Open an MP4 writer using H.264 so macOS/QuickTime can play the file.

    OpenCV's default ``mp4v`` codec is MPEG-4 Part 2. Finder, QuickTime, and
    Cursor render that as a green screen even though the frames are valid.
    """
    import cv2

    for codec in ("avc1", "H264", "mp4v"):
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*codec), fps, size
        )
        if writer.isOpened():
            logger.info("Using %s codec for %s", codec, output_path)
            return writer
        writer.release()
    return None


def _overlay_style(frame: Any) -> tuple[float, int, int]:
    """Keep overlays small so objects remain visible after 640x360 downscale."""
    height = int(frame.shape[0])
    font_scale = max(0.28, min(0.42, height / 1100.0))
    text_thickness = 1
    box_thickness = 1 if height <= 720 else 2
    return font_scale, text_thickness, box_thickness


def _draw_detections(
    frame: Any,
    detections: Sequence[Any],
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> tuple[Any, list[dict[str, Any]]]:
    import cv2

    annotated = frame.copy()
    frame_height, frame_width = annotated.shape[:2]
    font_scale, text_thickness, box_thickness = _overlay_style(annotated)
    border_color = (255, 255, 255)
    text_color = (0, 0, 0)
    pad = 2
    features: list[dict[str, Any]] = []
    for detection in detections:
        left, top, right, bottom = (
            round(detection.boundary.left * scale_x),
            round(detection.boundary.top * scale_y),
            round(detection.boundary.right * scale_x),
            round(detection.boundary.bottom * scale_y),
        )
        cv2.rectangle(annotated, (left, top), (right, bottom), border_color, box_thickness)
        label = f"{detection.label} #{detection.track_id}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )
        text_left = max(pad, min(left, frame_width - label_width - pad))
        label_top = top - label_height - baseline - (2 * pad)
        if label_top < 0:
            text_baseline = min(frame_height - pad, top + label_height + pad)
        else:
            text_baseline = top - pad
        cv2.rectangle(
            annotated,
            (text_left - pad, text_baseline - label_height - pad),
            (text_left + label_width + pad, text_baseline + baseline + pad),
            border_color,
            thickness=-1,
        )
        cv2.putText(
            annotated,
            label,
            (text_left, text_baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            text_thickness,
            lineType=cv2.LINE_AA,
        )
        features.append(
            {
                "track_id": detection.track_id,
                "label": detection.label,
                "confidence": round(float(detection.confidence), 2),
                "box_xyxy": [
                    round(detection.boundary.left),
                    round(detection.boundary.top),
                    round(detection.boundary.right),
                    round(detection.boundary.bottom),
                ],
                "spatial_position": detection.spatial_description,
            }
        )
    return annotated, features


def _strip_json_fence(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        return content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return content
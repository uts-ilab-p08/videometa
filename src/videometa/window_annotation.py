"""Separate preparation and LVLM annotation stages for VideoMeta windows."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Sequence

from videometa.annotation import ObjectWindowAnnotations, resolve_video_source


logger = logging.getLogger(__name__)


#: Shared description rules for every annotator prompt.
#:
#: The detector's own vocabulary is the problem this text exists to solve.
#: Spatial features reach the model as grid cells -- "top-left",
#: "middle-center" -- and a model handed that vocabulary writes "car #201 moves
#: from top-left to middle-left", which describes the picture rather than the
#: scene. These rules push the wording back towards what someone watching the
#: footage would actually say.
DESCRIPTION_GUIDANCE = (
    "You are a professional video annotator producing surveillance event "
    "records. Write for a reader who cannot see the footage and is not a "
    "technical specialist.\n"
    "\n"
    "event_name: a short, natural phrase in plain English and sentence case "
    "that anyone understands at a glance, such as 'Person loads a suitcase "
    "into a white SUV' or 'Silver hatchback reverses out of a parking bay'. "
    "Never use snake_case, ALL_CAPS, dataset class names or detector "
    "vocabulary such as 'person_opens_vehicle_door' or 'OBJECT_TRANSFER'.\n"
    "\n"
    "description: two to four complete sentences, written as a chronological "
    "account. Cover, briefly and in this order where it applies:\n"
    "  1. The setting and starting state: where each subject is and what it "
    "is doing when the window opens.\n"
    "  2. Each subject's appearance: colour (with shade, such as 'dark navy' "
    "or 'pale grey'), size relative to its surroundings ('a small child', "
    "'a full-size van'), shape or build, clothing, and anything carried, "
    "worn or towed.\n"
    "  3. Movement, step by step: the path taken, direction of travel by "
    "landmark, pace (standing, strolling, hurrying, creeping, accelerating, "
    "braking), changes of pace or direction, stops and starts, and how far "
    "the subject travels relative to the scene.\n"
    "  4. Interactions: who or what each subject approaches, follows, avoids, "
    "hands something to, opens, enters, exits, loads, waits for or passes, "
    "including relative position while doing so ('beside', 'behind', 'in "
    "front of', 'alongside') and the order in which things happen.\n"
    "  5. Timing cues from the clip where they can be judged: brief, "
    "sustained, or repeated actions, and roughly how far into the window a "
    "step happens ('at the start', 'midway', 'towards the end').\n"
    "  6. The end state: where every subject is and what has changed in the "
    "scene afterwards.\n"
    "Name subjects by observable appearance: 'the white SUV', 'a person in a "
    "dark jacket carrying a suitcase'. When several similar subjects appear, "
    "tell them apart consistently by appearance or position for the whole "
    "description.\n"
    "\n"
    "Rules for both fields:\n"
    "- Never write a track id, a '#' or the word 'track'. Identifiers belong "
    "only in involved_objects[].id.\n"
    "- Locate the action against features visible in the scene: the road, the "
    "kerb, a parking bay, a doorway, the building entrance, an adjacent "
    "vehicle. The grid cells in the features below ('top-left', "
    "'middle-center') are frame coordinates provided to help you locate the "
    "subject. Do not reproduce them in the written record.\n"
    "- State direction of travel by destination or landmark: 'reverses out of "
    "a parking bay and departs along the access road', never 'moves from top "
    "to bottom'. Use a compass bearing only where the scene establishes it "
    "with certainty.\n"
    "- Report every observable activity involving a person or a vehicle, "
    "including ordinary movement such as someone walking through the scene or "
    "a vehicle driving past. Do not limit yourself to unusual or noteworthy "
    "events.\n"
    "- Report only what is visible. Where a colour or type cannot be "
    "determined, stay general ('a dark hatchback') rather than speculate. If "
    "genuinely nothing moves in this window, return an empty list rather than "
    "inventing an event.\n"
    "\n"
    "Acceptable:\n"
    "  event_name: 'Person loads a suitcase into a white SUV'\n"
    "  description: 'A white full-size SUV is parked nose-in to a bay beside "
    "the building entrance with its tailgate raised. A tall person in a dark "
    "navy jacket walks along the kerb from the left carrying a large black "
    "suitcase, lifts it into the open boot and closes the tailgate. They then "
    "walk round to the driver's door, and the SUV stays parked with the boot "
    "closed.'\n"
    "Not acceptable:\n"
    "  event_name: 'person_unloads_vehicle'\n"
    "  description: 'car #201 moves from top-left to middle-left.'"
)

#: What `physical_details` should hold for each involved object.
PHYSICAL_DETAILS_GUIDANCE = (
    "For each involved object, physical_details is one short phrase: colour "
    "(with shade), type, make or garment style, approximate size relative to "
    "the scene, and anything it carries, wears or tows. Movement and "
    "interactions belong in the description, not here."
)

#: Plain-English actions the annotator must look for and name when it sees
#: them. The list is derived from the MEVA activity taxonomy so that a
#: downstream comparison against MEVA ground truth meets the same vocabulary,
#: and it is phrased the way a person would say it rather than as class names.
#:
#: This exists because an open-ended "describe every activity" prompt comes
#: back with one "person walks across the car park" event per window: the
#: model summarises the most visible movement and never names the door being
#: opened, the phone call, or the reverse out of the bay that happened in the
#: same ten seconds.
ACTION_VOCABULARY = (
    # people and vehicles
    "opens a vehicle door",
    "closes a vehicle door",
    "gets into a vehicle",
    "gets out of a vehicle",
    "opens a trunk or tailgate",
    "closes a trunk or tailgate",
    "loads something into a vehicle",
    "unloads something from a vehicle",
    # people
    "talks to another person",
    "talks on a phone",
    "texts or looks at a phone",
    "reads a document",
    "picks up an object",
    "puts down an object",
    "carries a heavy object",
    "hands an object to another person",
    "hugs another person",
    "shakes hands with or touches another person",
    "buys something or pays at a counter",
    "uses a laptop",
    "opens a building door",
    "closes a building door",
    "enters through a doorway",
    "exits through a doorway",
    "sits down",
    "stands up",
    "rides a bicycle",
    "walks through the scene",
    # vehicles
    "vehicle turns left",
    "vehicle turns right",
    "vehicle makes a U-turn",
    "vehicle stops",
    "vehicle starts moving",
    "vehicle reverses",
    "vehicle drops off a person",
    "vehicle picks up a person",
    "vehicle drives through the scene",
)

ACTION_GUIDANCE = (
    "Before writing, go through this checklist for EVERY person and EVERY "
    "vehicle in the window and decide which of these actions you can actually "
    "see: "
    + "; ".join(ACTION_VOCABULARY)
    + ".\n"
    "Return one event per subject per action. A person who gets out of a car, "
    "closes its door and then talks to someone is three events, not one. "
    "Ordinary walking or driving is also an event, but never let it stand in "
    "for a more specific action that is visible in the same window.\n"
    "Each event carries an actions list holding the checklist phrases it "
    "shows, copied exactly, and the description must state each of those "
    "actions in words. Do not list an action you cannot see."
)

#: Legend for the compact spatial features, framed so the grid vocabulary reads
#: as a lookup hint rather than as description material.
FEATURE_LEGEND = (
    "Tracked objects (id, l = label, c = mean confidence, p = frame-grid cells "
    "visited, for locating the object in the picture):"
)
FRAME_LEGEND = (
    "Sampled frames (t = seconds, d = detections as id / l = label / "
    "p = frame-grid cell):"
)


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
    #: (left, top, right, bottom) source-frame pixels the annotated video shows,
    #: or None when it shows the whole frame.
    crop_box: tuple[int, int, int, int] | None = None


def _crop_note(prepared_input: PreparedWindowInput) -> str:
    """One sentence telling the model the video is a crop, so grid cells still make sense."""
    if getattr(prepared_input, "crop_box", None) is None:
        return ""
    return (
        "The video is cropped to the part of the camera frame where the tracked "
        "subjects are, so they appear larger than in the full frame; the "
        "frame-grid cells in the features below still refer to the full frame. "
    )


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
        crop_to_activity: bool = False,
        crop_padding: float = 0.2,
        crop_max_area: float = 0.6,
    ) -> None:
        """Configure artifact output and the Qwen MP4 size and frame rate.

        ``crop_to_activity`` crops the annotated MP4 to the region holding the
        window's tracked people and vehicles before resizing it, instead of
        shrinking the whole frame. A 1080p camera resized to 640x360 turns a
        200 px person into a 70 px one, and door, phone and trunk actions are
        no longer readable at that size; cropping keeps them near full size
        for the same number of video tokens. ``crop_padding`` widens the
        region by that fraction on each axis, and when the padded region would
        exceed ``crop_max_area`` of the frame the whole frame is used because
        cropping would buy nothing. The crop is recorded in
        ``PreparedWindowInput.crop_box``.

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
        if not 0 <= crop_padding <= 2:
            raise ValueError("crop_padding must be within [0, 2].")
        if not 0 < crop_max_area <= 1:
            raise ValueError("crop_max_area must be within (0, 1].")
        self.crop_to_activity = crop_to_activity
        self.crop_padding = crop_padding
        self.crop_max_area = crop_max_area

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
        annotated_video_path, crop_box = _write_annotated_window_video(
            str(video_path),
            frame_annotations,
            artifact_directory,
            target_size=self.annotated_video_size,
            target_fps=self.annotated_video_fps,
            max_frames=self.annotated_video_max_frames,
            crop_to_activity=self.crop_to_activity,
            crop_padding=self.crop_padding,
            crop_max_area=self.crop_max_area,
        )
        prepared = PreparedWindowInput(
            window.start_seconds,
            window.end_seconds,
            object_features,
            tuple(frame_features),
            tuple(image_messages),
            str(artifact_directory) if artifact_directory else None,
            str(annotated_video_path) if annotated_video_path else None,
            crop_box,
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

    def _build_prompt(self, prepared_input: PreparedWindowInput) -> str:
        return (
            f"These ordered images cover video time {prepared_input.start_seconds:.2f}s "
            f"to {prepared_input.end_seconds:.2f}s. The overlays show detector "
            "boundaries labelled as `class #track_id`. "
            "Use the images and the spatial features to identify every observable "
            "activity. "
            "Detector labels are supporting evidence, not certain visual facts. "
            f"{_crop_note(prepared_input)}\n\n"
            f"{DESCRIPTION_GUIDANCE}\n\n"
            f"{ACTION_GUIDANCE}\n\n"
            f"{PHYSICAL_DETAILS_GUIDANCE}\n\n"
            'Return JSON only: {"events": [{"event_name": str, "description": str, '
            '"actions": [str], "involved_objects": [{"id": str, "label": str, '
            '"physical_details": str}]}]}.\n\n'
            f"{FEATURE_LEGEND}\n"
            f"{_compact_json(_compact_object_features(prepared_input.object_features))}\n\n"
            f"{FRAME_LEGEND}\n"
            f"{_compact_json(_sample_frame_features(prepared_input.frame_features, self.feature_frames))}"
        )

    def annotate(self, prepared_input: PreparedWindowInput) -> list[dict[str, Any]]:
        """Send a prepared window's images and spatial context to the LVLM."""
        prompt = self._build_prompt(prepared_input)
        content = list(prepared_input.image_messages)
        content.append({"type": "text", "text": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=1600,
        )
        return _clean_events(_parse_event_list(response.choices[0].message.content))


class LocalQwenEventAnnotator:
    """Annotate a prepared, boundary-annotated window using local MLX Qwen3-VL."""

    def __init__(
        self,
        model_id: str = "mlx-community/Qwen3-VL-4B-Instruct-4bit",
        *,
        video_fps: float = 2.0,
        max_tokens: int = 1200,
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
                "provide a concise event name and a detailed description, then list the "
                "involved objects with a label. "
                f"{PHYSICAL_DETAILS_GUIDANCE} "
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
            "observable activity that occurs within this window only. Ignore static "
            "background objects and do not infer events outside the displayed time. "
            "For each event, give a concise event name and description, then list "
            "only the objects involved. The overlays show detector boundaries "
            "labelled as `class #track_id`. Use the video and the spatial features "
            "as supporting evidence; do not treat detector labels as certain "
            f"visual facts. {_crop_note(prepared_input)}\n\n"
            f"{DESCRIPTION_GUIDANCE}\n\n"
            f"{ACTION_GUIDANCE}\n\n"
            "Each involved object carries its detector track id in the id field and "
            "its label. "
            f"{PHYSICAL_DETAILS_GUIDANCE}\n\n"
            'Return JSON only as a list of events: [{"event_name": str, '
            '"description": str, "actions": [str], "involved_objects": [{"id": str, '
            '"label": str, "physical_details": str}]}].\n\n'
            f"{FEATURE_LEGEND}\n"
            f"{_compact_json(_compact_object_features(prepared_input.object_features))}\n\n"
            f"{FRAME_LEGEND}\n"
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
        parsed = _parse_event_list(output_text)
        logger.info("Local Qwen returned %d events for %s", len(parsed), video_path)
        return _clean_events(parsed)


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
    crop_to_activity: bool = False,
    crop_padding: float = 0.2,
    crop_max_area: float = 0.6,
) -> tuple[Path | None, tuple[int, int, int, int] | None]:
    """Persist an annotated MP4 for a window when artifact output is enabled.

    Returns the video path and the source-frame crop it shows: both None when
    nothing was written, and the crop None when the whole frame was used.
    """
    if artifact_directory is None or not frame_annotations:
        return None, None
    import cv2

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise OSError(f"Cannot open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    fps = min(source_fps, target_fps) if target_fps else source_fps
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width, height = target_size or (source_width, source_height)
    crop_box = None
    if crop_to_activity:
        crop_box = _activity_crop(
            frame_annotations,
            (source_width, source_height),
            width / height,
            padding=crop_padding,
            max_area=crop_max_area,
        )
    if crop_box is None:
        offset_x, offset_y = 0, 0
        region_width, region_height = source_width, source_height
    else:
        offset_x, offset_y = crop_box[0], crop_box[1]
        region_width = crop_box[2] - crop_box[0]
        region_height = crop_box[3] - crop_box[1]
    scale_x = width / region_width
    scale_y = height / region_height
    output_path = artifact_directory / "annotated_window.mp4"
    selected = _select_frames_by_time(frame_annotations, fps, max_frames)
    logger.info(
        "Writing annotated window video at %dx%d, %.2f FPS, %d frames, crop %s: %s",
        width,
        height,
        fps,
        len(selected),
        crop_box or "none",
        output_path,
    )
    writer = _open_mp4_writer(output_path, fps, (width, height))
    if writer is None:
        capture.release()
        raise OSError(f"Cannot create annotated window video: {output_path}")
    written = 0
    try:
        for frame_annotation, frame in _iter_source_frames(capture, selected):
            if crop_box is not None:
                frame = frame[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
            if (width, height) != (region_width, region_height):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            annotated, _ = _draw_detections(
                frame,
                frame_annotation.detections,
                scale_x=scale_x,
                scale_y=scale_y,
                offset_x=offset_x,
                offset_y=offset_y,
            )
            writer.write(annotated)
            written += 1
    finally:
        writer.release()
        capture.release()
    if written == 0:
        raise OSError(f"Annotated window video is empty: {output_path}")
    logger.info("Wrote annotated window video (%d frames): %s", written, output_path)
    return output_path, crop_box


#: Detector labels that mark where the activity is when cropping a window.
_CROP_LABELS = frozenset({"person", "car", "truck", "bus", "motorcycle", "bicycle"})


def _activity_crop(
    frame_annotations: Sequence[Any],
    frame_size: tuple[int, int],
    aspect: float,
    *,
    padding: float = 0.2,
    max_area: float = 0.6,
) -> tuple[int, int, int, int] | None:
    """Smallest region at the target aspect ratio holding every tracked person or vehicle.

    Returns ``(left, top, right, bottom)`` in source pixels, or None when the
    window has no such tracks or the region would already cover more than
    ``max_area`` of the frame, in which case cropping buys no resolution.
    """
    width, height = frame_size
    boxes = [
        detection.boundary
        for frame in frame_annotations
        for detection in frame.detections
        if detection.label in _CROP_LABELS
    ]
    if not boxes or width <= 0 or height <= 0:
        return None
    left = min(box.left for box in boxes)
    top = min(box.top for box in boxes)
    right = max(box.right for box in boxes)
    bottom = max(box.bottom for box in boxes)
    crop_width = (right - left) * (1 + padding)
    crop_height = (bottom - top) * (1 + padding)
    if crop_width <= 0 or crop_height <= 0:
        return None
    # Grow the shorter side to the target aspect ratio so the resize does not
    # distort, then clamp to the frame.
    if crop_width / crop_height < aspect:
        crop_width = crop_height * aspect
    else:
        crop_height = crop_width / aspect
    crop_width = min(crop_width, width)
    crop_height = min(crop_height, height)
    if crop_width * crop_height > max_area * width * height:
        return None
    centre_x = (left + right) / 2
    centre_y = (top + bottom) / 2
    x0 = int(round(max(0.0, min(centre_x - crop_width / 2, width - crop_width))))
    y0 = int(round(max(0.0, min(centre_y - crop_height / 2, height - crop_height))))
    x1 = int(round(min(width, x0 + crop_width)))
    y1 = int(round(min(height, y0 + crop_height)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1


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


def _overlap_area(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> int:
    """Pixel area shared by two (x0, y0, x1, y1) rectangles."""
    width = min(first[2], second[2]) - max(first[0], second[0])
    height = min(first[3], second[3]) - max(first[1], second[1])
    return max(0, width) * max(0, height)


def _draw_detections(
    frame: Any,
    detections: Sequence[Any],
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> tuple[Any, list[dict[str, Any]]]:
    import cv2

    annotated = frame.copy()
    frame_height, frame_width = annotated.shape[:2]
    font_scale, text_thickness, box_thickness = _overlay_style(annotated)
    border_color = (255, 255, 255)
    text_color = (0, 0, 0)
    pad = 2
    features: list[dict[str, Any]] = []
    placed: list[tuple[int, int, int, int]] = []
    labels: list[tuple[str, tuple[int, int, int, int], int, int]] = []
    boxes: list[tuple[Any, tuple[int, int, int, int]]] = []
    for detection in detections:
        box = (
            round((detection.boundary.left - offset_x) * scale_x),
            round((detection.boundary.top - offset_y) * scale_y),
            round((detection.boundary.right - offset_x) * scale_x),
            round((detection.boundary.bottom - offset_y) * scale_y),
        )
        boxes.append((detection, box))
        cv2.rectangle(annotated, box[:2], box[2:], border_color, box_thickness)
    # Labels are placed after every box is drawn, and smallest boxes first: a
    # small or distant object has the fewest spots to put its label, so it
    # chooses before a large neighbour that can be read from almost anywhere.
    for detection, (left, top, right, bottom) in sorted(
        boxes, key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1])
    ):
        label = f"{detection.label} #{detection.track_id}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )
        full_width, full_height = label_width + 2 * pad, label_height + baseline + 2 * pad
        # Candidate top-left corners for the label plate, in order of preference:
        # above, below, inside-top, then beside the box on either side.
        candidates = [
            (left, top - full_height),
            (left, bottom),
            (left, top),
            (right, top),
            (left - full_width, top),
            (right - full_width, top - full_height),
            (right - full_width, bottom),
            (left, bottom - full_height),
        ]
        best, best_overlap = None, None
        for x, y in candidates:
            x = max(0, min(x, frame_width - full_width))
            y = max(0, min(y, frame_height - full_height))
            plate = (x, y, x + full_width, y + full_height)
            overlap = sum(_overlap_area(plate, other) for other in placed)
            if best_overlap is None or overlap < best_overlap:
                best, best_overlap = plate, overlap
            if overlap == 0:
                break
        plate = best
        placed.append(plate)
        labels.append((label, plate, label_height, pad))
    for label, (x0, y0, x1, y1), label_height, pad in labels:
        cv2.rectangle(annotated, (x0, y0), (x1, y1), border_color, thickness=-1)
        cv2.putText(
            annotated,
            label,
            (x0 + pad, y0 + pad + label_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            text_thickness,
            lineType=cv2.LINE_AA,
        )
    for detection in detections:
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


#: Ways a model writes a detector id into prose despite being told not to.
_TRACK_ID_PATTERNS = (
    re.compile(r"\s*\(\s*(?:track\s*)?(?:id\s*)?#?\s*\d+\s*\)", re.IGNORECASE),
    re.compile(r"\s*\btrack(?:\s*id)?\s*#?\s*\d+", re.IGNORECASE),
    re.compile(r"\s*#\s*\d+"),
)


def _strip_track_ids(text: str) -> str:
    """Remove detector ids from prose the prompt asked to keep them out of.

    The instruction is the real fix; this is the guarantee. A model that slips
    "the white SUV (track 201)" into a description would otherwise put an
    identifier in front of a reader that means nothing to them and changes
    between runs. The id stays available in ``involved_objects[].id``.
    """
    cleaned = text
    for pattern in _TRACK_ID_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return re.sub(r"\s+([,.;:])", r"\1", cleaned).strip()


_ACTION_LOOKUP = {action.lower(): action for action in ACTION_VOCABULARY}


def _normalise_actions(value: Any) -> tuple[list[str], list[str]]:
    """Split the model's actions into checklist phrases and anything else it wrote.

    A phrase is kept as a checklist action when, after trimming case and
    punctuation, it equals a vocabulary entry or contains one ("the driver
    opens a vehicle door" still counts). Everything else goes to
    ``other_actions`` so a real observation outside the vocabulary is not
    silently dropped.
    """
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return [], []
    known: list[str] = []
    other: list[str] = []
    for raw in value:
        if not isinstance(raw, str):
            continue
        text = re.sub(r"\s+", " ", raw).strip().strip(".;,").strip().lower()
        if not text:
            continue
        match = _ACTION_LOOKUP.get(text)
        if match is None:
            match = next(
                (action for phrase, action in _ACTION_LOOKUP.items() if phrase in text),
                None,
            )
        if match is None:
            if raw.strip() not in other:
                other.append(raw.strip())
        elif match not in known:
            known.append(match)
    return known, other


def _salvage_events(text: str) -> list[dict[str, Any]]:
    """Decode the leading complete objects of a JSON event array whose tail is missing.

    A model that runs out of output tokens stops mid-string, and one bad
    character would otherwise cost the whole window. The objects before the
    cut are intact, so they are kept and the partial one is dropped.
    """
    start = text.find("[")
    if start < 0:
        return []
    decoder = json.JSONDecoder()
    position = start + 1
    events: list[dict[str, Any]] = []
    while True:
        while position < len(text) and text[position] in " \t\r\n,":
            position += 1
        if position >= len(text) or text[position] != "{":
            break
        try:
            event, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            break
        events.append(event)
    return events


def _parse_event_list(content: str) -> list[Any]:
    """Parse the model's JSON, keeping every complete event when the output was cut off."""
    cleaned = _strip_json_fence(content)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = _salvage_events(cleaned)
        if not parsed:
            raise
        logger.warning("Model output was truncated; salvaged %d complete events", len(parsed))
    if isinstance(parsed, dict):
        parsed = parsed.get("events")
    if not isinstance(parsed, list):
        raise ValueError("Model response must be a JSON event list.")
    return parsed


def _clean_events(events: Sequence[Any]) -> list[Any]:
    """Enforce the description rules on whatever the model actually returned."""
    cleaned: list[Any] = []
    for event in events:
        if not isinstance(event, dict):
            cleaned.append(event)
            continue
        item = dict(event)
        for key in ("event_name", "description"):
            if isinstance(item.get(key), str):
                item[key] = _strip_track_ids(item[key])
        if "actions" in item:
            item["actions"], other_actions = _normalise_actions(item["actions"])
            if other_actions:
                item["other_actions"] = other_actions
        objects = item.get("involved_objects")
        if isinstance(objects, list):
            item["involved_objects"] = [
                {
                    **entry,
                    "physical_details": _strip_track_ids(entry["physical_details"]),
                }
                if isinstance(entry, dict) and isinstance(entry.get("physical_details"), str)
                else entry
                for entry in objects
            ]
        cleaned.append(item)
    return cleaned


def _strip_json_fence(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        return content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return content
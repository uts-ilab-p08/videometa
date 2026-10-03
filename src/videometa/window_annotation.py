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
from videometa.local_qwen import (
    DEFAULT_MODEL_ID,
    count_tokens,
    generate_from_video,
    load_local_qwen,
)


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
    "  3. Movement, step by step: the shape of the path (see the movement "
    "rule below), direction of travel by landmark, pace (standing, "
    "strolling, hurrying, creeping, accelerating, braking), changes of pace "
    "or direction, stops and starts, and how far the subject travels "
    "relative to the scene.\n"
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
    "- Movement rule: say what shape every movement takes, for vehicles and "
    "people alike, and do not leave it at 'moves' or 'drives past' when the "
    "path can be seen. Say whether the subject keeps straight on, turns left "
    "or turns right, makes a U-turn, reverses, goes round in a circle or "
    "loop, weaves or zigzags, walks back and forth, turns around, slows "
    "down, speeds up, stops, pauses and sets off again, pulls into or out of "
    "a parking bay, parks, or follows another subject. Left and right are "
    "from the subject's own direction of travel, as its driver or the walker "
    "would say it, not from the camera's point of view; when the two differ, "
    "say which way the subject's front swings ('turns left, swinging its "
    "nose towards the building'). A turn is a change of heading; a vehicle "
    "that only crosses the frame on a curved road is keeping straight on. "
    "Each of these that applies also belongs in the event's actions list "
    "whenever a matching action phrase is available.\n"
    "- Report every observable activity involving a person or a vehicle, "
    "including ordinary movement such as someone walking through the scene or "
    "a vehicle driving past. Do not limit yourself to unusual or noteworthy "
    "events.\n"
    "- Report only what is visible. Where a colour or type cannot be "
    "determined, stay general ('a dark hatchback') rather than speculate. If "
    "genuinely nothing moves in this window, return an empty list rather than "
    "inventing an event.\n"
    "- One event, one set of subjects. The description covers only the "
    "subjects listed in involved_objects: their appearance, movement, "
    "interactions and end state. Anyone or anything else in the frame may be "
    "named once as a location reference ('beside the parked silver car', "
    "'past a person standing at the counter') but is never described or "
    "given actions of its own. Something unrelated happening elsewhere in "
    "the frame is a separate event, not part of this one.\n"
    "- involved_objects and the description must point at the same subjects. "
    "The subject performing the action is always listed, and so is every "
    "other person or object directly part of the action: the person talked "
    "to, handed to, hugged or walked with; the object carried, picked up, put "
    "down or handed over; the vehicle entered, exited, loaded or driven. "
    "Before finishing an event, check the overlay boxes one by one and list "
    "every box that is directly part of it, people and objects alike. An "
    "item that is part of the event but has no box is described in the text "
    "and never given an invented id. Match each id to its subject by the "
    "overlay box and label: when the description says 'a person in a dark "
    "jacket walks toward the counter', the id is the box around that walker, "
    "not the box around the person already standing at the counter.\n"
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
    "  description: 'car #201 moves from top-left to middle-left.'\n"
    "Not acceptable either (the description is about the walker, but the only "
    "involved object is the person being approached):\n"
    "  event_name: 'Person walks toward the counter'\n"
    "  description: 'A person in a dark jacket walks from the right toward "
    "the counter, where a person in a pink shirt is standing and sorting "
    "cups.'\n"
    "  involved_objects: [{id of the person in the pink shirt}]"
)

#: What `physical_details` should hold for each involved object.
PHYSICAL_DETAILS_GUIDANCE = (
    "For each involved object, physical_details is one short phrase: colour "
    "(with shade), type, make or garment style, approximate size relative to "
    "the scene, and anything it carries, wears or tows. Movement and "
    "interactions belong in the description, not here."
)

#: The self-review the model runs on its draft before answering. Written as
#: questions because a model told "be careful" is not more careful; a model
#: told what to check, one event at a time, drops the events that fail.
REVIEW_GUIDANCE = (
    "Before you answer, review your draft critically, one event at a time, "
    "thinking it through step by step. Do this silently: the output stays "
    "JSON only. For each event ask:\n"
    "  1. Did it actually happen? A viewer must be able to point to the frames "
    "where it does. Drop anything inferred, guessed, or carried over from a "
    "detector label rather than seen.\n"
    "  2. Is it worth recording? A static background object, a box flickering "
    "on a parked vehicle, or a subject that is merely visible without doing "
    "anything is not an event.\n"
    "  3. Does it make sense? The path must be physically possible, the "
    "timing must fit inside the window, and the start state, the steps and "
    "the end state must agree with each other.\n"
    "  4. Is it the same scene as another event in the list? Two events that "
    "retell the same subject's movement are one event with all of its "
    "actions; merge them.\n"
    "  5. Do the involved objects match the text? Every subject the "
    "description is about is listed by its own box, nothing is listed that "
    "the description does not involve, and no id is invented.\n"
    "  6. Is every listed action visible in the frames and stated in the "
    "description, and is the shape of each movement named?\n"
    "Fix what can be fixed and remove the rest. A shorter list of events that "
    "are all real and correctly attributed is worth more than a longer list "
    "with one that is not."
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

def action_guidance(vocabulary: Sequence[str] | None = ACTION_VOCABULARY) -> str:
    """The checklist block for a prompt, or an empty string when no vocabulary is set.

    The vocabulary is a parameter so the annotators work for any dataset:
    pass the actions your evaluation cares about, or None to let the model
    name actions freely.
    """
    if not vocabulary:
        return ""
    return (
        "Before writing, go through this checklist for EVERY person and EVERY "
        "vehicle in the window and decide which of these actions you can actually "
        "see: "
        + "; ".join(vocabulary)
        + ".\n"
        "Return one event per subject per continuous scene, not one per action. "
        "The event's actions list holds every checklist phrase that subject shows "
        "in the window, in order, copied exactly, and the description tells them "
        "as one account: a person who enters through the doors, walks to the "
        "counter and pays is ONE event with three actions, never three events "
        "that each retell the same walk. Start a new event only when the subjects "
        "change (a different person or vehicle, or someone joining or leaving the "
        "interaction) or when the subject's activity clearly ends and a separate "
        "one begins later in the window.\n"
        "Ordinary walking or driving is an event on its own only when nothing "
        "more specific is visible for that subject; never let it stand in for a "
        "more specific action, and never repeat the same subject's movement as a "
        "second event. The description must state each listed action in words. "
        "Do not list an action you cannot see."
    )


#: The checklist block built from the default vocabulary.
ACTION_GUIDANCE = action_guidance(ACTION_VOCABULARY)

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
        action_vocabulary: Sequence[str] | None = ACTION_VOCABULARY,
    ) -> None:
        """``action_vocabulary`` is the checklist the model is asked to name actions
        from (see `ACTION_VOCABULARY`); pass your dataset's own list, or None to
        drop the checklist and let the model describe actions freely."""
        if not base_url or not api_key:
            raise ValueError("base_url and api_key are required")
        try:
            from openai import OpenAI
        except ImportError as error:
            raise ImportError("LVLM annotation requires `pip install openai`.") from error
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.feature_frames = feature_frames
        self.action_vocabulary = tuple(action_vocabulary) if action_vocabulary else None

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
            f"{action_guidance(_vocabulary_of(self))}\n\n"
            f"{PHYSICAL_DETAILS_GUIDANCE}\n\n"
            f"{REVIEW_GUIDANCE}\n\n"
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
        return _clean_events(
            _parse_event_list(response.choices[0].message.content),
            vocabulary=_vocabulary_of(self),
        )


class LocalQwenEventAnnotator:
    """Annotate a prepared, boundary-annotated window using local MLX Qwen3-VL."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        model: Any | None = None,
        processor: Any | None = None,
        video_fps: float = 2.0,
        max_tokens: int = 1200,
        prompt_token_budget: int = DEFAULT_PROMPT_TOKEN_BUDGET,
        feature_frames: int = DEFAULT_FEATURE_FRAMES,
        action_vocabulary: Sequence[str] | None = ACTION_VOCABULARY,
    ) -> None:
        """``action_vocabulary`` is the checklist the model is asked to name actions
        from (see `ACTION_VOCABULARY`); pass your dataset's own list, or None to
        drop the checklist and let the model describe actions freely.

        Pass ``model`` and ``processor`` to reuse weights that are already
        loaded, for instance by a `LocalQwenActivityScorer` in the same
        process; otherwise ``model_id`` is loaded here."""
        if (model is None) != (processor is None):
            raise ValueError("pass both model and processor, or neither")
        if model is None:
            model, processor = load_local_qwen(model_id)
        self.model = model
        self.processor = processor
        self.video_fps = video_fps
        self.max_tokens = max_tokens
        self.prompt_token_budget = prompt_token_budget
        self.feature_frames = feature_frames
        self.action_vocabulary = tuple(action_vocabulary) if action_vocabulary else None

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
            "only the objects involved, each by the id of the overlay box around that "
            "subject. The overlays show detector boundaries "
            "labelled as `class #track_id`. Use the video and the spatial features "
            "as supporting evidence; do not treat detector labels as certain "
            f"visual facts. {_crop_note(prepared_input)}\n\n"
            f"{DESCRIPTION_GUIDANCE}\n\n"
            f"{action_guidance(_vocabulary_of(self))}\n\n"
            "Each involved object carries its detector track id in the id field and "
            "its label. "
            f"{PHYSICAL_DETAILS_GUIDANCE}\n\n"
            f"{REVIEW_GUIDANCE}\n\n"
            'Return JSON only as a list of events: [{"event_name": str, '
            '"description": str, "actions": [str], "involved_objects": [{"id": str, '
            '"label": str, "physical_details": str}]}].\n\n'
            f"{FEATURE_LEGEND}\n"
            f"{_compact_json(_compact_object_features(prepared_input.object_features))}\n\n"
            f"{FRAME_LEGEND}\n"
            f"{_compact_json(_sample_frame_features(prepared_input.frame_features, frame_count))}"
        )

    def _count_tokens(self, prompt: str) -> int:
        return count_tokens(self.processor, prompt)

    def _generate_events(self, video_path: str, prompt: str) -> list[dict[str, Any]]:
        """Run local Qwen3-VL over a local video path and parse its JSON event list."""
        output_text = generate_from_video(
            self.model,
            self.processor,
            video_path,
            prompt,
            fps=self.video_fps,
            max_tokens=self.max_tokens,
        )
        parsed = _parse_event_list(output_text)
        logger.info("Local Qwen returned %d events for %s", len(parsed), video_path)
        return _clean_events(parsed, vocabulary=_vocabulary_of(self))


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
            min_size=(width / _CROP_MAX_UPSCALE, height / _CROP_MAX_UPSCALE),
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
    min_size: tuple[float, float] | None = None,
) -> tuple[int, int, int, int] | None:
    """Smallest region at the target aspect ratio around the window's actors.

    The actors are the people in the window plus every vehicle that moves;
    a parked car is scenery. In a car park the union of *all* tracked vehicles
    is the whole frame, so cropping to it never engaged. Three candidate sets
    are tried in order and the first one that fits under ``max_area`` wins:
    people plus moving vehicles, moving tracks only, people only.

    ``min_size`` is the smallest ``(width, height)`` the region may have, so a
    lone distant person is not blown up into a blur; the writer passes half
    its output size, which caps the upscale at 2x.

    Returns ``(left, top, right, bottom)`` in source pixels, or None when no
    candidate fits, in which case the whole frame is used.
    """
    width, height = frame_size
    if width <= 0 or height <= 0:
        return None
    tracks: dict[Any, dict[str, Any]] = {}
    for frame in frame_annotations:
        for detection in frame.detections:
            if detection.label not in _CROP_LABELS:
                continue
            track = tracks.setdefault(
                detection.track_id,
                {"label": detection.label, "first": detection.boundary, "boxes": []},
            )
            track["boxes"].append(detection.boundary)
            track["last"] = detection.boundary
    if not tracks:
        return None
    diagonal = (width**2 + height**2) ** 0.5
    people: list[Any] = []
    moving: list[Any] = []
    moving_vehicles: list[Any] = []
    for track in tracks.values():
        first, last = track["first"], track["last"]
        travelled = _centre_distance(first, last)
        size = ((first.right - first.left) ** 2 + (first.bottom - first.top) ** 2) ** 0.5
        is_moving = travelled > _CROP_MOVING_FRACTION * diagonal or travelled > 0.5 * size
        if track["label"] == "person":
            people.extend(track["boxes"])
        elif is_moving:
            moving_vehicles.extend(track["boxes"])
        if is_moving:
            moving.extend(track["boxes"])
    for boxes in (people + moving_vehicles, moving, people):
        crop = _fit_crop(boxes, (width, height), aspect, padding, max_area, min_size)
        if crop is not None:
            return crop
    return None


#: The most a crop may be enlarged on its way to the output size. Beyond 2x a
#: distant figure is a smear of upscaled pixels and the model gains nothing.
_CROP_MAX_UPSCALE = 2.0

#: Fraction of the frame diagonal a track's centre must travel to count as
#: moving. Two percent of a 1080p diagonal is 44 px, above tracker jitter and
#: below anything that walks or drives.
_CROP_MOVING_FRACTION = 0.02


def _centre_distance(first: Any, second: Any) -> float:
    return (
        ((first.left + first.right) / 2 - (second.left + second.right) / 2) ** 2
        + ((first.top + first.bottom) / 2 - (second.top + second.bottom) / 2) ** 2
    ) ** 0.5


def _fit_crop(
    boxes: Sequence[Any],
    frame_size: tuple[int, int],
    aspect: float,
    padding: float,
    max_area: float,
    min_size: tuple[float, float] | None = None,
) -> tuple[int, int, int, int] | None:
    """The padded, aspect-corrected region around ``boxes``, or None when it is too large."""
    width, height = frame_size
    if not boxes:
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
    if min_size is not None:
        crop_width = max(crop_width, min_size[0], min_size[1] * aspect)
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


_OBJECT_ID_SUFFIX = re.compile(r"(\d+)\s*$")


def _normalise_object_id(value: Any) -> Any:
    """Reduce an involved object's id to the bare track id the overlay carries.

    The prompt asks for the id alone, but a model that reads ``person #20``
    off the overlay sometimes returns exactly that. Downstream joins compare
    ids as strings against the tracker's ``"20"``, so the prefix would make a
    correctly identified object untraceable. Anything without a trailing
    number is returned as given.
    """
    if value is None:
        return None
    text = str(value).strip()
    match = _OBJECT_ID_SUFFIX.search(text)
    return match.group(1) if match else text


def _clean_involved_object(entry: dict[str, Any]) -> dict[str, Any]:
    """Bare track id, prose free of ids; keys the model did not send stay absent."""
    cleaned = dict(entry)
    if "id" in cleaned:
        cleaned["id"] = _normalise_object_id(cleaned["id"])
    if isinstance(cleaned.get("physical_details"), str):
        cleaned["physical_details"] = _strip_track_ids(cleaned["physical_details"])
    return cleaned


def _vocabulary_of(annotator: Any) -> tuple[str, ...] | None:
    """The annotator's checklist; the default one when the object does not carry it."""
    if annotator is None or not hasattr(annotator, "action_vocabulary"):
        return ACTION_VOCABULARY
    return annotator.action_vocabulary


def _normalise_actions(
    value: Any, vocabulary: Sequence[str] | None = ACTION_VOCABULARY
) -> tuple[list[str], list[str]]:
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
    lookup = {action.lower(): action for action in (vocabulary or ())}
    known: list[str] = []
    other: list[str] = []
    for raw in value:
        if not isinstance(raw, str):
            continue
        text = re.sub(r"\s+", " ", raw).strip().strip(".;,").strip().lower()
        if not text:
            continue
        match = lookup.get(text)
        if match is None:
            match = next(
                (action for phrase, action in lookup.items() if phrase in text),
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


def _clean_events(
    events: Sequence[Any], vocabulary: Sequence[str] | None = ACTION_VOCABULARY
) -> list[Any]:
    """Enforce the description rules on whatever the model actually returned.

    Exact duplicates (same event_name and description) are dropped: a model
    asked for one event per action pads its answer by repeating the same
    event, and a repeated event describes nothing new.
    """
    seen: set[tuple[str, str]] = set()
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
            item["actions"], other_actions = _normalise_actions(item["actions"], vocabulary)
            if other_actions:
                item["other_actions"] = other_actions
        key = (str(item.get("event_name", "")).strip(), str(item.get("description", "")).strip())
        if key != ("", "") and key in seen:
            continue
        seen.add(key)
        objects = item.get("involved_objects")
        if isinstance(objects, list):
            item["involved_objects"] = [
                _clean_involved_object(entry) if isinstance(entry, dict) else entry
                for entry in objects
            ]
        cleaned.append(item)
    return cleaned


def _strip_json_fence(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        return content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return content
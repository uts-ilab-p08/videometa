"""Local MLX Qwen3-VL: one place to load the model and run it over a video.

Both the activity gate (`videometa.activity_gate`) and the event annotator
(`videometa.window_annotation`) talk to the same model. Keeping the MLX calls
here means the two stages cannot drift apart in how they format a video
message, pass the frame rate, or release Metal memory after a call, and it
lets one loaded model serve both stages in a single process.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "mlx-community/Qwen3-VL-4B-Instruct-4bit"


def load_local_qwen(model_id: str = DEFAULT_MODEL_ID) -> tuple[Any, Any]:
    """Load an MLX vision model and its processor."""
    try:
        from mlx_vlm import load
    except ImportError as error:
        raise ImportError("Local Qwen annotation requires `pip install mlx-vlm`.") from error
    model, processor = load(model_id)
    logger.info("Loaded local Qwen model: %s", model_id)
    return model, processor


def count_tokens(processor: Any, text: str) -> int:
    """Number of tokens `text` occupies in the model's prompt."""
    tokenizer = getattr(processor, "tokenizer", processor)
    return len(tokenizer(text).input_ids)


def generate_from_video(
    model: Any,
    processor: Any,
    video_path: str,
    prompt: str,
    *,
    fps: float,
    max_tokens: int,
) -> str:
    """Run the model over one local video file with a text prompt and return its reply.

    The Metal buffer pool is cleared after every call, succeed or fail: across
    dozens of clips in one loop that pool otherwise grows until allocation
    fails.
    """
    try:
        import mlx.core as mx
        from mlx_vlm import generate
    except ImportError as error:
        raise ImportError("Local Qwen annotation requires `pip install mlx-vlm`.") from error

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    formatted_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    try:
        output = generate(
            model,
            processor,
            formatted_prompt,
            video=[video_path],
            fps=fps,
            max_tokens=max_tokens,
            verbose=False,
        )
    finally:
        mx.clear_cache()
    return output.text if hasattr(output, "text") else str(output)

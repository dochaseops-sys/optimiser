from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Sequence

from PIL import Image

try:  # pragma: no cover - import shim for direct execution
    from .background_generators import (
        blur_expand_background,
        extract_dominant_color,
        parse_hex_color,
        solid_background,
    )
    from .image_processor import load_image_bytes
    from .utils import (
        extension_for_format,
        normalize_format,
        sanitize_stem,
        save_image_to_bytes,
        is_video_file,
    )
except ImportError:  # pragma: no cover
    from background_generators import (
        blur_expand_background,
        extract_dominant_color,
        parse_hex_color,
        solid_background,
    )
    from image_processor import load_image_bytes
    from utils import (
        extension_for_format,
        normalize_format,
        sanitize_stem,
        save_image_to_bytes,
        is_video_file,
    )


try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - compatibility fallback
    LANCZOS = Image.LANCZOS


FIT_EXPAND = "fit_expand"
BLUR_EXPAND = "blur_expand"
SOLID_BACKGROUND = "solid_background"
DOMINANT_COLOR_BACKGROUND = "dominant_color_background"

ALLOWED_RESIZE_MODES = {
    FIT_EXPAND,
    BLUR_EXPAND,
    SOLID_BACKGROUND,
    DOMINANT_COLOR_BACKGROUND,
}

DEFAULT_BACKGROUND_COLOR = (255, 255, 255, 255)


@dataclass(slots=True, frozen=True)
class ResizeTarget:
    width: int
    height: int


@dataclass(slots=True)
class ResizeTask:
    filename: str
    image_bytes: bytes
    target: ResizeTarget
    mode: str
    background_color: str | None = None
    blur_radius: float = 24.0


@dataclass(slots=True)
class ResizeResult:
    source_name: str
    output_name: str | None
    output_bytes: bytes | None
    error: str | None = None


def normalize_resize_mode(mode: str | None) -> str:
    """Normalize request mode names to internal constants."""

    if not mode or not mode.strip():
        return FIT_EXPAND
    return mode.strip().lower().replace("-", "_")


def parse_target_dimensions(raw_targets: Sequence[str], width: str | None, height: str | None) -> list[ResizeTarget]:
    """Parse target dimensions from either repeated targets or width/height fields."""

    targets: list[ResizeTarget] = []

    for raw in raw_targets:
        for part in raw.replace("\n", ",").split(","):
            value = part.strip().lower()
            if not value:
                continue
            pieces = value.split("x")
            if len(pieces) != 2:
                raise ValueError(f"invalid target dimension '{part.strip()}'; expected WIDTHxHEIGHT")
            targets.append(_build_target(pieces[0], pieces[1]))

    if width is not None or height is not None:
        if width is None or height is None:
            raise ValueError("target_width and target_height must be provided together")
        targets.append(_build_target(width, height))

    if not targets:
        raise ValueError("at least one target dimension is required")

    return targets


def adapt_image(
    image: Image.Image,
    target_size: tuple[int, int],
    *,
    mode: str = FIT_EXPAND,
    background_color: str | None = None,
    blur_radius: float = 24.0,
) -> Image.Image:
    """Adapt an image to a target canvas without stretching or cropping foreground content."""

    target_width, target_height = target_size
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target dimensions must be positive")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("source dimensions must be positive")

    normalized_mode = normalize_resize_mode(mode)
    if normalized_mode not in ALLOWED_RESIZE_MODES:
        allowed = ", ".join(sorted(ALLOWED_RESIZE_MODES))
        raise ValueError(f"invalid resize mode '{mode}'; expected one of: {allowed}")

    foreground = image.convert("RGBA")
    source_ratio = foreground.width / foreground.height
    target_ratio = target_width / target_height

    if source_ratio > target_ratio:
        scaled_width = target_width
        scaled_height = max(1, int(round(target_width / source_ratio)))
    else:
        scaled_height = target_height
        scaled_width = max(1, int(round(target_height * source_ratio)))

    foreground = foreground.resize((scaled_width, scaled_height), LANCZOS)
    offset = ((target_width - scaled_width) // 2, (target_height - scaled_height) // 2)

    if normalized_mode == BLUR_EXPAND:
        canvas = blur_expand_background(
            image,
            target_size,
            blur_radius=blur_radius,
        )
    elif normalized_mode in {FIT_EXPAND, DOMINANT_COLOR_BACKGROUND}:
        canvas = solid_background(target_size, extract_dominant_color(image))
    else:
        fill = parse_hex_color(background_color, fallback=DEFAULT_BACKGROUND_COLOR)
        canvas = solid_background(target_size, fill)

    canvas.alpha_composite(foreground, dest=offset)
    return canvas


def process_single_video_resize(task: ResizeTask) -> ResizeResult:
    """Adapt a video to a target canvas size using the requested background mode."""
    import tempfile
    import numpy as np
    from PIL import ImageFilter, ImageEnhance
    from moviepy import VideoFileClip, ColorClip, CompositeVideoClip

    source_name = sanitize_stem(task.filename)
    ext = Path(task.filename).suffix.lower() if task.filename else ".mp4"

    temp_in_path = None
    temp_out_path = None
    video = None
    bg_video = None
    bg_clip = None
    resized_video = None
    composited = None

    try:
        # Write input video bytes to temp file
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_in:
            temp_in.write(task.image_bytes)
            temp_in_path = temp_in.name

        video = VideoFileClip(temp_in_path)
        
        target_width = task.target.width
        target_height = task.target.height
        
        # Calculate proportional size
        source_ratio = video.w / video.h
        target_ratio = target_width / target_height

        if source_ratio > target_ratio:
            scaled_width = target_width
            scaled_height = max(1, int(round(target_width / source_ratio)))
        else:
            scaled_height = target_height
            scaled_width = max(1, int(round(target_height * source_ratio)))

        resized_video = video.resized((scaled_width, scaled_height))
        offset = ((target_width - scaled_width) // 2, (target_height - scaled_height) // 2)
        resized_video = resized_video.with_position(offset)

        normalized_mode = normalize_resize_mode(task.mode)
        
        if normalized_mode == BLUR_EXPAND:
            scale = max(target_width / video.w, target_height / video.h)
            fill_width = max(1, int(round(video.w * scale)))
            fill_height = max(1, int(round(video.h * scale)))
            bg_video = video.resized((fill_width, fill_height))
            
            # Crop to target size
            bg_video = bg_video.cropped(
                x_center=bg_video.w // 2,
                y_center=bg_video.h // 2,
                width=target_width,
                height=target_height,
            )
            
            # Apply blur and darken per frame
            radius = task.blur_radius
            def blur_and_darken_frame(frame_arr):
                img = Image.fromarray(frame_arr)
                if radius > 0:
                    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
                img = ImageEnhance.Brightness(img).enhance(0.82)
                return np.array(img)
                
            bg_clip = bg_video.image_transform(blur_and_darken_frame)
            
        elif normalized_mode in {FIT_EXPAND, DOMINANT_COLOR_BACKGROUND}:
            first_frame_arr = video.get_frame(0)
            first_frame_image = Image.fromarray(first_frame_arr)
            dominant = extract_dominant_color(first_frame_image)
            bg_clip = ColorClip(size=(target_width, target_height), color=dominant[:3], duration=video.duration)
            
        else: # solid_background
            fill = parse_hex_color(task.background_color, fallback=DEFAULT_BACKGROUND_COLOR)
            bg_clip = ColorClip(size=(target_width, target_height), color=fill[:3], duration=video.duration)

        composited = CompositeVideoClip([bg_clip, resized_video])

        # Write output video bytes to temp file
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_out:
            temp_out_path = temp_out.name

        # Select codecs
        codec = "libx264"
        audio_codec = "aac"
        if ext == ".webm":
            codec = "libvpx"
            audio_codec = "libvorbis"

        composited.write_videofile(
            temp_out_path,
            codec=codec,
            audio_codec=audio_codec if video.audio is not None else None,
            logger=None,
        )

        # Read back processed bytes
        with open(temp_out_path, "rb") as f:
            output_bytes = f.read()

        output_name = (
            f"{source_name}_{target_width}x{target_height}_"
            f"{normalized_mode}{ext}"
        )
        return ResizeResult(source_name, output_name, output_bytes, None)

    except Exception as exc:
        return ResizeResult(
            source_name=source_name,
            output_name=None,
            output_bytes=None,
            error=f"{task.filename or source_name} {task.target.width}x{task.target.height}: {exc}",
        )

    finally:
        # Close all moviepy clips
        for clip in (video, bg_video, bg_clip, resized_video, composited):
            if clip:
                try:
                    clip.close()
                except Exception:
                    pass
        # Clean up temp files
        for path in (temp_in_path, temp_out_path):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass


def process_single_resize(task: ResizeTask) -> ResizeResult:
    """Process a single image/target adaptation task."""

    source_name = sanitize_stem(task.filename)

    if is_video_file(task.filename):
        return process_single_video_resize(task)

    try:
        image, image_format = load_image_bytes(task.image_bytes)
        output_format = _resolve_output_format(image, image_format)
        adapted = adapt_image(
            image,
            (task.target.width, task.target.height),
            mode=task.mode,
            background_color=task.background_color,
            blur_radius=task.blur_radius,
        )
        final_image = _flatten_for_format(adapted, output_format)
        output_bytes = save_image_to_bytes(final_image, output_format)
        output_name = (
            f"{source_name}_{task.target.width}x{task.target.height}_"
            f"{normalize_resize_mode(task.mode)}.{extension_for_format(output_format)}"
        )
        return ResizeResult(source_name, output_name, output_bytes, None)
    except Exception as exc:
        return ResizeResult(
            source_name=source_name,
            output_name=None,
            output_bytes=None,
            error=f"{task.filename or source_name} {task.target.width}x{task.target.height}: {exc}",
        )


def process_resize_batch(tasks: Sequence[ResizeTask]) -> list[ResizeResult]:
    """Process resize tasks in a bounded thread pool for batch creative operations."""

    if not tasks:
        return []
    if len(tasks) == 1:
        return [process_single_resize(tasks[0])]

    worker_count = min(8, len(tasks), os.cpu_count() or 1)
    if worker_count <= 1:
        return [process_single_resize(task) for task in tasks]

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(process_single_resize, tasks))


def _build_target(width: str, height: str) -> ResizeTarget:
    try:
        target_width = int(width)
        target_height = int(height)
    except (TypeError, ValueError) as exc:
        raise ValueError("target dimensions must be integers") from exc

    if target_width <= 0 or target_height <= 0:
        raise ValueError("target dimensions must be positive")

    return ResizeTarget(target_width, target_height)


def _has_alpha(image: Image.Image) -> bool:
    if image.mode in {"RGBA", "LA"}:
        return True
    if image.mode == "P" and "transparency" in image.info:
        return True
    return False


def _resolve_output_format(image: Image.Image, image_format: str | None) -> str:
    if _has_alpha(image):
        return "PNG"
    return normalize_format(image_format if image_format else "JPEG")


def _flatten_for_format(image: Image.Image, image_format: str) -> Image.Image:
    if image_format in {"PNG", "WEBP", "TIFF"}:
        return image

    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    background.alpha_composite(image.convert("RGBA"))
    return background.convert("RGB")

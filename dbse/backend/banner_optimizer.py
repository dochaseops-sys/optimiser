from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
from typing import Sequence

from PIL import Image

try:  # pragma: no cover - import shim for direct execution
    from .image_processor import load_image_bytes
    from .utils import extension_for_format, sanitize_stem, is_video_file
except ImportError:  # pragma: no cover
    from image_processor import load_image_bytes
    from utils import extension_for_format, sanitize_stem, is_video_file


try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - compatibility fallback
    LANCZOS = Image.LANCZOS


@dataclass(slots=True, frozen=True)
class BannerOptimizationSettings:
    supported_dimensions: tuple[tuple[int, int], ...] = (
        (300, 250),
        (320, 480),
        (728, 90),
        (320, 50),
        (300, 50),
        (300, 600),
        (970, 250),
        (970, 90),
        (120, 600),
        (160, 600),
        (250, 250),
        (320, 100),
    )
    max_output_bytes: int = 150 * 1024
    jpeg_quality_min: int = 20
    jpeg_quality_max: int = 95
    png_quantize_colors: tuple[int, ...] = (256, 128, 64, 32, 16)


OPTIMIZATION_SETTINGS = BannerOptimizationSettings()


@dataclass(slots=True)
class OptimizationTask:
    filename: str
    banner_bytes: bytes


@dataclass(slots=True)
class OptimizationResult:
    source_name: str
    output_name: str | None
    output_bytes: bytes | None
    error: str | None = None


def resolve_target_dimensions(size: tuple[int, int]) -> tuple[int, int] | None:
    """Resolve a banner size to a supported target or a larger proportional version of one."""

    width, height = size
    matches: list[tuple[int, int, tuple[int, int]]] = []

    for target_width, target_height in OPTIMIZATION_SETTINGS.supported_dimensions:
        # Exact match
        if (width, height) == (target_width, target_height):
            return (target_width, target_height)

        # Only downscale; do not upscale smaller images
        if width < target_width or height < target_height:
            continue

        # Same aspect ratio check:
        # 450x375 matches 300x250 because 450 * 250 == 375 * 300
        if width * target_height != height * target_width:
            continue

        # Prefer the largest valid target area, so an image maps to the
        # most specific supported size rather than an unnecessarily small one
        matches.append((target_width * target_height, max(width // target_width, height // target_height), (target_width, target_height)))

    if not matches:
        return None

    return max(matches, key=lambda item: item[0])[2]


def parse_dimension_from_filename(filename: str | None) -> tuple[int, int] | None:
    """Parse target dimensions from filename if it contains a supported size."""
    if not filename:
        return None

    # Matches pattern like 300x250, 300_x_250, 300-x-250, 300 x 250, etc.
    matches = re.findall(r"(\d+)\s*[_.-]?\s*[xX]\s*[_.-]?\s*(\d+)", filename)
    for w_str, h_str in matches:
        w, h = int(w_str), int(h_str)
        if (w, h) in OPTIMIZATION_SETTINGS.supported_dimensions:
            return (w, h)
    return None


def _has_alpha(image: Image.Image) -> bool:
    if image.mode in {"RGBA", "LA"}:
        return True
    if image.mode == "P" and "transparency" in image.info:
        return True
    return False


def _resize_banner(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    if image.size == target_size:
        return image.copy()
    return image.resize(target_size, LANCZOS)


def _save_jpeg_candidate(image: Image.Image, quality: int) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=quality,
    )
    return buffer.getvalue()


def _encode_jpeg_under_limit(image: Image.Image) -> bytes | None:
    best_bytes: bytes | None = None
    low = OPTIMIZATION_SETTINGS.jpeg_quality_min
    high = OPTIMIZATION_SETTINGS.jpeg_quality_max

    while low <= high:
        quality = (low + high) // 2
        candidate = _save_jpeg_candidate(image, quality)
        if len(candidate) <= OPTIMIZATION_SETTINGS.max_output_bytes:
            best_bytes = candidate
            low = quality + 1
        else:
            high = quality - 1

    return best_bytes


def _save_png_candidate(image: Image.Image, colors: int) -> bytes:
    candidate = image.convert("RGBA").quantize(colors=colors)
    buffer = BytesIO()
    candidate.save(buffer, format="PNG", optimize=True, compress_level=9)
    return buffer.getvalue()


def _encode_png_under_limit(image: Image.Image) -> bytes | None:
    best_bytes: bytes | None = None
    best_size: int | None = None

    for colors in OPTIMIZATION_SETTINGS.png_quantize_colors:
        candidate = _save_png_candidate(image, colors)
        candidate_size = len(candidate)
        if best_size is None or candidate_size < best_size:
            best_bytes = candidate
            best_size = candidate_size
        if candidate_size <= OPTIMIZATION_SETTINGS.max_output_bytes:
            return candidate

    return None if best_size is None else (
        best_bytes if best_size <= OPTIMIZATION_SETTINGS.max_output_bytes else None
    )


def optimize_banner_bytes(
    image: Image.Image,
) -> tuple[bytes | None, str | None]:
    """Encode a banner under the configured size limit."""

    if _has_alpha(image):
        optimized = _encode_png_under_limit(image)
        return optimized, ("PNG" if optimized is not None else None)

    optimized = _encode_jpeg_under_limit(image)
    return optimized, ("JPEG" if optimized is not None else None)


def process_single_video_optimization(task: OptimizationTask) -> OptimizationResult:
    """Resize a video banner to a supported target size and compress it below the size cap."""
    import tempfile
    import os
    from moviepy import VideoFileClip

    source_name = sanitize_stem(task.filename)
    ext = Path(task.filename).suffix.lower() if task.filename else ".mp4"

    temp_in_path = None
    temp_out_path = None
    video = None
    resized_video = None

    try:
        # Write input video bytes to temp file
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_in:
            temp_in.write(task.banner_bytes)
            temp_in_path = temp_in.name

        video = VideoFileClip(temp_in_path)
        
        target_size = resolve_target_dimensions((video.w, video.h))
        if target_size is None:
            target_size = parse_dimension_from_filename(task.filename)
            
        if target_size is None:
            allowed = ", ".join(
                f"{width}x{height}"
                for width, height in OPTIMIZATION_SETTINGS.supported_dimensions
            )
            raise ValueError(
                f"unsupported banner size {video.w}x{video.h}; "
                f"use one of the supported sizes or an exact multiple: {allowed}"
            )

        resized_video = video.resized(target_size)
        duration = video.duration or 1.0

        # Try to compress the video under the cap (150 KB)
        target_bytes = OPTIMIZATION_SETTINGS.max_output_bytes
        bitrate_factor = 0.9
        output_bytes = None

        # Codec setup
        codec = "libx264"
        audio_codec = "aac"
        if ext == ".webm":
            codec = "libvpx"
            audio_codec = "libvorbis"

        # Optimization loop to find a bitrate that fits the 150 KB limit
        for attempt in range(3):
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_out:
                temp_out_path = temp_out.name

            # Calculate bitrate in bps
            available_bits = target_bytes * 8 * bitrate_factor
            target_bitrate_bps = max(20000, int(available_bits / duration))

            resized_video.write_videofile(
                temp_out_path,
                codec=codec,
                bitrate=f"{target_bitrate_bps}",
                audio_codec=audio_codec if video.audio is not None else None,
                logger=None,
            )

            file_size = os.path.getsize(temp_out_path)
            if file_size <= target_bytes:
                with open(temp_out_path, "rb") as f:
                    output_bytes = f.read()
                os.unlink(temp_out_path)
                temp_out_path = None
                break
            else:
                # If size exceeded, reduce factor and clean up
                os.unlink(temp_out_path)
                temp_out_path = None
                bitrate_factor *= 0.6

        if output_bytes is None:
            raise ValueError(
                f"could not compress video banner to <= {target_bytes // 1024} KB"
            )

        output_name = f"{source_name}_optimized{ext}"
        return OptimizationResult(
            source_name=source_name,
            output_name=output_name,
            output_bytes=output_bytes,
            error=None,
        )

    except Exception as exc:
        return OptimizationResult(
            source_name=source_name,
            output_name=None,
            output_bytes=None,
            error=f"{task.filename or source_name}: {exc}",
        )

    finally:
        # Close moviepy clips
        for clip in (video, resized_video):
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


def process_single_optimization(task: OptimizationTask) -> OptimizationResult:
    """Resize a banner to a supported target and compress it below the size cap."""

    source_name = sanitize_stem(task.filename)

    if is_video_file(task.filename):
        return process_single_video_optimization(task)

    try:
        banner, _ = load_image_bytes(task.banner_bytes)
        target_size = resolve_target_dimensions(banner.size)
        if target_size is None:
            target_size = parse_dimension_from_filename(task.filename)

        if target_size is None:
            allowed = ", ".join(
                f"{width}x{height}"
                for width, height in OPTIMIZATION_SETTINGS.supported_dimensions
            )
            raise ValueError(
                f"unsupported banner size {banner.width}x{banner.height}; "
                f"use one of the supported sizes or an exact multiple: {allowed}"
            )

        resized_banner = _resize_banner(banner, target_size)
        output_bytes, output_format = optimize_banner_bytes(resized_banner)
        if output_bytes is None or output_format is None:
            raise ValueError(
                f"could not compress banner to <= {OPTIMIZATION_SETTINGS.max_output_bytes // 1024} KB"
            )

        output_name = f"{source_name}_optimized.{extension_for_format(output_format)}"
        return OptimizationResult(
            source_name=source_name,
            output_name=output_name,
            output_bytes=output_bytes,
            error=None,
        )
    except Exception as exc:
        return OptimizationResult(
            source_name=source_name,
            output_name=None,
            output_bytes=None,
            error=f"{task.filename or source_name}: {exc}",
        )


def process_optimization_batch(tasks: Sequence[OptimizationTask]) -> list[OptimizationResult]:
    """Optimize banners in a stable, deterministic order."""

    if not tasks:
        return []

    # Pillow encoding has proven flaky under threaded optimization here, so keep
    # this path stable and deterministic instead of risking partial batch failure.
    return [process_single_optimization(task) for task in tasks]

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

try:  # pragma: no cover - import shim for direct execution
    from .utils import (
        extension_for_format,
        normalize_format,
        sanitize_stem,
        save_image_to_bytes,
        is_video_file,
    )
except ImportError:  # pragma: no cover
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


ALLOWED_PLACEMENT_MODES = {"smart", "top-left", "top-right"}


@dataclass(slots=True, frozen=True)
class ProcessingSettings:
    smart_region_width_ratio: float = 0.20
    smart_region_height_ratio: float = 0.15
    standard_logo_width_ratio: float = 0.015
    large_banner_logo_width_ratio: float = 0.025
    xl_banner_logo_width_ratio: float = 0.035
    large_banner_dimension_threshold: int = 1000
    xl_banner_dimension_threshold: int = 1600
    edge_margin_ratio: float = 0.0
    minimum_logo_width_px: int = 8
    minimum_logo_height_px: int = 8


SETTINGS = ProcessingSettings()


@dataclass(slots=True)
class BannerTask:
    filename: str
    banner_bytes: bytes
    logo_image: Image.Image
    placement_mode: str
    custom_name: str | None = None


@dataclass(slots=True)
class BannerResult:
    source_name: str
    output_name: str | None
    output_bytes: bytes | None
    error: str | None = None


@dataclass(slots=True, frozen=True)
class PlacementDecision:
    position: str
    x: int
    y: int
    logo_width: int
    logo_height: int


def load_image_bytes(image_bytes: bytes) -> tuple[Image.Image, str]:
    """Open image bytes with Pillow and return a detached image plus its format."""

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            image_format = image.format or "PNG"
            return image.copy(), image_format
    except Exception as exc:  # Pillow raises many exception types for bad input
        raise ValueError("corrupted or unsupported image") from exc


def load_image(image_bytes: bytes) -> Image.Image:
    """Open image bytes and return only the decoded Pillow image."""

    image, _ = load_image_bytes(image_bytes)
    return image


def resize_to_fit(
    source_size: tuple[int, int],
    max_size: tuple[int, int],
    *,
    allow_upscale: bool = False,
) -> tuple[int, int]:
    """Resize a size tuple to fit inside a bounding box without distortion."""

    source_width, source_height = source_size
    max_width, max_height = max_size
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source dimensions must be positive")
    if max_width <= 0 or max_height <= 0:
        raise ValueError("max dimensions must be positive")

    scale = min(max_width / source_width, max_height / source_height)
    if not allow_upscale:
        scale = min(scale, 1.0)

    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    return resized_width, resized_height


def _edge_noise_score(region: Image.Image) -> float:
    """Compute a mean edge-intensity score for a candidate placement region."""

    gray = region.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    if arr.size == 0:
        return float("inf")

    padded = np.pad(arr, 1, mode="edge")
    gx = (
        padded[:-2, 2:]
        + 2.0 * padded[1:-1, 2:]
        + padded[2:, 2:]
        - padded[:-2, :-2]
        - 2.0 * padded[1:-1, :-2]
        - padded[2:, :-2]
    )
    gy = (
        padded[2:, :-2]
        + 2.0 * padded[2:, 1:-1]
        + padded[2:, 2:]
        - padded[:-2, :-2]
        - 2.0 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
    )
    edges = np.abs(gx) + np.abs(gy)
    return float(edges.mean())


def _candidate_boxes(
    width: int,
    height: int,
    logo_size: tuple[int, int] | None = None,
) -> dict[str, tuple[int, int, int, int]]:
    region_width = max(1, int(round(width * SETTINGS.smart_region_width_ratio)))
    region_height = max(1, int(round(height * SETTINGS.smart_region_height_ratio)))
    if logo_size is not None:
        margin = max(0, int(round(width * SETTINGS.edge_margin_ratio)))
        logo_width, logo_height = logo_size
        region_width = max(region_width, min(width, logo_width + (margin * 2)))
        region_height = max(region_height, min(height, logo_height + (margin * 2)))
    region_width = min(region_width, width)
    region_height = min(region_height, height)

    return {
        "top-left": (0, 0, region_width, region_height),
        "top-right": (max(0, width - region_width), 0, width, region_height),
    }


def choose_position(
    banner: Image.Image,
    placement_mode: str,
    logo_size: tuple[int, int] | None = None,
) -> str:
    """Choose a top-left or top-right placement based on the requested mode."""

    if placement_mode in ALLOWED_PLACEMENT_MODES - {"smart"}:
        return placement_mode

    if placement_mode != "smart":
        raise ValueError("placement_mode must be one of smart, top-left, top-right")

    boxes = _candidate_boxes(*banner.size, logo_size=logo_size)
    scores = {
        position: _edge_noise_score(banner.crop(box))
        for position, box in boxes.items()
    }
    return min(scores, key=scores.get)


def calculate_logo_size(
    banner_size: tuple[int, int],
    logo_size: tuple[int, int],
) -> tuple[int, int]:
    """Size the logo from banner width while keeping it inside the banner."""

    banner_width, banner_height = banner_size
    logo_width, logo_height = logo_size
    if banner_width <= 0 or banner_height <= 0:
        raise ValueError("banner dimensions must be positive")
    if logo_width <= 0 or logo_height <= 0:
        raise ValueError("logo dimensions must be positive")

    margin = max(0, int(round(banner_width * SETTINGS.edge_margin_ratio)))
    largest_dimension = max(banner_width, banner_height)
    width_ratio = SETTINGS.standard_logo_width_ratio
    if largest_dimension >= SETTINGS.xl_banner_dimension_threshold:
        width_ratio = SETTINGS.xl_banner_logo_width_ratio
    elif largest_dimension >= SETTINGS.large_banner_dimension_threshold:
        width_ratio = SETTINGS.large_banner_logo_width_ratio

    target_width = max(1, int(round(banner_width * width_ratio)))
    max_width = max(1, banner_width - (margin * 2))
    max_height = max(1, banner_height - (margin * 2))

    target_scale = target_width / logo_width
    fit_scale = min(max_width / logo_width, max_height / logo_height)
    minimum_scale = max(
        SETTINGS.minimum_logo_width_px / logo_width,
        SETTINGS.minimum_logo_height_px / logo_height,
    )
    scale = min(max(target_scale, minimum_scale), fit_scale)

    width = max(1, int(round(logo_width * scale)))
    height = max(1, int(round(logo_height * scale)))
    return resize_to_fit((width, height), (max_width, max_height), allow_upscale=False)


def select_placement(
    banner: Image.Image,
    logo_size: tuple[int, int],
    placement_mode: str,
) -> PlacementDecision:
    """Resolve the exact top-left or top-right placement coordinates."""

    position = choose_position(banner, placement_mode, logo_size=logo_size)
    banner_width, banner_height = banner.size
    logo_width, logo_height = logo_size
    margin = max(0, int(round(banner_width * SETTINGS.edge_margin_ratio)))

    if position == "top-right":
        x = max(0, banner_width - margin - logo_width)
    else:
        x = min(margin, max(0, banner_width - logo_width))
    y = min(margin, max(0, banner_height - logo_height))

    return PlacementDecision(
        position=position,
        x=x,
        y=y,
        logo_width=logo_width,
        logo_height=logo_height,
    )


def _prepare_logo(logo: Image.Image, banner_size: tuple[int, int]) -> Image.Image:
    target_size = calculate_logo_size(banner_size, logo.size)
    return logo.convert("RGBA").resize(target_size, LANCZOS).convert("RGBA")


def _flatten_for_format(image: Image.Image, image_format: str) -> Image.Image:
    if image_format in {"PNG", "WEBP", "TIFF"}:
        return image

    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    background.alpha_composite(image)
    return background.convert("RGB")


def process_single_video_banner(task: BannerTask) -> BannerResult:
    """Process a single video banner by overlaying a logo on each frame."""
    import tempfile
    from moviepy import VideoFileClip, ImageClip, CompositeVideoClip

    source_name = sanitize_stem(task.filename)
    ext = Path(task.filename).suffix.lower() if task.filename else ".mp4"

    temp_in_path = None
    temp_logo_path = None
    temp_out_path = None
    video = None
    logo_clip = None
    composited = None

    try:
        # Write input video bytes to temp file
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_in:
            temp_in.write(task.banner_bytes)
            temp_in_path = temp_in.name

        video = VideoFileClip(temp_in_path)
        
        banner_size = (video.w, video.h)
        exceptions = {(1200, 628), (600, 600), (1080, 1080), (1080, 1920), (1920, 1080)}
        
        video_to_stamp = video
        is_optimized = False
        
        # Banners with width >= 1080 are not scaled down
        is_wide_banner = banner_size[0] >= 1080
        
        if banner_size not in exceptions and not is_wide_banner:
            from .banner_optimizer import resolve_target_dimensions, parse_dimension_from_filename
            target_size = resolve_target_dimensions(banner_size)
            if target_size is None:
                target_size = parse_dimension_from_filename(task.filename)
                
            if target_size is None:
                from .banner_optimizer import OPTIMIZATION_SETTINGS
                allowed = ", ".join(f"{w}x{h}" for w, h in OPTIMIZATION_SETTINGS.supported_dimensions)
                raise ValueError(
                    f"unsupported banner size {banner_size[0]}x{banner_size[1]}; "
                    f"use one of the supported sizes, an exact multiple, or name the file with target dimensions: {allowed}"
                )
            
            video_to_stamp = video.resized(target_size)
            is_optimized = True

        # Get first frame as PIL Image for logo placement and size calculations
        first_frame_arr = video_to_stamp.get_frame(0)
        first_frame_image = Image.fromarray(first_frame_arr)
        
        # Size and position calculations (same as images)
        logo_rgba = _prepare_logo(task.logo_image.copy(), first_frame_image.size)
        placement = select_placement(
            first_frame_image.convert("RGBA"),
            logo_rgba.size,
            task.placement_mode,
        )

        # Write logo to temp PNG file for moviepy ImageClip
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_logo:
            logo_rgba.save(temp_logo.name, format="PNG")
            temp_logo_path = temp_logo.name

        logo_clip = (
            ImageClip(temp_logo_path)
            .with_duration(video_to_stamp.duration)
            .with_position((placement.x, placement.y))
        )

        composited = CompositeVideoClip([video_to_stamp, logo_clip])

        # Select codecs
        codec = "libx264"
        audio_codec = "aac"
        if ext == ".webm":
            codec = "libvpx"
            audio_codec = "libvorbis"

        output_bytes = None

        if is_optimized:
            from .banner_optimizer import OPTIMIZATION_SETTINGS
            target_bytes = OPTIMIZATION_SETTINGS.max_output_bytes
            bitrate_factor = 0.9
            duration = video_to_stamp.duration or 1.0

            for attempt in range(3):
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_out:
                    temp_out_path = temp_out.name

                available_bits = target_bytes * 8 * bitrate_factor
                target_bitrate_bps = max(20000, int(available_bits / duration))

                composited.write_videofile(
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
                    os.unlink(temp_out_path)
                    temp_out_path = None
                    bitrate_factor *= 0.6

            if output_bytes is None:
                raise ValueError(
                    f"could not compress video banner to <= {target_bytes // 1024} KB"
                )
        else:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_out:
                temp_out_path = temp_out.name

            composited.write_videofile(
                temp_out_path,
                codec=codec,
                audio_codec=audio_codec if video.audio is not None else None,
                logger=None,
            )

            with open(temp_out_path, "rb") as f:
                output_bytes = f.read()

        suffix = f"_{sanitize_stem(task.custom_name)}" if task.custom_name else "_processed"
        output_name = f"{source_name}{suffix}{ext}"
        return BannerResult(
            source_name=source_name,
            output_name=output_name,
            output_bytes=output_bytes,
            error=None,
        )

    except Exception as exc:
        return BannerResult(
            source_name=source_name,
            output_name=None,
            output_bytes=None,
            error=f"{task.filename or source_name}: {exc}",
        )

    finally:
        # Close all moviepy clips
        if video:
            try:
                video.close()
            except Exception:
                pass
        if logo_clip:
            try:
                logo_clip.close()
            except Exception:
                pass
        if composited:
            try:
                composited.close()
            except Exception:
                pass
        # Clean up temp files
        for path in (temp_in_path, temp_logo_path, temp_out_path):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass


def process_single_banner(task: BannerTask) -> BannerResult:
    """Process a single banner and return a ZIP-ready result object."""

    source_name = sanitize_stem(task.filename)

    if is_video_file(task.filename):
        return process_single_video_banner(task)

    try:
        banner, banner_format = load_image_bytes(task.banner_bytes)
        banner_rgba = banner.convert("RGBA")
        
        exceptions = {(1200, 628), (600, 600), (1080, 1080), (1080, 1920), (1920, 1080)}
        banner_to_stamp = banner_rgba
        is_optimized = False
        
        # Banners with width >= 1080 are not scaled down
        is_wide_banner = banner_rgba.width >= 1080
        
        if banner_rgba.size not in exceptions and not is_wide_banner:
            from .banner_optimizer import resolve_target_dimensions, parse_dimension_from_filename
            target_size = resolve_target_dimensions(banner_rgba.size)
            if target_size is None:
                target_size = parse_dimension_from_filename(task.filename)
                
            if target_size is None:
                from .banner_optimizer import OPTIMIZATION_SETTINGS
                allowed = ", ".join(f"{w}x{h}" for w, h in OPTIMIZATION_SETTINGS.supported_dimensions)
                raise ValueError(
                    f"unsupported banner size {banner_rgba.width}x{banner_rgba.height}; "
                    f"use one of the supported sizes, an exact multiple, or name the file with target dimensions: {allowed}"
                )
            
            from .banner_optimizer import _resize_banner
            banner_to_stamp = _resize_banner(banner_rgba, target_size)
            is_optimized = True

        logo_rgba = _prepare_logo(task.logo_image.copy(), banner_to_stamp.size)
        placement = select_placement(
            banner_to_stamp,
            logo_rgba.size,
            task.placement_mode,
        )

        composited = banner_to_stamp.copy()
        composited.alpha_composite(logo_rgba, dest=(placement.x, placement.y))

        output_format = normalize_format(banner_format)
        final_image = _flatten_for_format(composited, output_format)
        
        suffix = f"_{sanitize_stem(task.custom_name)}" if task.custom_name else "_processed"
        
        if is_optimized:
            from .banner_optimizer import optimize_banner_bytes
            output_bytes, opt_format = optimize_banner_bytes(final_image)
            if output_bytes is None or opt_format is None:
                raise ValueError("could not compress banner to <= 150 KB")
            output_format = opt_format
        else:
            output_bytes = save_image_to_bytes(final_image, output_format)

        output_name = f"{source_name}{suffix}.{extension_for_format(output_format)}"

        return BannerResult(
            source_name=source_name,
            output_name=output_name,
            output_bytes=output_bytes,
            error=None,
        )
    except Exception as exc:
        return BannerResult(
            source_name=source_name,
            output_name=None,
            output_bytes=None,
            error=f"{task.filename or source_name}: {exc}",
        )


def process_banner_batch(tasks: Sequence[BannerTask]) -> list[BannerResult]:
    """Process banners sequentially for small batches and in parallel for larger ones."""

    if not tasks:
        return []

    if len(tasks) == 1:
        return [process_single_banner(tasks[0])]

    # Use a bounded thread pool for batch image work without oversubscribing tiny batches.
    worker_count = min(8, len(tasks), os.cpu_count() or 1)
    if worker_count <= 1:
        return [process_single_banner(task) for task in tasks]

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(process_single_banner, tasks))

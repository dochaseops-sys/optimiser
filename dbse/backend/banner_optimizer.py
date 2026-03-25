from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Sequence

from PIL import Image

try:  # pragma: no cover - import shim for direct execution
    from .image_processor import load_image_bytes
    from .utils import extension_for_format, sanitize_stem
except ImportError:  # pragma: no cover
    from image_processor import load_image_bytes
    from utils import extension_for_format, sanitize_stem


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
        (120, 600),
        (160, 600),
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
    """Resolve a banner size to a supported target or a smaller exact multiple target."""

    width, height = size
    matches: list[tuple[int, int, tuple[int, int]]] = []

    for target_width, target_height in OPTIMIZATION_SETTINGS.supported_dimensions:
        if (width, height) == (target_width, target_height):
            matches.append((1, target_width * target_height, (target_width, target_height)))
            continue

        if width % target_width != 0 or height % target_height != 0:
            continue

        width_factor = width // target_width
        height_factor = height // target_height
        if width_factor == height_factor and width_factor > 1:
            matches.append(
                (width_factor, target_width * target_height, (target_width, target_height))
            )

    if not matches:
        return None

    scale_factor, _, target = max(matches, key=lambda item: (item[0], -item[1]))
    return target if scale_factor >= 1 else None


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


def process_single_optimization(task: OptimizationTask) -> OptimizationResult:
    """Resize a banner to a supported target and compress it below the size cap."""

    source_name = sanitize_stem(task.filename)

    try:
        banner, _ = load_image_bytes(task.banner_bytes)
        target_size = resolve_target_dimensions(banner.size)
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

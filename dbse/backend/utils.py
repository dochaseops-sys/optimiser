from __future__ import annotations

import io
import mimetypes
import os
import re
import zipfile
from pathlib import Path
from typing import Iterable, Sequence


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def is_video_file(filename: str | None) -> bool:
    """Return True if the filename has a supported video extension."""
    if not filename:
        return False
    ext = Path(filename).suffix.lower()
    return ext in {".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v"}



def sanitize_stem(filename: str | None, fallback: str = "image") -> str:
    """Return a filesystem- and zip-safe stem for an uploaded filename."""

    if not filename:
        return fallback

    stem = Path(os.path.basename(filename)).stem
    stem = _SAFE_NAME_RE.sub("_", stem).strip("._-")
    return stem or fallback


def normalize_format(image_format: str | None) -> str:
    """Map Pillow formats to a consistent save format."""

    if not image_format:
        return "PNG"

    fmt = image_format.upper()
    if fmt == "JPG":
        return "JPEG"
    if fmt in {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}:
        return fmt
    return "PNG"


def extension_for_format(image_format: str) -> str:
    """Return a reasonable file extension for a Pillow save format."""

    fmt = normalize_format(image_format)
    if fmt == "JPEG":
        return "jpg"
    if fmt == "PNG":
        return "png"
    if fmt == "WEBP":
        return "webp"
    if fmt == "BMP":
        return "bmp"
    if fmt == "TIFF":
        return "tiff"
    return "png"


def media_type_for_filename(filename: str) -> str:
    """Return a best-effort media type for a generated filename."""

    media_type, _ = mimetypes.guess_type(filename)
    return media_type or "application/octet-stream"


def unique_name(name: str, existing: set[str]) -> str:
    """Ensure ZIP entry names do not collide."""

    if name not in existing:
        existing.add(name)
        return name

    stem, dot, suffix = name.rpartition(".")
    if not dot:
        stem, suffix = name, ""

    index = 2
    while True:
        candidate = f"{stem}_{index}"
        if suffix:
            candidate = f"{candidate}.{suffix}"
        if candidate not in existing:
            existing.add(candidate)
            return candidate
        index += 1


def save_image_to_bytes(image, image_format: str) -> bytes:
    """Serialize a Pillow image to bytes using a stable format."""

    output = io.BytesIO()
    save_kwargs = {}

    if image_format == "JPEG":
        save_kwargs["quality"] = 95
        save_kwargs["optimize"] = True
    elif image_format == "PNG":
        save_kwargs["optimize"] = True

    image.save(output, format=image_format, **save_kwargs)
    return output.getvalue()


def build_zip_bytes(
    files: Sequence[tuple[str, bytes]],
    report_lines: Iterable[str] | None = None,
) -> bytes:
    """Create an in-memory ZIP archive containing processed images and a report."""

    buffer = io.BytesIO()
    used_names: set[str] = set()

    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files:
            archive.writestr(unique_name(name, used_names), data)

        if report_lines:
            report = "\n".join(report_lines).strip()
            if report:
                archive.writestr(unique_name("errors.txt", used_names), report + "\n")

    return buffer.getvalue()

from __future__ import annotations

import io
from pathlib import Path
import sys
import zipfile

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.background_generators import extract_dominant_color
from backend.image_processor import load_image_bytes
from backend.main import app
from backend.resizer import (
    BLUR_EXPAND,
    DOMINANT_COLOR_BACKGROUND,
    FIT_EXPAND,
    SOLID_BACKGROUND,
    adapt_image,
    parse_target_dimensions,
)


def _image_bytes(image: Image.Image, *, format_name: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=format_name)
    return buffer.getvalue()


def _solid_bytes(size: tuple[int, int], color: tuple[int, ...], *, mode: str = "RGBA") -> bytes:
    return _image_bytes(Image.new(mode, size, color))


def _content_bounds(
    image: Image.Image,
    color: tuple[int, int, int],
    *,
    tolerance: int = 0,
) -> tuple[int, int, int, int]:
    pixels = np.array(image.convert("RGBA"))
    target = np.array(color, dtype=np.int16)
    diff = np.abs(pixels[:, :, :3].astype(np.int16) - target)
    mask = (
        (diff[:, :, 0] <= tolerance)
        & (diff[:, :, 1] <= tolerance)
        & (diff[:, :, 2] <= tolerance)
        & (pixels[:, :, 3] > 0)
    )
    ys, xs = np.where(mask)
    assert xs.size and ys.size
    return xs.min(), ys.min(), xs.max(), ys.max()


def test_parse_target_dimensions_accepts_repeated_and_comma_separated_targets() -> None:
    assert parse_target_dimensions(["300x250, 728x90", "160x600"], None, None) == [
        parse_target_dimensions([], "300", "250")[0],
        parse_target_dimensions([], "728", "90")[0],
        parse_target_dimensions([], "160", "600")[0],
    ]


def test_fit_expand_preserves_landscape_image_without_cropping_or_stretching() -> None:
    source = Image.new("RGBA", (400, 100), (20, 120, 20, 255))
    for inset in range(8):
        for x in range(source.width):
            source.putpixel((x, inset), (255, 0, 0, 255))
            source.putpixel((x, source.height - 1 - inset), (255, 0, 0, 255))
        for y in range(source.height):
            source.putpixel((inset, y), (255, 0, 0, 255))
            source.putpixel((source.width - 1 - inset, y), (255, 0, 0, 255))
    output = adapt_image(source, (300, 250), mode=FIT_EXPAND)

    assert output.size == (300, 250)
    left, top, right, bottom = _content_bounds(output, (255, 0, 0), tolerance=6)
    assert (right - left + 1, bottom - top + 1) == (300, 75)
    assert (left, top) == (0, 87)


def test_fit_expand_preserves_portrait_image_without_cropping_or_stretching() -> None:
    source = Image.new("RGBA", (100, 400), (20, 120, 20, 255))
    for inset in range(8):
        for x in range(source.width):
            source.putpixel((x, inset), (0, 128, 255, 255))
            source.putpixel((x, source.height - 1 - inset), (0, 128, 255, 255))
        for y in range(source.height):
            source.putpixel((inset, y), (0, 128, 255, 255))
            source.putpixel((source.width - 1 - inset, y), (0, 128, 255, 255))
    output = adapt_image(source, (300, 250), mode=FIT_EXPAND)

    assert output.size == (300, 250)
    left, top, right, bottom = _content_bounds(output, (0, 128, 255), tolerance=6)
    assert (right - left + 1, bottom - top + 1) == (62, 250)
    assert top == 0
    assert left in {119, 120}


def test_solid_background_uses_configured_color() -> None:
    source = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
    output = adapt_image(
        source,
        (200, 100),
        mode=SOLID_BACKGROUND,
        background_color="#123456",
    )

    assert output.getpixel((0, 0)) == (18, 52, 86, 255)


def test_dominant_color_background_extracts_visible_source_color() -> None:
    source = Image.new("RGBA", (80, 80), (14, 90, 180, 255))
    output = adapt_image(source, (200, 100), mode=DOMINANT_COLOR_BACKGROUND)

    assert extract_dominant_color(source) == (14, 90, 180, 255)
    assert output.getpixel((0, 0)) == (14, 90, 180, 255)


def test_blur_expand_preserves_foreground_and_fills_canvas() -> None:
    source = Image.new("RGBA", (100, 200), (0, 0, 255, 255))
    output = adapt_image(source, (300, 120), mode=BLUR_EXPAND, blur_radius=8)

    assert output.size == (300, 120)
    left, top, right, bottom = _content_bounds(output, (0, 0, 255))
    assert (right - left + 1, bottom - top + 1) == (60, 120)
    assert left == 120
    assert top == 0


def test_resize_endpoint_returns_single_png_for_transparent_png() -> None:
    client = TestClient(app)
    source = _solid_bytes((80, 80), (255, 0, 0, 128))

    response = client.post(
        "/resize",
        data={
            "target_width": "200",
            "target_height": "100",
            "mode": "solid_background",
            "background_color": "#00000000",
        },
        files=[("image", ("transparent.png", source, "image/png"))],
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert 'filename="transparent_200x100_solid_background.png"' in response.headers["content-disposition"]

    image, image_format = load_image_bytes(response.content)
    assert image_format == "PNG"
    assert image.size == (200, 100)
    assert image.convert("RGBA").getpixel((0, 0))[3] == 0


def test_resize_endpoint_returns_zip_for_multiple_targets() -> None:
    client = TestClient(app)

    response = client.post(
        "/resize",
        data={
            "targets": "300x250,728x90",
            "mode": "fit_expand",
        },
        files=[("images", ("creative.png", _solid_bytes((200, 200), (255, 0, 0, 255)), "image/png"))],
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    assert sorted(archive.namelist()) == [
        "creative_300x250_fit_expand.png",
        "creative_728x90_fit_expand.png",
    ]

from __future__ import annotations

import io
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.image_processor import (
    SETTINGS,
    BannerTask,
    calculate_logo_size,
    choose_position,
    load_image_bytes,
    process_single_banner,
    resize_to_fit,
)
from backend.main import app


def _make_image_bytes(
    size: tuple[int, int],
    *,
    mode: str = "RGBA",
    color=(255, 255, 255, 255),
) -> bytes:
    image = Image.new(mode, size, color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _make_noisy_banner(noisy_side: str) -> Image.Image:
    width, height = 1000, 400
    image = Image.new("RGB", (width, height), "white")
    pixels = np.array(image)
    region_width = int(round(width * 0.20))
    region_height = int(round(height * 0.15))
    noise = (np.indices((region_height, region_width)).sum(axis=0) % 2) * 255
    noise_rgb = np.repeat(noise[:, :, None], 3, axis=2).astype(np.uint8)

    if noisy_side == "top-left":
        pixels[:region_height, :region_width] = noise_rgb
    else:
        pixels[:region_height, width - region_width:] = noise_rgb

    return Image.fromarray(pixels)


def _red_bounds(image: Image.Image) -> tuple[int, int, int, int]:
    pixels = np.array(image.convert("RGBA"))
    mask = (
        (pixels[:, :, 0] > 220)
        & (pixels[:, :, 1] < 40)
        & (pixels[:, :, 2] < 40)
        & (pixels[:, :, 3] > 0)
    )
    ys, xs = np.where(mask)
    assert xs.size and ys.size
    return xs.min(), ys.min(), xs.max(), ys.max()


def test_load_image_bytes_accepts_valid_image_bytes() -> None:
    image, image_format = load_image_bytes(_make_image_bytes((160, 90)))

    assert image.size == (160, 90)
    assert image_format == "PNG"


@pytest.mark.parametrize("payload", [b"not-an-image", _make_image_bytes((16, 16))[:12]])
def test_load_image_bytes_rejects_corrupted_files(payload: bytes) -> None:
    with pytest.raises(ValueError):
        load_image_bytes(payload)


@pytest.mark.parametrize(
    ("source_size", "max_size", "expected"),
    [
        ((64, 18), (500, 500), (64, 18)),
        ((4096, 1024), (1024, 256), (1024, 256)),
        ((800, 2000), (300, 300), (120, 300)),
    ],
)
def test_resize_to_fit_preserves_aspect_ratio_and_bounds(
    source_size: tuple[int, int],
    max_size: tuple[int, int],
    expected: tuple[int, int],
) -> None:
    assert resize_to_fit(source_size, max_size, allow_upscale=False) == expected


def test_calculate_logo_size_uses_base_rule_for_standard_banners() -> None:
    banner_width = SETTINGS.large_banner_dimension_threshold - 1
    assert calculate_logo_size((banner_width, 400), (200, 100)) == (16, 8)


def test_calculate_logo_size_uses_8x8_minimum_for_small_square_logos() -> None:
    assert calculate_logo_size((300, 50), (200, 200)) == (8, 8)


def test_calculate_logo_size_keeps_aspect_ratio_while_hitting_small_banner_minimum() -> None:
    assert calculate_logo_size((300, 50), (200, 100)) == (16, 8)


def test_calculate_logo_size_scales_up_for_large_banners() -> None:
    assert calculate_logo_size((SETTINGS.large_banner_dimension_threshold, 400), (200, 100)) == (25, 12)
    assert calculate_logo_size((SETTINGS.xl_banner_dimension_threshold, 600), (200, 100)) == (56, 28)


def test_choose_position_prefers_the_lower_noise_corner() -> None:
    assert choose_position(_make_noisy_banner("top-left"), "smart") == "top-right"
    assert choose_position(_make_noisy_banner("top-right"), "smart") == "top-left"


@pytest.mark.parametrize("placement_mode", ["top-left", "top-right"])
def test_process_single_banner_places_logo_only_in_allowed_corner(placement_mode: str) -> None:
    banner_bytes = _make_image_bytes((1000, 400), color=(255, 255, 255, 255))
    logo = Image.new("RGBA", (200, 100), (255, 0, 0, 255))
    result = process_single_banner(
        BannerTask(
            filename="banner.png",
            banner_bytes=banner_bytes,
            logo_image=logo,
            placement_mode=placement_mode,
        )
    )

    assert result.error is None
    assert result.output_bytes is not None

    processed_image, _ = load_image_bytes(result.output_bytes)
    left, top, right, bottom = _red_bounds(processed_image)
    logo_width, logo_height = calculate_logo_size((1000, 400), logo.size)
    margin = 0
    expected_left = margin if placement_mode == "top-left" else 1000 - margin - logo_width

    assert left == expected_left
    assert top == margin
    assert (right - left + 1, bottom - top + 1) == (logo_width, logo_height)

    opposite_corner_x = 1000 - margin - 1 if placement_mode == "top-left" else margin
    opposite_pixel = processed_image.convert("RGBA").getpixel((opposite_corner_x, margin))
    assert opposite_pixel[:3] == (255, 255, 255)


def test_process_single_banner_preserves_logo_whitespace() -> None:
    banner_bytes = _make_image_bytes((300, 250), color=(0, 0, 0, 255))
    padded_logo = Image.new("RGB", (300, 300), (255, 255, 255))
    logo_block = Image.new("RGB", (180, 180), (255, 0, 0))
    padded_logo.paste(logo_block, (60, 60))

    result = process_single_banner(
        BannerTask(
            filename="banner.png",
            banner_bytes=banner_bytes,
            logo_image=padded_logo,
            placement_mode="top-left",
        )
    )

    assert result.error is None
    processed_image, _ = load_image_bytes(result.output_bytes)
    left, top, _, _ = _red_bounds(processed_image)

    top_left_pixel = processed_image.convert("RGB").getpixel((0, 0))
    assert all(channel >= 235 for channel in top_left_pixel)
    assert left > 0
    assert top > 0


def test_process_single_banner_returns_an_error_for_corrupted_banner() -> None:
    logo = Image.new("RGBA", (100, 50), (255, 0, 0, 255))
    result = process_single_banner(
        BannerTask(
            filename="broken.png",
            banner_bytes=b"not-an-image",
            logo_image=logo,
            placement_mode="smart",
        )
    )

    assert result.output_bytes is None
    assert result.output_name is None
    assert result.error is not None


def test_process_endpoint_returns_a_downloadable_zip() -> None:
    client = TestClient(app)
    banner_one = _make_image_bytes((1200, 400))
    banner_two = _make_image_bytes((900, 300))
    logo = _make_image_bytes((240, 120), color=(255, 0, 0, 255))

    response = client.post(
        "/process",
        data={"placement_mode": "smart"},
        files=[
            ("banners", ("banner-1.png", banner_one, "image/png")),
            ("banners", ("banner-2.png", banner_two, "image/png")),
            ("logo", ("logo.png", logo, "image/png")),
        ],
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    names = sorted(archive.namelist())

    assert "banner-1_processed.png" in names
    assert "banner-2_processed.png" in names
    assert "errors.txt" not in names


def test_process_endpoint_returns_single_image_for_single_banner() -> None:
    client = TestClient(app)
    response = client.post(
        "/process",
        data={"placement_mode": "top-left"},
        files=[
            ("banners", ("banner.png", _make_image_bytes((1200, 400)), "image/png")),
            ("logo", ("logo.png", _make_image_bytes((240, 120), color=(255, 0, 0, 255)), "image/png")),
        ],
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert 'filename="banner_processed.png"' in response.headers["content-disposition"]


def test_process_endpoint_accepts_frontend_alias_field_name() -> None:
    client = TestClient(app)
    response = client.post(
        "/process",
        data={"placement": "top-left"},
        files=[
            ("banners", ("banner.png", _make_image_bytes((1200, 400)), "image/png")),
            ("logo", ("logo.png", _make_image_bytes((240, 120), color=(255, 0, 0, 255)), "image/png")),
        ],
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_frontend_route_serves_the_ui() -> None:
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "DBSE" in response.text
    assert "Optimize Banners" in response.text
    assert "Adapt Creatives" in response.text
    assert "Output Preview" in response.text


def test_optimiser_route_serves_the_ui() -> None:
    client = TestClient(app)

    for route in ("/optimiser", "/optimiser/"):
        response = client.get(route)

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "DBSE" in response.text
        assert "Optimize Banners" in response.text
        assert "Adapt Creatives" in response.text
        assert "Output Preview" in response.text


def test_process_endpoint_accepts_zip_file() -> None:
    client = TestClient(app)
    banner_one = _make_image_bytes((1200, 400))
    banner_two = _make_image_bytes((900, 300))
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("banner-1.png", banner_one)
        zf.writestr("banner-2.png", banner_two)
        zf.writestr("__MACOSX/._banner-1.png", b"junk")
        zf.writestr("readme.txt", b"not an image")
    zip_bytes = zip_buffer.getvalue()
    
    logo = _make_image_bytes((240, 120), color=(255, 0, 0, 255))
    
    response = client.post(
        "/process",
        data={"placement_mode": "smart"},
        files=[
            ("banners", ("banners.zip", zip_bytes, "application/zip")),
            ("logo", ("logo.png", logo, "image/png")),
        ],
    )
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    names = sorted(archive.namelist())
    assert "banner-1_processed.png" in names
    assert "banner-2_processed.png" in names
    assert len(names) == 2


def test_process_endpoint_uses_custom_name_for_single_file() -> None:
    client = TestClient(app)
    response = client.post(
        "/process",
        data={"placement_mode": "top-left", "custom_name": "my-special-banner"},
        files=[
            ("banners", ("banner.png", _make_image_bytes((1200, 400)), "image/png")),
            ("logo", ("logo.png", _make_image_bytes((240, 120), color=(255, 0, 0, 255)), "image/png")),
        ],
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert 'filename="my-special-banner.png"' in response.headers["content-disposition"]


def test_process_endpoint_uses_custom_name_for_zip() -> None:
    client = TestClient(app)
    banner_one = _make_image_bytes((1200, 400))
    banner_two = _make_image_bytes((900, 300))
    logo = _make_image_bytes((240, 120), color=(255, 0, 0, 255))

    response = client.post(
        "/process",
        data={"placement_mode": "smart", "custom_name": "my-campaign-folder"},
        files=[
            ("banners", ("banner-1.png", banner_one, "image/png")),
            ("banners", ("banner-2.png", banner_two, "image/png")),
            ("logo", ("logo.png", logo, "image/png")),
        ],
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert 'filename="my-campaign-folder.zip"' in response.headers["content-disposition"]


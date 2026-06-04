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

from backend.banner_optimizer import OPTIMIZATION_SETTINGS, resolve_target_dimensions, parse_dimension_from_filename
from backend.main import app


def _make_image_bytes(
    size: tuple[int, int],
    *,
    noisy: bool = False,
    format_name: str = "PNG",
) -> bytes:
    if noisy:
        rng = np.random.default_rng(42)
        pixels = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
    else:
        image = Image.new("RGB", size, (255, 255, 255))

    buffer = io.BytesIO()
    image.save(buffer, format=format_name)
    return buffer.getvalue()


def test_resolve_target_dimensions_returns_exact_supported_size() -> None:
    assert resolve_target_dimensions((300, 250)) == (300, 250)
    assert resolve_target_dimensions((970, 250)) == (970, 250)


def test_resolve_target_dimensions_reduces_exact_multiples() -> None:
    assert resolve_target_dimensions((900, 750)) == (300, 250)
    assert resolve_target_dimensions((640, 100)) == (320, 50)
    assert resolve_target_dimensions((960, 1440)) == (320, 480)


def test_resolve_target_dimensions_rejects_unsupported_sizes() -> None:
    assert resolve_target_dimensions((301, 251)) is None
    assert resolve_target_dimensions((640, 120)) is None


def test_parse_dimension_from_filename() -> None:
    assert parse_dimension_from_filename("banner_300x250.png") == (300, 250)
    assert parse_dimension_from_filename("banner_300 x 250_final.jpg") == (300, 250)
    assert parse_dimension_from_filename("320x100_banner.png") == (320, 100)
    assert parse_dimension_from_filename("120_x_600_dochase_smart_banking.png") == (120, 600)
    assert parse_dimension_from_filename("970_x_90_dochase_smart_banking.png") == (970, 90)
    assert parse_dimension_from_filename("no_dimensions.png") is None
    assert parse_dimension_from_filename("banner_400x400.png") is None


def test_optimize_endpoint_returns_single_file_under_150kb_and_resized() -> None:
    client = TestClient(app)
    response = client.post(
        "/optimize",
        files=[
            (
                "banners",
                ("banner.png", _make_image_bytes((900, 750), noisy=True), "image/png"),
            ),
        ],
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) <= OPTIMIZATION_SETTINGS.max_output_bytes

    optimized = Image.open(io.BytesIO(response.content))
    assert optimized.size == (300, 250)


def test_optimize_endpoint_returns_zip_for_multiple_banners() -> None:
    client = TestClient(app)
    response = client.post(
        "/optimize",
        files=[
            ("banners", ("one.png", _make_image_bytes((300, 250), noisy=True), "image/png")),
            ("banners", ("two.png", _make_image_bytes((640, 100), noisy=True), "image/png")),
        ],
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    names = sorted(archive.namelist())

    assert "one_optimized.jpg" in names
    assert "two_optimized.jpg" in names


def test_optimize_endpoint_returns_error_zip_for_unsupported_banner() -> None:
    client = TestClient(app)
    response = client.post(
        "/optimize",
        files=[
            ("banners", ("bad.png", _make_image_bytes((301, 251), noisy=True), "image/png")),
        ],
    )

    assert response.status_code == 422
    assert response.headers["content-type"] == "application/zip"

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    names = archive.namelist()

    assert "errors.txt" in names
    assert "unsupported banner size 301x251" in archive.read("errors.txt").decode("utf-8")


def test_optimize_endpoint_resolves_via_filename() -> None:
    client = TestClient(app)
    response = client.post(
        "/optimize",
        files=[
            (
                "banners",
                ("my_banner_300x250_final.png", _make_image_bytes((400, 300), noisy=True), "image/png"),
            ),
        ],
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) <= OPTIMIZATION_SETTINGS.max_output_bytes

    optimized = Image.open(io.BytesIO(response.content))
    assert optimized.size == (300, 250)

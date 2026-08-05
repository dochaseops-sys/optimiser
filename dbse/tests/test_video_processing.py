from __future__ import annotations

import io
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.image_processor import BannerTask, process_single_banner
from backend.resizer import ResizeTask, ResizeTarget, process_single_resize
from backend.banner_optimizer import OptimizationTask, process_single_optimization
from backend.utils import is_video_file
from backend.main import app


def _make_test_video_bytes(size: tuple[int, int] = (300, 250), duration: float = 1.0) -> bytes:
    from moviepy import ColorClip
    clip = ColorClip(size=size, color=(128, 128, 128), duration=duration)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_f:
        temp_path = temp_f.name
    try:
        clip.write_videofile(temp_path, fps=5, codec="libx264", logger=None)
        with open(temp_path, "rb") as f:
            video_bytes = f.read()
    finally:
        clip.close()
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    return video_bytes


def _make_image_bytes(size: tuple[int, int], color=(255, 0, 0, 255)) -> bytes:
    image = Image.new("RGBA", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_is_video_file() -> None:
    assert is_video_file("banner.mp4") is True
    assert is_video_file("banner.WEBM") is True
    assert is_video_file("banner.png") is False
    assert is_video_file(None) is False


def test_process_video_banner_stamping() -> None:
    video_bytes = _make_test_video_bytes((300, 250))
    logo_image = Image.new("RGBA", (50, 50), (255, 0, 0, 255))
    
    task = BannerTask(
        filename="test_banner.mp4",
        banner_bytes=video_bytes,
        logo_image=logo_image,
        placement_mode="top-right",
    )
    result = process_single_banner(task)
    
    assert result.error is None
    assert result.output_name == "test_banner_processed.mp4"
    assert result.output_bytes is not None
    assert len(result.output_bytes) > 0


def test_process_video_resize_adaptation() -> None:
    video_bytes = _make_test_video_bytes((300, 200))
    
    task = ResizeTask(
        filename="creative.mp4",
        image_bytes=video_bytes,
        target=ResizeTarget(300, 250),
        mode="blur_expand",
        blur_radius=8.0,
    )
    result = process_single_resize(task)
    
    assert result.error is None
    assert result.output_name == "creative_300x250_blur_expand.mp4"
    assert result.output_bytes is not None
    assert len(result.output_bytes) > 0


def test_process_video_optimization() -> None:
    # 600x500 is a multiple of supported size 300x250
    video_bytes = _make_test_video_bytes((600, 500))
    
    task = OptimizationTask(
        filename="banner.mp4",
        banner_bytes=video_bytes,
    )
    result = process_single_optimization(task)
    
    assert result.error is None
    assert result.output_name == "banner_optimized.mp4"
    assert result.output_bytes is not None
    assert len(result.output_bytes) <= 150 * 1024


def test_main_process_endpoint_accepts_video() -> None:
    client = TestClient(app)
    video_bytes = _make_test_video_bytes((300, 250))
    logo_bytes = _make_image_bytes((40, 40))

    response = client.post(
        "/process",
        data={"placement_mode": "smart"},
        files=[
            ("banners", ("video_banner.mp4", video_bytes, "video/mp4")),
            ("logo", ("logo.png", logo_bytes, "image/png")),
        ],
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert 'filename="video_banner_processed.mp4"' in response.headers["content-disposition"]


def test_process_video_banner_stamping_uses_custom_name() -> None:
    video_bytes = _make_test_video_bytes((300, 250))
    logo_image = Image.new("RGBA", (50, 50), (255, 0, 0, 255))
    
    task = BannerTask(
        filename="test_banner.mp4",
        banner_bytes=video_bytes,
        logo_image=logo_image,
        placement_mode="top-right",
        custom_name="camp-123",
    )
    result = process_single_banner(task)
    
    assert result.error is None
    assert result.output_name == "test_banner_camp-123.mp4"


def test_process_video_optimization_resolves_via_filename() -> None:
    # 400x300 is not a multiple of any supported size, but filename has 300x250
    video_bytes = _make_test_video_bytes((400, 300))
    task = OptimizationTask(
        filename="banner_300x250_camp.mp4",
        banner_bytes=video_bytes,
    )
    result = process_single_optimization(task)
    
    assert result.error is None
    assert result.output_name == "banner_300x250_camp_optimized.mp4"
    assert result.output_bytes is not None
    assert len(result.output_bytes) <= 150 * 1024


def test_process_single_video_banner_preserves_original_dimensions() -> None:
    video_bytes = _make_test_video_bytes((600, 500))
    logo_image = Image.new("RGBA", (50, 50), (255, 0, 0, 255))
    task = BannerTask(
        filename="test_video_multiple.mp4",
        banner_bytes=video_bytes,
        logo_image=logo_image,
        placement_mode="top-right",
    )
    result = process_single_banner(task)
    assert result.error is None
    # Verify the video size retains original 600x500 dimensions
    from moviepy import VideoFileClip
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(result.output_bytes)
        path = f.name
    try:
        clip = VideoFileClip(path)
        assert clip.size == [600, 500]
        clip.close()
    finally:
        import os
        if os.path.exists(path):
            os.unlink(path)


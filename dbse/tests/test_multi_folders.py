from __future__ import annotations

import io
import zipfile
from pathlib import Path
import sys

from fastapi.testclient import TestClient
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app, _get_output_names, _expand_output_folders


def _png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (320, 200), "white").save(output, format="PNG")
    return output.getvalue()


def test_multiple_names_are_sanitized_and_deduplicated() -> None:
    class Form:
        def getlist(self, key):
            return ["Client A\nClient A\n../Client B"] if key == "custom_names" else []
    single, folders = _get_output_names(Form())
    assert single is None
    assert folders == ["Client_A", "Client_B"]


def test_zip_entries_are_grouped_without_reprocessing() -> None:
    entries = _expand_output_folders([("banner.png", b"data")], ["First", "Second"])
    assert entries == [("First/banner.png", b"data"), ("Second/banner.png", b"data")]


def test_optimize_multiple_folders_returns_zip() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/optimize",
            data={"custom_names": "Campaign_A\nCampaign_B"},
            files={"banners": ("banner.png", _png(), "image/png")},
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert sorted(archive.namelist()) == [
            "Campaign_A/banner_optimized.png",
            "Campaign_B/banner_optimized.png",
        ] or all(
            name.startswith(("Campaign_A/", "Campaign_B/"))
            for name in archive.namelist()
        )


def test_more_than_one_hundred_names_rejected() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/optimize",
            data={"custom_names": "\n".join(f"Folder_{n}" for n in range(101))},
            files={"banners": ("banner.png", _png(), "image/png")},
        )
    assert response.status_code == 422


def test_validation_error_is_explained_in_response_header() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/process",
            data={"custom_names": "Folder_A\\nFolder_B", "placement_mode": "smart"},
            files={"banners": ("banner.png", _png(), "image/png")},
        )
    assert response.status_code == 422
    assert "No logo image was uploaded" in response.headers["x-processing-error"]

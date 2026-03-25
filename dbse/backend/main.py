from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import os
from typing import Any, Iterable

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

try:  # pragma: no cover - import shim for direct execution
    from .banner_optimizer import OptimizationTask, process_optimization_batch
    from .image_processor import (
        ALLOWED_PLACEMENT_MODES,
        BannerTask,
        load_image_bytes,
        process_banner_batch,
    )
    from .utils import build_zip_bytes, media_type_for_filename
except ImportError:  # pragma: no cover
    from banner_optimizer import OptimizationTask, process_optimization_batch
    from image_processor import (
        ALLOWED_PLACEMENT_MODES,
        BannerTask,
        load_image_bytes,
        process_banner_batch,
    )
    from utils import build_zip_bytes, media_type_for_filename


app = FastAPI(title="DBSE Banner Processor", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_INDEX = Path(__file__).resolve().parents[1] / "frontend" / "index.html"
DIST_INDEX = Path(__file__).resolve().parents[1] / "dist" / "index.html"


def _chunked(items: list[Any], size: int) -> Iterator[list[Any]]:
    for index in range(0, len(items), size):
        yield items[index:index + size]


def _is_upload(value: Any) -> bool:
    return hasattr(value, "read") and hasattr(value, "filename")


def _collect_uploads(form: Any, keys: Iterable[str]) -> list[Any]:
    uploads: list[Any] = []
    for key in keys:
        if hasattr(form, "getlist"):
            values = form.getlist(key)
        else:
            value = form.get(key)
            values = [] if value is None else [value]
        uploads.extend([value for value in values if _is_upload(value)])
    return uploads


def _first_text(form: Any, keys: Iterable[str]) -> str | None:
    for key in keys:
        if hasattr(form, "getlist"):
            values = form.getlist(key)
            for value in values:
                if isinstance(value, str) and value.strip():
                    return value.strip()
        else:
            value = form.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _error_zip_response(
    messages: Iterable[str],
    http_status: int,
    *,
    filename: str = "dbse_processed_banners_errors.zip",
) -> Response:
    zip_bytes = build_zip_bytes([], messages)
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers=headers,
        status_code=http_status,
    )


def _frontend_index_path() -> Path | None:
    if DIST_INDEX.exists():
        return DIST_INDEX
    if FRONTEND_INDEX.exists():
        return FRONTEND_INDEX
    return None


def _normalize_mode(mode: str) -> str:
    return mode.strip().lower().replace("_", "-")


def _download_response(
    processed_files: list[tuple[str, bytes]],
    errors: list[str],
    *,
    empty_message: str,
    zip_filename: str,
    error_filename: str,
) -> Response:
    if not processed_files:
        return _error_zip_response(
            errors or [empty_message],
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            filename=error_filename,
        )

    if len(processed_files) == 1 and not errors:
        output_name, output_bytes = processed_files[0]
        headers = {"Content-Disposition": f'attachment; filename="{output_name}"'}
        return Response(
            content=output_bytes,
            media_type=media_type_for_filename(output_name),
            headers=headers,
        )

    zip_bytes = build_zip_bytes(processed_files, errors or None)
    headers = {"Content-Disposition": f'attachment; filename="{zip_filename}"'}
    return Response(content=zip_bytes, media_type="application/zip", headers=headers)


@app.get("/", include_in_schema=False)
@app.get("/optimiser", include_in_schema=False)
@app.get("/optimiser/", include_in_schema=False)
async def frontend() -> Response:
    frontend_index = _frontend_index_path()
    if frontend_index is not None:
        return FileResponse(frontend_index)
    return Response(
        content="DBSE frontend not found. Open dbse/frontend/index.html directly.",
        media_type="text/plain",
    )


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/process")
async def process(request: Request) -> Response:
    form = await request.form()

    placement_mode_raw = _first_text(form, ("placement_mode", "placement"))
    if not placement_mode_raw:
        return _error_zip_response(
            ["Missing required field: placement_mode."],
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            filename="dbse_processed_banners_errors.zip",
        )

    placement_mode = _normalize_mode(placement_mode_raw)
    if placement_mode not in ALLOWED_PLACEMENT_MODES:
        return _error_zip_response(
            [
                "Invalid placement_mode.",
                "Expected one of: smart, top-left, top-right.",
            ],
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            filename="dbse_processed_banners_errors.zip",
        )

    banner_uploads = _collect_uploads(
        form,
        ("banner_images", "banners", "banner_files", "banner"),
    )
    if not banner_uploads:
        return _error_zip_response(
            ["No banner images were uploaded."],
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            filename="dbse_processed_banners_errors.zip",
        )

    logo_uploads = _collect_uploads(form, ("logo_image", "logo", "logo_file"))
    if not logo_uploads:
        return _error_zip_response(
            ["No logo image was uploaded."],
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            filename="dbse_processed_banners_errors.zip",
        )

    logo_upload = logo_uploads[0]
    logo_bytes = await logo_upload.read()
    if not logo_bytes:
        return _error_zip_response(
            ["The uploaded logo image is empty."],
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            filename="dbse_processed_banners_errors.zip",
        )

    try:
        logo_image, _ = load_image_bytes(logo_bytes)
    except ValueError:
        return _error_zip_response(
            ["The uploaded logo image is corrupted or unsupported."],
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            filename="dbse_processed_banners_errors.zip",
        )

    processed_files = []
    errors = []
    chunk_size = max(1, min(16, (os.cpu_count() or 1) * 2))

    for chunk_index, upload_chunk in enumerate(_chunked(banner_uploads, chunk_size)):
        tasks = []
        for offset, upload in enumerate(upload_chunk):
            banner_bytes = await upload.read()
            source_index = (chunk_index * chunk_size) + offset + 1
            source_name = upload.filename or f"banner_{source_index}"
            if not banner_bytes:
                errors.append(f"{source_name}: uploaded file is empty")
                continue

            tasks.append(
                BannerTask(
                    filename=source_name,
                    banner_bytes=banner_bytes,
                    logo_image=logo_image,
                    placement_mode=placement_mode,
                )
            )

        for result in process_banner_batch(tasks):
            if result.output_name and result.output_bytes:
                processed_files.append((result.output_name, result.output_bytes))
            if result.error:
                errors.append(result.error)

    return _download_response(
        processed_files,
        errors,
        empty_message="No banners could be processed.",
        zip_filename="dbse_processed_banners.zip",
        error_filename="dbse_processed_banners_errors.zip",
    )


@app.post("/optimize")
async def optimize(request: Request) -> Response:
    form = await request.form()

    banner_uploads = _collect_uploads(
        form,
        ("banner_images", "banners", "banner_files", "banner"),
    )
    if not banner_uploads:
        return _error_zip_response(
            ["No banner images were uploaded."],
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            filename="dbse_optimized_banners_errors.zip",
        )

    processed_files = []
    errors = []
    chunk_size = max(1, min(16, (os.cpu_count() or 1) * 2))

    for chunk_index, upload_chunk in enumerate(_chunked(banner_uploads, chunk_size)):
        tasks = []
        for offset, upload in enumerate(upload_chunk):
            banner_bytes = await upload.read()
            source_index = (chunk_index * chunk_size) + offset + 1
            source_name = upload.filename or f"banner_{source_index}"
            if not banner_bytes:
                errors.append(f"{source_name}: uploaded file is empty")
                continue

            tasks.append(
                OptimizationTask(
                    filename=source_name,
                    banner_bytes=banner_bytes,
                )
            )

        for result in process_optimization_batch(tasks):
            if result.output_name and result.output_bytes:
                processed_files.append((result.output_name, result.output_bytes))
            if result.error:
                errors.append(result.error)

    return _download_response(
        processed_files,
        errors,
        empty_message="No banners could be optimized.",
        zip_filename="dbse_optimized_banners.zip",
        error_filename="dbse_optimized_banners_errors.zip",
    )

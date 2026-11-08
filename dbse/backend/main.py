from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import io
import os
import zipfile
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
    from .resizer import (
        ALLOWED_RESIZE_MODES,
        ResizeTask,
        normalize_resize_mode,
        parse_target_dimensions,
        process_resize_batch,
    )
    from .utils import build_zip_bytes, media_type_for_filename, sanitize_stem
except ImportError:  # pragma: no cover
    from banner_optimizer import OptimizationTask, process_optimization_batch
    from image_processor import (
        ALLOWED_PLACEMENT_MODES,
        BannerTask,
        load_image_bytes,
        process_banner_batch,
    )
    from resizer import (
        ALLOWED_RESIZE_MODES,
        ResizeTask,
        normalize_resize_mode,
        parse_target_dimensions,
        process_resize_batch,
    )
    from utils import build_zip_bytes, media_type_for_filename, sanitize_stem


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


def _all_text(form: Any, keys: Iterable[str]) -> list[str]:
    values: list[str] = []
    for key in keys:
        if hasattr(form, "getlist"):
            values.extend(
                value.strip()
                for value in form.getlist(key)
                if isinstance(value, str) and value.strip()
            )
        else:
            value = form.get(key)
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
    return values


MAX_CUSTOM_FOLDERS = 100
MAX_FOLDER_OUTPUT_BYTES = 250 * 1024 * 1024


def _get_output_names(form: Any) -> tuple[str | None, list[str]]:
    """Return a legacy single name or deduplicated safe folder names."""
    raw = _all_text(form, ("custom_names",))
    if not raw:
        raw = _all_text(form, ("custom_name", "output_name"))
    entries = [name.strip() for value in raw for name in value.replace("\r", "\n").split("\n") if name.strip()]
    if len(entries) > MAX_CUSTOM_FOLDERS:
        raise ValueError(f"Maximum {MAX_CUSTOM_FOLDERS} custom folder names allowed.")
    names = []
    for name in entries:
        if len(name) > 100:
            raise ValueError("Each custom folder name must be at most 100 characters.")
        # Avoid absolute paths, traversal, and ZIP directory separators.
        clean = sanitize_stem(name)
        if clean not in names:
            names.append(clean)
    if len(names) > 1:
        return None, names
    return (names[0] if names else None), []


def _expand_output_folders(
    processed_files: list[tuple[str, bytes]], folders: list[str],
) -> list[tuple[str, bytes]]:
    if not folders:
        return processed_files
    total_bytes = sum(len(data) for _, data in processed_files) * len(folders)
    if total_bytes > MAX_FOLDER_OUTPUT_BYTES:
        raise ValueError("Generated output exceeds the 250 MB multi-folder limit. Use fewer folders or smaller files.")
    return [(f"{folder}/{Path(name).name}", data) for folder in folders for name, data in processed_files]


def _download_response(
    processed_files: list[tuple[str, bytes]],
    errors: list[str],
    *,
    empty_message: str,
    zip_filename: str,
    error_filename: str,
    custom_name: str | None = None,
    append_custom_name: bool = False,
    folder_names: list[str] | None = None,
) -> Response:
    if not processed_files:
        return _error_zip_response(
            errors or [empty_message],
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            filename=error_filename,
        )

    if folder_names:
        try:
            processed_files = _expand_output_folders(processed_files, folder_names)
        except ValueError as exc:
            return _error_zip_response([str(exc)], status.HTTP_422_UNPROCESSABLE_CONTENT, filename=error_filename)

    if custom_name:
        stem = sanitize_stem(custom_name)
        if len(processed_files) == 1 and not errors:
            output_name, output_bytes = processed_files[0]
            if not append_custom_name:
                ext = Path(output_name).suffix
                output_name = f"{stem}{ext}"
            headers = {"Content-Disposition": f'attachment; filename="{output_name}"'}
            return Response(
                content=output_bytes,
                media_type=media_type_for_filename(output_name),
                headers=headers,
            )
        else:
            zip_filename = f"{stem}.zip"
    else:
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
    try:
        custom_name, folder_names = _get_output_names(form)
    except ValueError as exc:
        return _error_zip_response([str(exc)], status.HTTP_422_UNPROCESSABLE_CONTENT)

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

    banner_files: list[tuple[str, bytes]] = []
    for upload in banner_uploads:
        banner_bytes = await upload.read()
        source_name = upload.filename or "unknown"
        if not banner_bytes:
            errors.append(f"{source_name}: uploaded file is empty")
            continue
            
        if source_name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(banner_bytes)) as zf:
                    for zinfo in zf.infolist():
                        if zinfo.is_dir() or zinfo.filename.startswith("__MACOSX/") or zinfo.filename.split("/")[-1].startswith("."):
                            continue
                        ext = zinfo.filename.lower().rpartition(".")[-1]
                        if ext not in {"png", "jpg", "jpeg", "webp", "bmp", "tiff", "gif", "mp4", "webm", "mov", "avi", "mkv", "m4v"}:
                            continue
                        
                        file_data = zf.read(zinfo.filename)
                        if file_data:
                            banner_files.append((os.path.basename(zinfo.filename), file_data))
            except zipfile.BadZipFile:
                errors.append(f"{source_name}: invalid zip archive")
        else:
            banner_files.append((source_name, banner_bytes))

    if not banner_files:
        return _error_zip_response(
            ["No valid banner images or videos found in the upload."] + errors,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            filename="dbse_processed_banners_errors.zip",
        )

    for chunk_index, upload_chunk in enumerate(_chunked(banner_files, chunk_size)):
        tasks = []
        for offset, (source_name, banner_bytes) in enumerate(upload_chunk):
            if not banner_bytes:
                continue


            tasks.append(
                BannerTask(
                    filename=source_name,
                    banner_bytes=banner_bytes,
                    logo_image=logo_image,
                    placement_mode=placement_mode,
                    custom_name=custom_name,
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
        custom_name=custom_name,
        append_custom_name=True,
        folder_names=folder_names,
    )


@app.post("/optimize")
async def optimize(request: Request) -> Response:
    form = await request.form()
    try:
        custom_name, folder_names = _get_output_names(form)
    except ValueError as exc:
        return _error_zip_response([str(exc)], status.HTTP_422_UNPROCESSABLE_CONTENT)

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
        custom_name=custom_name,
        folder_names=folder_names,
    )


@app.post("/resize")
async def resize(request: Request) -> Response:
    form = await request.form()
    try:
        custom_name, folder_names = _get_output_names(form)
    except ValueError as exc:
        return _error_zip_response([str(exc)], status.HTTP_422_UNPROCESSABLE_CONTENT)

    mode = normalize_resize_mode(_first_text(form, ("mode", "resize_mode")))
    if mode not in ALLOWED_RESIZE_MODES:
        return _error_zip_response(
            [
                "Invalid mode.",
                "Expected one of: fit_expand, blur_expand, solid_background, dominant_color_background.",
            ],
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            filename="dochase_adapted_creatives_errors.zip",
        )

    try:
        targets = parse_target_dimensions(
            _all_text(form, ("targets", "dimensions", "target")),
            _first_text(form, ("target_width", "width")),
            _first_text(form, ("target_height", "height")),
        )
    except ValueError as exc:
        return _error_zip_response(
            [str(exc)],
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            filename="dochase_adapted_creatives_errors.zip",
        )

    try:
        blur_radius = float(_first_text(form, ("blur_radius", "blur")) or "24")
    except ValueError:
        return _error_zip_response(
            ["blur_radius must be a number."],
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            filename="dochase_adapted_creatives_errors.zip",
        )

    uploads = _collect_uploads(form, ("images", "image", "files", "creative", "creatives"))
    if not uploads:
        return _error_zip_response(
            ["No creative images were uploaded."],
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            filename="dochase_adapted_creatives_errors.zip",
        )

    background_color = _first_text(form, ("background_color", "background", "color"))
    processed_files = []
    errors = []
    chunk_size = max(1, min(16, (os.cpu_count() or 1) * 2))

    for chunk_index, upload_chunk in enumerate(_chunked(uploads, chunk_size)):
        tasks = []
        for offset, upload in enumerate(upload_chunk):
            image_bytes = await upload.read()
            source_index = (chunk_index * chunk_size) + offset + 1
            source_name = upload.filename or f"creative_{source_index}"
            if not image_bytes:
                errors.append(f"{source_name}: uploaded file is empty")
                continue

            for target in targets:
                tasks.append(
                    ResizeTask(
                        filename=source_name,
                        image_bytes=image_bytes,
                        target=target,
                        mode=mode,
                        background_color=background_color,
                        blur_radius=blur_radius,
                    )
                )

        for result in process_resize_batch(tasks):
            if result.output_name and result.output_bytes:
                processed_files.append((result.output_name, result.output_bytes))
            if result.error:
                errors.append(result.error)

    return _download_response(
        processed_files,
        errors,
        empty_message="No creatives could be adapted.",
        zip_filename="dochase_adapted_creatives.zip",
        error_filename="dochase_adapted_creatives_errors.zip",
        custom_name=custom_name,
        folder_names=folder_names,
    )

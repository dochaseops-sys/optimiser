# Dochase Banner Stamping Engine (DBSE)

DBSE is an internal tool with two banner workflows:

- stamping a brand logo onto banner images
- optimizing banner files to supported ad sizes under a 150 KB cap

It exposes a FastAPI backend for processing uploads and a lightweight browser UI for uploading banners, selecting the workflow, and downloading either a single processed banner or a ZIP for multi-banner batches.

## Architecture

### Stack

- Backend: FastAPI
- Image processing: Pillow + NumPy
- Frontend: plain HTML, CSS, and JavaScript
- Packaging: Python `zipfile`
- Test suite: `pytest`

### Folder structure

```text
dbse/
  backend/
    __init__.py
    banner_optimizer.py
    main.py
    image_processor.py
    utils.py
  scripts/
    build_dist.py
  dist/
    index.html
  frontend/
    index.html
  tests/
    test_banner_optimizer.py
    test_image_processing.py
  requirements.txt
  README.md
```

### Component plan

- `backend/main.py` hosts the FastAPI app, serves the frontend, validates uploads, batches banner work, and returns either a single processed image or a ZIP archive.
- `backend/banner_optimizer.py` contains supported banner-size rules, exact-multiple reduction logic, and size-constrained encoding for the optimizer workflow.
- `backend/image_processor.py` contains the smart placement algorithm, a centralized processing settings block, logo sizing helpers, and compositing logic.
- `backend/utils.py` contains shared helpers for safe filenames, format handling, and in-memory ZIP packaging.
- `frontend/index.html` provides a drag-and-drop upload UI and download flow for both workflows.
- `tests/test_image_processing.py` validates stamping placement, sizing, API behavior, and failure modes.
- `tests/test_banner_optimizer.py` validates optimizer size mapping, exact-multiple reduction, and output-size constraints.

### Processing flow

### Stamp flow

1. The frontend sends multipart form data with multiple banners, one logo, and a placement mode.
2. The backend validates the request and loads the logo once.
3. Each banner is processed independently.
4. The smart algorithm compares only the top-left and top-right candidate regions.
5. Banners are processed in bounded chunks to reduce peak memory during large batches.
6. A single successful banner is returned directly as an image download.
7. Multi-banner batches are packaged into a ZIP and returned as a downloadable response.

### Optimize flow

1. The frontend sends one or many banner files to `/optimize`.
2. The backend checks whether each banner already matches a supported target size or is an exact integer multiple of one.
3. Multiples are reduced to the supported base size. Example: `900x750` becomes `300x250`.
4. Each optimized banner is encoded to stay under `150 KB`.
5. A single successful banner is returned directly as an image download.
6. Multi-banner batches are packaged into a ZIP and returned as a downloadable response.

### Placement rules

- Allowed logo positions: `top-left` or `top-right` only
- Smart mode evaluates:
  - top-left region: 20% of banner width, 15% of banner height
  - top-right region: 20% of banner width, 15% of banner height
- Noise score:
  - convert region to grayscale
  - apply edge detection
  - use mean intensity as the score
  - choose the lower-noise candidate
- Logo width:
  - standard banners: 1.5% of banner width
  - large banners (`max(width, height) >= 1000`): 2.5% of banner width
  - extra-large banners (`max(width, height) >= 1600`): 3.5% of banner width
- Small-banner safeguard: the rendered logo will not shrink below an `8x8` visible box, while still preserving aspect ratio and staying within the banner
- Margin: 0% so the logo sits at the extreme top corner
- Aspect ratio must be preserved
- Uploaded logo whitespace is preserved as provided

## Running locally

### 1. Install dependencies

```bash
cd dbse
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the backend

```bash
cd dbse
uvicorn backend.main:app --reload
```

The API and hosted UI will be available at `http://127.0.0.1:8000`.

### 3. Open the frontend

Preferred: visit `http://127.0.0.1:8000/` in a browser.

Alternative: open `frontend/index.html` directly in a browser. When opened from disk, the page automatically targets `http://127.0.0.1:8000/process`.

## Deployment

### Build the uploadable frontend

```bash
cd dbse
python3 scripts/build_dist.py
```

This creates:

- `dist/index.html`
- `dist/optimiser/index.html`

### Deploy options

1. Full-stack deployment:
Run the FastAPI app on your server and point your domain at it. If `dist/index.html` exists, the backend serves it automatically at `/`.

If you want the UI at `dochaseadx.com/optimiser`, the backend also serves the same frontend at `/optimiser`.

2. Static frontend + separate API:
Upload the `dist/` folder to your static host, then use `dist/optimiser/index.html` for the `dochaseadx.com/optimiser` path. Set the API base in the file by editing:

```html
<meta name="dbse-api-base" content="https://your-api-domain.com">
```

Leave it empty when the frontend and API share the same origin.

## API

### `POST /process`

Multipart form fields:

- `banners`: multiple image files
- `logo`: single image file
- `placement_mode`: `smart`, `top-left`, or `top-right`

Alias accepted for frontend compatibility:

- `placement`: `smart`, `top-left`, or `top-right`

Response:

- `200 OK` with a single processed image when one banner is uploaded successfully
- `200 OK` with a ZIP file when multiple banners are uploaded successfully

### `POST /optimize`

Multipart form fields:

- `banners`: one or more banner images

Optimizer rules:

- supported target dimensions:
  - `300x250`
  - `320x480`
  - `728x90`
  - `320x50`
  - `300x50`
  - `300x600`
  - `970x250`
  - `120x600`
  - `160x600`
- exact integer multiples of those sizes are reduced to the base target size
  - example: `900x750` becomes `300x250`
- optimized output is capped at `150 KB`

Response:

- `200 OK` with a single optimized image when one banner is uploaded successfully
- `200 OK` with a ZIP file when multiple banners are uploaded successfully

## Performance notes

- The backend processes banners independently to support batch workloads.
- Stamping uses a bounded thread pool for image work without oversubscribing small batches.
- Banner uploads are processed in chunks so the service does not hold all input files in memory at once.
- Optimization currently runs in deterministic sequence because Pillow compression proved unstable under threaded batch encoding in local verification.
- Optimizer output is encoded to meet a `150 KB` delivery budget for supported banner sizes.

## Testing

```bash
cd dbse
pytest
```

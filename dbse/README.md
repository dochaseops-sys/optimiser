# Dochase Banner Stamping Engine (DBSE)

DBSE is an internal tool with three banner workflows:

- stamping a brand logo onto banner images
- optimizing banner files to supported ad sizes under a 150 KB cap
- adapting creatives into new target dimensions without stretching or cropping the original image

It exposes a FastAPI backend for processing uploads and a lightweight browser UI for stamping banners, optimizing banners, adapting creatives, and downloading either a single processed image or a ZIP for batch jobs.

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
    background_generators.py
    banner_optimizer.py
    main.py
    image_processor.py
    resizer.py
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
    test_resizer.py
  requirements.txt
  README.md
```

### Component plan

- `backend/main.py` hosts the FastAPI app, serves the frontend, validates uploads, batches banner work, and returns either a single processed image or a ZIP archive.
- `backend/banner_optimizer.py` contains supported banner-size rules, exact-multiple reduction logic, and size-constrained encoding for the optimizer workflow.
- `backend/image_processor.py` contains the smart placement algorithm, a centralized processing settings block, logo sizing helpers, and compositing logic.
- `backend/resizer.py` contains the creative adaptation engine: aspect-ratio-safe scaling, canvas expansion, output naming, and threaded batch processing.
- `backend/background_generators.py` contains blur expansion, solid-fill parsing, and dominant color extraction.
- `backend/utils.py` contains shared helpers for safe filenames, format handling, and in-memory ZIP packaging.
- `frontend/index.html` provides a drag-and-drop UI for stamping, optimization, and creative adaptation with target presets and batch downloads.
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

### Creative adaptation flow

1. The frontend sends one or many creative image files to `/resize`.
2. The request includes one target size via `target_width` and `target_height`, or multiple target strings such as `300x250,728x90`.
3. Each uploaded image is adapted independently for each target dimension.
4. The foreground image is scaled proportionally using `Image.Resampling.LANCZOS`.
5. The foreground is centered on the target canvas. It is never stretched and never cropped.
6. Remaining canvas space is generated from the selected background mode.
7. A single successful image/target is returned directly as an image download. Batch jobs return a ZIP.

### Adaptation modes

- `fit_expand` default: scales the full creative to fit inside the target and expands the canvas with the source image's dominant color.
- `blur_expand`: scales a duplicate of the original to cover the target canvas, applies Gaussian blur, slightly darkens it, then overlays the sharp fitted creative centered on top.
- `solid_background`: fills expanded canvas space with `background_color` such as `#ffffff`.
- `dominant_color_background`: fills expanded canvas space with the dominant visible source color.

### Quality rules

- All resizing uses `Image.Resampling.LANCZOS`.
- The foreground image always keeps its source aspect ratio.
- JPEG output uses quality `95` with Pillow optimization.
- PNG output is preserved for images with transparency.
- Batch resize work uses a bounded thread pool and the API reads uploads in chunks to reduce peak memory pressure.

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

Alternative: open `frontend/index.html` directly in a browser. When opened from disk, the page automatically targets `http://127.0.0.1:8000` for all API workflows.

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

### `POST /resize`

Multipart form fields:

- `images`: one or more creative image files
- `target_width`: target width in pixels
- `target_height`: target height in pixels
- `targets`: optional repeated or comma-separated dimensions like `300x250,728x90`
- `mode`: `fit_expand`, `blur_expand`, `solid_background`, or `dominant_color_background`
- `background_color`: optional hex color for `solid_background`, for example `#ffffff`
- `blur_radius`: optional number for `blur_expand`; default is `24`

Aliases accepted for upload fields:

- `image`, `files`, `creative`, `creatives`

Response:

- `200 OK` with a single processed image when one image and one target are uploaded successfully
- `200 OK` with a ZIP file when multiple images or multiple targets are uploaded successfully
- `422 Unprocessable Entity` with a ZIP containing `errors.txt` when no output could be produced

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
- Creative adaptation uses a bounded thread pool across image/target tasks.
- Banner uploads are processed in chunks so the service does not hold all input files in memory at once.
- Optimization currently runs in deterministic sequence because Pillow compression proved unstable under threaded batch encoding in local verification.
- Optimizer output is encoded to meet a `150 KB` delivery budget for supported banner sizes.

## Testing

```bash
cd dbse
pytest
```

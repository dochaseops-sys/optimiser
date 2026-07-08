# Dochase Logo Stamp

A FastAPI-powered image processing workspace for preparing display creatives. The app provides a browser UI and backend endpoints for stamping brand logos, optimizing banner images, and adapting creatives to target dimensions.

## Project Structure

```text
.
├── dbse/
│   ├── backend/        # FastAPI app and image-processing modules
│   ├── frontend/       # Single-page browser UI
│   ├── scripts/        # Distribution build script
│   ├── dist/           # Built static frontend output
│   ├── tests/          # Pytest coverage for processing behavior
│   ├── requirements.txt
│   └── README.md       # Detailed DBSE module documentation
├── passenger_wsgi.py
├── requirements.txt
└── vercel.json
```

## Features

- Stamp a logo onto one or more banner images.
- Choose smart, top-left, or top-right logo placement.
- Optimize banner files for common ad dimensions and delivery-size limits.
- Resize/adapt creative images without stretching or cropping.
- Return a single processed file directly, or a ZIP archive for batch jobs.
- Serve the frontend from FastAPI at `/` and `/optimiser`.

## Requirements

- Python 3.11 or newer
- FastAPI
- Uvicorn
- Pillow
- NumPy
- pytest, for local verification

Install the exact Python dependencies from the project requirements file.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r dbse/requirements.txt
```

## Run Locally

```bash
source .venv/bin/activate
uvicorn dbse.backend.main:app --reload
```

Then open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/optimiser`
- `http://127.0.0.1:8000/health`

## Build Static Frontend

```bash
python3 dbse/scripts/build_dist.py
```

This refreshes:

- `dbse/dist/index.html`
- `dbse/dist/optimiser/index.html`

## API Overview

### `GET /health`

Returns a simple health payload:

```json
{"status": "ok"}
```

### `POST /process`

Stamps a logo onto uploaded banner images.

Common multipart fields:

- `banners`: one or more banner files
- `logo`: the logo image
- `placement_mode`: `smart`, `top-left`, or `top-right`
- `custom_name`: optional output name

### `POST /optimize`

Optimizes uploaded banner images.

Common multipart fields:

- `banners`: one or more banner image files
- `target_size_kb`: optional target size
- `enforce_banner_dimensions`: optional boolean
- `custom_name`: optional output name

### `POST /resize`

Adapts images to one or more target dimensions.

Common multipart fields:

- `images`: one or more creative image files
- `target_width`: target width in pixels
- `target_height`: target height in pixels
- `mode`: `fit_expand`, `blur_expand`, `solid_background`, or `dominant_color_background`
- `background_color`: optional hex color for solid backgrounds
- `blur_radius`: optional blur radius for blur expansion

## Testing

```bash
source .venv/bin/activate
pytest dbse/tests
```

## Notes

- `dbse/README.md` contains deeper implementation details for the DBSE package.
- The current repository state does not include a `jpgoptimiser/` folder. If that folder is restored, add or move a project-specific README to `jpgoptimiser/README.md`.

from __future__ import annotations

from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_INDEX = PROJECT_ROOT / "frontend" / "index.html"
DIST_DIR = PROJECT_ROOT / "dist"
DIST_INDEX = DIST_DIR / "index.html"
OPTIMISER_DIR = DIST_DIR / "optimiser"
OPTIMISER_INDEX = OPTIMISER_DIR / "index.html"


def main() -> None:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    OPTIMISER_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_INDEX, DIST_INDEX)
    shutil.copy2(SOURCE_INDEX, OPTIMISER_INDEX)
    print(f"Built {DIST_INDEX}")
    print(f"Built {OPTIMISER_INDEX}")


if __name__ == "__main__":
    main()

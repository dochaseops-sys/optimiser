from __future__ import annotations

from collections import Counter
import re

from PIL import Image, ImageEnhance, ImageFilter

try:  # pragma: no cover - compatibility fallback
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover
    LANCZOS = Image.LANCZOS


_HEX_COLOR_RE = re.compile(r"^#?([0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")


def parse_hex_color(value: str | None, *, fallback: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Parse #RRGGBB or #RRGGBBAA into an RGBA tuple."""

    if not value:
        return fallback

    match = _HEX_COLOR_RE.match(value.strip())
    if not match:
        raise ValueError("background_color must be a hex value like #ffffff")

    raw = match.group(1)
    red = int(raw[0:2], 16)
    green = int(raw[2:4], 16)
    blue = int(raw[4:6], 16)
    alpha = int(raw[6:8], 16) if len(raw) == 8 else 255
    return red, green, blue, alpha


def solid_background(size: tuple[int, int], color: tuple[int, int, int, int]) -> Image.Image:
    """Create a solid RGBA background."""

    return Image.new("RGBA", size, color)


def blur_expand_background(
    source: Image.Image,
    target_size: tuple[int, int],
    *,
    blur_radius: float,
    darken_factor: float = 0.82,
) -> Image.Image:
    """Resize a duplicate to cover the canvas, blur it, and gently darken it."""

    source_rgba = source.convert("RGBA")
    source_width, source_height = source_rgba.size
    target_width, target_height = target_size

    scale = max(target_width / source_width, target_height / source_height)
    fill_size = (
        max(1, int(round(source_width * scale))),
        max(1, int(round(source_height * scale))),
    )
    background = source_rgba.resize(fill_size, LANCZOS)

    left = max(0, (background.width - target_width) // 2)
    top = max(0, (background.height - target_height) // 2)
    background = background.crop((left, top, left + target_width, top + target_height))
    background = background.filter(ImageFilter.GaussianBlur(radius=max(0.0, blur_radius)))

    if darken_factor < 1:
        rgb = ImageEnhance.Brightness(background.convert("RGB")).enhance(darken_factor)
        background = Image.merge("RGBA", (*rgb.split(), background.getchannel("A")))

    return background.convert("RGBA")


def extract_dominant_color(source: Image.Image) -> tuple[int, int, int, int]:
    """Extract a dominant visible color from the image using quantization."""

    rgba = source.convert("RGBA")
    rgba.thumbnail((160, 160), LANCZOS)

    visible = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    visible.alpha_composite(rgba)
    rgb = Image.new("RGB", visible.size, (255, 255, 255))
    rgb.paste(visible, mask=visible.getchannel("A"))

    quantized = rgb.quantize(colors=8, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette() or []
    counts = Counter(quantized.getdata())

    if not counts:
        return (255, 255, 255, 255)

    for index, _ in counts.most_common():
        offset = int(index) * 3
        if offset + 2 >= len(palette):
            continue
        red, green, blue = palette[offset:offset + 3]
        return int(red), int(green), int(blue), 255

    return (255, 255, 255, 255)

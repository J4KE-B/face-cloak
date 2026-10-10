"""Upload validation and image encoding helpers (everything stays in memory)."""

from __future__ import annotations

import base64
import io
import warnings

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from server.settings import MAX_PIXELS, MAX_SIDE, MIN_SIDE

Image.MAX_IMAGE_PIXELS = MAX_PIXELS  # PIL itself also refuses decompression bombs


class UploadError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def sniff_format(head: bytes) -> str | None:
    if head[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    return None


def decode_upload(data: bytes, what: str = "photo") -> np.ndarray:
    """Validate magic bytes + size, apply EXIF orientation, downscale to <=1024 px. Returns RGB uint8."""
    fmt = sniff_format(data[:16])
    if fmt is None:
        raise UploadError(415, f"The {what} must be a JPEG, PNG or WebP image.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            im = Image.open(
                io.BytesIO(data), formats=[fmt.upper()]
            )  # header only: size is known before any pixel is decoded
        w, h = im.size
        if w * h > MAX_PIXELS:
            raise UploadError(
                413,
                f"The {what} is too large ({w * h / 1e6:.0f} MP; the limit is {MAX_PIXELS // 1_000_000} MP).",
            )
        if fmt == "jpeg":
            im.draft("RGB", (2048, 2048))  # DCT-domain downscale: decode far fewer pixels (<=1/8 size)
        im.load()
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
    except UploadError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise UploadError(
            413, f"The {what} is too large (the limit is {MAX_PIXELS // 1_000_000} MP)."
        ) from None
    except Exception:  # noqa: BLE001
        raise UploadError(
            415, f"The {what} could not be decoded; try re-saving it as a JPEG or PNG."
        ) from None
    img = np.ascontiguousarray(np.asarray(im, dtype=np.uint8))
    h, w = img.shape[:2]
    if min(h, w) < MIN_SIDE:
        raise UploadError(422, f"The {what} is too small (at least {MIN_SIDE} px on the short side).")
    return resize_long_side(img, MAX_SIDE)


def resize_long_side(img: np.ndarray, side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def _b64(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def png_data_url(img_rgb: np.ndarray, max_side: int | None = None) -> str:
    if max_side:
        img_rgb = resize_long_side(img_rgb, max_side)
    ok, buf = cv2.imencode(
        ".png", np.ascontiguousarray(img_rgb[:, :, ::-1]), [cv2.IMWRITE_PNG_COMPRESSION, 3]
    )
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return _b64(buf.tobytes(), "image/png")


def jpeg_data_url(img_rgb: np.ndarray, max_side: int = 256, quality: int = 80) -> str:
    img_rgb = resize_long_side(img_rgb, max_side)
    ok, buf = cv2.imencode(
        ".jpg", np.ascontiguousarray(img_rgb[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return _b64(buf.tobytes(), "image/jpeg")


def perturbation_image(original: np.ndarray, cloaked: np.ndarray, gain: float = 10.0) -> np.ndarray:
    """|cloaked - original| x gain, clipped to [0,255] (uint8 RGB)."""
    d = np.abs(cloaked.astype(np.int16) - original.astype(np.int16)).astype(np.float32) * gain
    return np.clip(d, 0, 255).astype(np.uint8)


SYNTHETIC_LABEL = "SYNTHETIC \u00b7 face-cloak demo"
SWAP_MAX_SIDE = 512


def stamp_synthetic(img_rgb: np.ndarray, label: str = SYNTHETIC_LABEL) -> np.ndarray:
    """Burn a visible semi-transparent label band along the bottom edge (marks face-swap output as synthetic)."""
    h, w = img_rgb.shape[:2]
    band = max(12, round(h * 0.08))
    base = Image.fromarray(np.ascontiguousarray(img_rgb)).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    top = max(0, h - band)
    d.rectangle([0, top, w, h], fill=(0, 0, 0, 150))
    size = max(6, int(band * 0.6))
    font = ImageFont.load_default(size)
    while size > 6 and d.textlength(label, font=font) > w - 8:
        size -= 1
        font = ImageFont.load_default(size)
    d.text((w / 2, (top + h) / 2), label, font=font, fill=(255, 255, 255, 235), anchor="mm")
    return np.asarray(Image.alpha_composite(base, overlay).convert("RGB"), dtype=np.uint8).copy()


def swap_output(img_rgb: np.ndarray) -> np.ndarray:
    """Face-swap output as returned to the client: long side <= 512 px and visibly labelled synthetic."""
    return stamp_synthetic(resize_long_side(img_rgb, SWAP_MAX_SIDE))

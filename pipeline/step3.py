"""Optical Character Recognition (OCR).
Crops annotation strips first, runs EasyOCR on each strip independently,
then offsets bounding boxes back to full-image coordinates.
Cropping before OCR prevents CRAFT (EasyOCR's detector) from being confused
by large medical image regions (dark scan cones, lung fields, etc.)."""
from __future__ import annotations
import numpy as np
import easyocr
from config import OCR_LANGUAGES, OCR_CONFIDENCE_THRESHOLD
from functools import lru_cache


@lru_cache(maxsize=1)
def _get_reader() -> easyocr.Reader:
    print("Initialising EasyOCR")
    return easyocr.Reader(OCR_LANGUAGES, gpu=False)


def _annotation_strips(h: int, w: int, modality: str) -> list[tuple[int, int, int, int]]:
    """Return (y0, y1, x0, x1) tuples for each annotation region to OCR."""
    if modality in ("US", "XA", "RF"):
        top_h  = int(h * 0.20)
        bot_h  = int(h * 0.15)
        side_w = int(w * 0.22)
        return [
            (0,          top_h,     0,        w),           # top strip
            (h - bot_h,  h,         0,        w),           # bottom strip
            (top_h,      h - bot_h, 0,        side_w),      # left margin
            (top_h,      h - bot_h, w-side_w, w),           # right margin
        ]
    else:
        # CT / MR / DX / MG / CR — PHI lives in top and bottom annotation bands
        top_h = int(h * 0.25)
        bot_h = int(h * 0.15)
        return [
            (0,          top_h, 0, w),
            (h - bot_h,  h,     0, w),
        ]


def _ocr_strip(reader: easyocr.Reader, strip: np.ndarray) -> list[tuple]:
    if strip.shape[0] < 8 or strip.shape[1] < 8:
        return []
    try:
        return reader.readtext(strip, detail=1, paragraph=False)
    except Exception as exc:
        print(f"  [Step 3] Strip OCR failed — {exc}")
        return []


def run_ocr(pixels: np.ndarray, modality: str = "CT") -> list[dict]:
    if pixels is None or pixels.size == 0:
        print("  [Step 3] Skipping OCR — empty pixel array")
        return []

    reader = _get_reader()

    if pixels.ndim == 2:
        rgb = np.stack([pixels] * 3, axis=-1)
    else:
        rgb = pixels.copy()

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    if rgb.max() == rgb.min():
        print("  [Step 3] Skipping OCR — uniform image, no content")
        return []

    h, w = rgb.shape[:2]
    strips = _annotation_strips(h, w, modality)

    # Collect raw results from each strip with coordinate offsets
    raw_results: list[tuple] = []
    for y0, y1, x0, x1 in strips:
        strip = rgb[y0:y1, x0:x1]
        for bbox, text, conf in _ocr_strip(reader, strip):
            # Shift bbox back to full-image coordinates
            shifted = [[pt[0] + x0, pt[1] + y0] for pt in bbox]
            raw_results.append((shifted, text, conf))

    detections: list[dict] = []
    for bbox, text, conf in raw_results:
        if conf < OCR_CONFIDENCE_THRESHOLD:
            continue
        if not text.strip() or len(text.strip()) < 2:
            continue

        xs = [int(pt[0]) for pt in bbox]
        ys = [int(pt[1]) for pt in bbox]
        x0b = max(0, min(xs))
        x1b = min(w, max(xs))
        y0b = max(0, min(ys))
        y1b = min(h, max(ys))

        detections.append({
            "bbox":       bbox,
            "text":       text.strip(),
            "confidence": round(float(conf), 4),
            "x0": x0b, "y0": y0b, "x1": x1b, "y1": y1b,
        })

    print(f"  [Step 3] OCR found {len(detections)} text region(s) "
          f"(modality={modality}, threshold≥{OCR_CONFIDENCE_THRESHOLD})")
    return detections

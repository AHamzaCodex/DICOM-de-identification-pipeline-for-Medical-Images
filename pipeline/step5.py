"""redaction of  sensitive pixel regions (which were classified as is_phi) with pixel redacted value we have set."""
from __future__ import annotations
import numpy as np
from config import REDACT_FILL_VALUE
def redact_pixels(pixels: np.ndarray, detections: list[dict]) -> np.ndarray:
    redacted = pixels.copy()
    count = 0
    for det in detections:
        if not det.get("is_phi", False):
            continue
        x0, y0 = det["x0"], det["y0"]
        x1, y1 = det["x1"], det["y1"]

        # Add a small padding around the box (2 px) for safety
        pad = 2
        h, w = redacted.shape[:2]
        rx0 = max(0, x0 - pad)
        ry0 = max(0, y0 - pad)
        rx1 = min(w, x1 + pad)
        ry1 = min(h, y1 + pad)

        redacted[ry0:ry1, rx0:rx1] = REDACT_FILL_VALUE
        count += 1
    print(f"[Step 5] Redacted {count} pixel region(s) "
          f"(fill value={REDACT_FILL_VALUE})")
    return redacted
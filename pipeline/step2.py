"""Load DICOM metadata and pixel array.
Uses pydicom directly for pixels — MONAI's LoadImage applies an ITK coordinate
transform that rotates/mirrors the image, breaking downstream OCR."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pydicom
from monai.transforms import ScaleIntensity

_scaler = ScaleIntensity(minv=0.0, maxv=255.0)


def load_dicom(dcm_path: str | Path) -> dict:
    path = Path(dcm_path)
    ds   = pydicom.dcmread(str(path), force=True)

    arr = ds.pixel_array.astype(np.float32)

    # Multi-frame DICOM: use the middle frame
    if arr.ndim == 3:
        arr = arr[arr.shape[0] // 2]

    # Normalise to 0-255 uint8 for EasyOCR
    arr_scaled = np.array(_scaler(arr))
    pixels     = np.clip(arr_scaled, 0, 255).astype(np.uint8)

    return {
        "dataset": ds,
        "pixels":  pixels,
        "path":    path,
    }

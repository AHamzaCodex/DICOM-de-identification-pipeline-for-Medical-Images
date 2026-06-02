"""==> Save the de-identified output
Writes the redacted pixel array back into the scrubbed pydicom dataset and saves a new .dcm file under the output directory, and preserves original folder structure relative to the input root.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pydicom
from pydicom.dataset import FileDataset
from pydicom.uid import ExplicitVRLittleEndian
from config import OUTPUT_DIR, RAW_DICOM_DIR

def save_deidentified(
    ds: FileDataset,
    redacted_pixels: np.ndarray,
    source_path: Path,
    input_root: Path = RAW_DICOM_DIR,
    output_root: Path = OUTPUT_DIR,
) -> Path:
    """Embed redacted pixels into the dataset and write a new DICOM file"""
    # ── 1. Set dtype and pixel metadata based on color depth
    is_color = redacted_pixels.ndim == 3 and redacted_pixels.shape[2] == 3
    if is_color:
        pixels_out              = redacted_pixels.astype(np.uint8)
        ds.BitsAllocated        = 8
        ds.BitsStored           = 8
        ds.HighBit              = 7
        ds.PixelRepresentation  = 0
        ds.SamplesPerPixel      = 3
        ds.PlanarConfiguration  = 0     # pixel-interleaved
        ds.PhotometricInterpretation = "RGB"
    else:
        pixels_out = (redacted_pixels if redacted_pixels.dtype == np.uint16
                      else redacted_pixels.astype(np.uint16))
        ds.BitsAllocated        = 16
        ds.BitsStored           = 16
        ds.HighBit              = 15
        ds.PixelRepresentation  = 0
        ds.SamplesPerPixel      = 1
        ds.PhotometricInterpretation = "MONOCHROME2"

    # ── 2. Update pixel-related metadata tags
    ds.PixelData = pixels_out.tobytes()
    ds.Rows      = pixels_out.shape[0]
    ds.Columns   = pixels_out.shape[1]

    # Ensure file meta is present and uses explicit VR little-endian
    if not hasattr(ds, "file_meta") or ds.file_meta is None:
        ds.file_meta = pydicom.Dataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    # ── 3. Compute output path (mirror input structure)
    try:
        rel = source_path.resolve().relative_to(input_root.resolve())
    except ValueError:
        rel = Path(source_path.name)

    out_path = output_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 4. Write the file
    pydicom.dcmwrite(str(out_path), ds)
    print(f"  [Step 7] Saved → {out_path}")
    return out_path

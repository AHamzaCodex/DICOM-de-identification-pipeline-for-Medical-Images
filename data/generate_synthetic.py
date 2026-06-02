"""
data/generate_synthetic.py
--------------------------
Generate synthetic DICOM files with fake PHI burned into pixel overlays
and embedded in metadata tags, for stress-testing the de-identification pipeline.

Modalities produced: CT, MR, US, DX, MG
Each gets a unique realistic-looking image + PHI text overlaid in the annotation
areas that real scanners use (corners, top/bottom strips).

Usage:
    python data/generate_synthetic.py

Then run the pipeline on the output:
    python run.py --input data/synthetic_phi --no-download
"""

from __future__ import annotations
import hashlib
import sys
from pathlib import Path

import os
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BASE_DIR

OUT_DIR = BASE_DIR / "data" / "synthetic_phi"

# --------------------------------------------------------------------------- #
# Fake PHI records                                                             #
# --------------------------------------------------------------------------- #
PATIENTS = [
    {
        "PatientName":      "Smith^John^Michael",
        "PatientID":        "PT-001234",
        "PatientBirthDate": "19800115",
        "PatientSex":       "M",
        "PatientAge":       "044Y",
    },
    {
        "PatientName":      "Johnson^Emily^Rose",
        "PatientID":        "MRN-005678",
        "PatientBirthDate": "19920307",
        "PatientSex":       "F",
        "PatientAge":       "032Y",
    },
    {
        "PatientName":      "Williams^Robert^James",
        "PatientID":        "PT-789012",
        "PatientBirthDate": "19651122",
        "PatientSex":       "M",
        "PatientAge":       "058Y",
    },
]

INSTITUTION  = "City General Hospital"
PHYSICIAN    = "Brown^David^R"
STUDY_DATES  = ["20240115", "20240210", "20240305"]
STUDY_TIMES  = ["083012",   "141530",   "092245"]

MODALITIES   = ["CT", "MR", "US", "DX", "MG"]


# --------------------------------------------------------------------------- #
# Image generators — one per modality                                          #
# --------------------------------------------------------------------------- #

def _make_ct(h: int = 512, w: int = 512) -> np.ndarray:
    """Axial CT slice: air background, soft-tissue body, spine, lung fields."""
    rng = np.random.default_rng(42)
    img = np.full((h, w), 20.0)

    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]

    body  = ((X - cx) / (w * 0.38)) ** 2 + ((Y - cy) / (h * 0.45)) ** 2 <= 1
    img[body] = 100 + rng.normal(0, 15, img[body].shape)

    spine = ((X - cx) / 30) ** 2 + ((Y - (cy + int(h * 0.18))) / 20) ** 2 <= 1
    img[spine] = 700 + rng.normal(0, 30, img[spine].shape)

    for dx in (-int(w * 0.2), int(w * 0.2)):
        lung = ((X - (cx + dx)) / int(w * 0.15)) ** 2 + ((Y - cy) / int(h * 0.22)) ** 2 <= 1
        img[lung] = -700 + rng.normal(0, 40, img[lung].shape)

    img += rng.normal(0, 8, img.shape)
    img  = np.clip(img, -1024, 3000)
    img  = (img + 1024) / 4024 * 255          # rescale to 0-255
    return np.clip(img, 0, 255).astype(np.uint8)


def _make_mri(h: int = 256, w: int = 256) -> np.ndarray:
    """T1-weighted brain MRI: white matter bright, CSF dark."""
    rng = np.random.default_rng(7)
    img = np.zeros((h, w), dtype=np.float32)

    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]

    brain = ((X - cx) / (w * 0.35)) ** 2 + ((Y - cy) / (h * 0.40)) ** 2 <= 1
    img[brain] = 80 + rng.normal(0, 10, img[brain].shape)

    gm = brain & ~(((X - cx) / (w * 0.30)) ** 2 + ((Y - cy) / (h * 0.34)) ** 2 <= 1)
    img[gm] = 140 + rng.normal(0, 12, img[gm].shape)

    wm = ((X - cx) / (w * 0.28)) ** 2 + ((Y - cy) / (h * 0.32)) ** 2 <= 1
    img[wm] = 180 + rng.normal(0, 8, img[wm].shape)

    return np.clip(img, 0, 255).astype(np.uint8)


def _make_us(h: int = 480, w: int = 640) -> np.ndarray:
    """Ultrasound: sector-shaped speckle cone with bright reflector lines."""
    rng = np.random.default_rng(13)
    img = np.zeros((h, w), dtype=np.float32)

    Y, X  = np.ogrid[:h, :w]
    cx    = w // 2
    dx    = X - cx
    dy    = np.maximum(Y, 1)                   # avoid zero at top row
    angles = np.arctan2(np.abs(dx), dy)
    dist   = np.sqrt(dx ** 2 + dy ** 2)

    cone  = (angles < np.radians(50)) & (dist > 50) & (dist < h * 0.88)
    img[cone] = rng.rayleigh(scale=55, size=(h, w))[cone]

    for depth in (150, 260, 370):
        row_mask = cone[depth]
        img[depth, row_mask] = rng.uniform(180, 220, row_mask.sum())

    return np.clip(img, 0, 255).astype(np.uint8)


def _make_dx(h: int = 1024, w: int = 1024) -> np.ndarray:
    """Chest X-ray (DX/CR): bright background, dark lung fields, rib bands."""
    rng = np.random.default_rng(99)
    img = np.full((h, w), 180.0) + rng.normal(0, 5, (h, w))

    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]

    for dx in (-int(w * 0.2), int(w * 0.2)):
        lung = ((X - (cx + dx)) / int(w * 0.18)) ** 2 + ((Y - cy) / int(h * 0.28)) ** 2 <= 1
        img[lung] = 60 + rng.normal(0, 10, img[lung].shape)

    rib_y_start = int(cy - h * 0.25)
    rib_y_end   = int(cy + h * 0.25)
    x_start     = int(cx - w * 0.38)
    x_end       = int(cx + w * 0.38)
    for rib_y in range(rib_y_start, rib_y_end, 30):
        img[rib_y:rib_y + 4, x_start:x_end] = 210 + rng.normal(0, 5, (4, x_end - x_start))

    return np.clip(img, 0, 255).astype(np.uint8)


def _make_mg(h: int = 1024, w: int = 512) -> np.ndarray:
    """Mammogram: triangular bright tissue on black background."""
    rng = np.random.default_rng(55)
    img = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        x_width = int(w * 0.8 * (1 - y / h * 0.35))
        if x_width > 0:
            img[y, :x_width] = 120 + rng.normal(0, 20, x_width)

    Y, X = np.ogrid[:h, :w]
    for (cy, cx, r) in ((300, 180, 70), (550, 140, 55), (450, 280, 35)):
        blob = ((X - cx) / r) ** 2 + ((Y - cy) / r) ** 2 <= 1
        img[blob] += 50 + rng.normal(0, 10, img[blob].shape)

    return np.clip(img, 0, 255).astype(np.uint8)


GENERATORS = {"CT": _make_ct, "MR": _make_mri, "US": _make_us,
              "DX": _make_dx, "MG": _make_mg}


# --------------------------------------------------------------------------- #
# PHI text overlay                                                             #
# --------------------------------------------------------------------------- #

def _load_font(size: int) -> ImageFont.ImageFont:
    windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    candidates = [
        windir / "Fonts" / "arial.ttf",
        windir / "Fonts" / "calibri.ttf",
        windir / "Fonts" / "verdana.ttf",
        windir / "Fonts" / "segoeui.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except (IOError, OSError):
                pass
    print("  [WARNING] No TrueType font found — OCR may not detect text")
    return ImageFont.load_default()


def _burn_phi(pixels: np.ndarray, patient: dict, study_date: str) -> np.ndarray:
    """Burn fake PHI using PIL TrueType font on a solid band — mirrors real scanner overlays
    and gives CRAFT a clean text-on-uniform-background region to detect."""
    h, w = pixels.shape[:2]

    font_size = min(48, max(24, min(h, w) // 14))
    lh        = int(font_size * 1.6)

    font = _load_font(font_size)

    name     = patient["PatientName"].replace("^", " ")
    pid      = patient["PatientID"]
    dob      = patient["PatientBirthDate"]
    date_fmt = f"{study_date[:4]}/{study_date[4:6]}/{study_date[6:]}"
    acc      = f"ACC {study_date[:4]}-{pid[-4:]}"
    dr       = PHYSICIAN.replace("^", " ")

    all_lines = [
        INSTITUTION,
        f"Pt: {name}",
        f"ID: {pid}",
        f"DOB: {dob}",
        f"Date: {date_fmt}",
        f"Dr: {dr}",
    ]
    max_lines = max(2, (h // 2) // lh)
    lines     = all_lines[:max_lines]

    pil_img = Image.fromarray(pixels.astype(np.uint8), mode="L")
    draw    = ImageDraw.Draw(pil_img)

    # Solid black band at top — CRAFT reliably detects text on uniform backgrounds
    band_h = len(lines) * lh + 10
    draw.rectangle([(0, 0), (w, band_h)], fill=0)
    for i, line in enumerate(lines):
        draw.text((6, 5 + i * lh), line, font=font, fill=240)

    # Solid black band at bottom for accession number
    draw.rectangle([(0, h - lh - 8), (w, h)], fill=0)
    draw.text((6, h - lh - 4), acc, font=font, fill=240)

    return np.array(pil_img)


# --------------------------------------------------------------------------- #
# DICOM builder                                                                #
# --------------------------------------------------------------------------- #

def _deterministic_uid(seed: str) -> str:
    digest  = hashlib.sha256(seed.encode()).hexdigest()
    decimal = str(int(digest[:16], 16))
    return ("2.25." + decimal)[:64]


def _build_dicom(
    pixels:     np.ndarray,
    modality:   str,
    patient:    dict,
    study_date: str,
    study_time: str,
    study_uid:  str,
    series_uid: str,
    sop_uid:    str,
) -> FileDataset:
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID    = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID          = ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.is_implicit_VR   = False   # required for correct explicit-VR encoding on write
    ds.is_little_endian = True

    # --- Patient ---
    ds.PatientName      = patient["PatientName"]
    ds.PatientID        = patient["PatientID"]
    ds.PatientBirthDate = patient["PatientBirthDate"]
    ds.PatientSex       = patient["PatientSex"]
    ds.PatientAge       = patient["PatientAge"]
    ds.PatientAddress   = "456 Oak Street, Springfield, IL"

    # --- Study ---
    ds.StudyInstanceUID       = study_uid
    ds.StudyDate              = study_date
    ds.StudyTime              = study_time
    ds.AccessionNumber        = f"ACC-{study_date[2:]}-{patient['PatientID'][-3:]}"  # ≤16 chars
    ds.StudyID                = "1"
    ds.ReferringPhysicianName = PHYSICIAN

    # --- Series ---
    ds.SeriesInstanceUID = series_uid
    ds.SeriesDate        = study_date
    ds.SeriesTime        = study_time
    ds.Modality          = modality
    ds.SeriesNumber      = "1"
    ds.SeriesDescription = f"Synthetic {modality} series"

    # --- Instance ---
    ds.SOPClassUID    = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = sop_uid
    ds.InstanceNumber = "1"
    ds.ContentDate    = study_date
    ds.ContentTime    = study_time

    # --- Equipment ---
    ds.InstitutionName          = INSTITUTION
    ds.InstitutionAddress       = "123 Medical Drive, Springfield, IL 62701"
    ds.StationName              = "SCANNER-01"
    ds.OperatorsName            = "TechOp^Jane"
    ds.PerformingPhysicianName  = PHYSICIAN
    ds.RequestingPhysician      = PHYSICIAN

    # --- Pixel data (always grayscale uint16 for simplicity) ---
    pixels_out = pixels.astype(np.uint16)
    ds.SamplesPerPixel           = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows                      = pixels_out.shape[0]
    ds.Columns                   = pixels_out.shape[1]
    ds.BitsAllocated             = 16
    ds.BitsStored                = 16
    ds.HighBit                   = 15
    ds.PixelRepresentation       = 0
    ds.WindowCenter              = "128"
    ds.WindowWidth               = "256"
    ds.PixelData                 = pixels_out.tobytes()

    return ds


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def generate(out_dir: Path = OUT_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    total = len(PATIENTS) * len(MODALITIES)
    idx   = 0

    for p_idx, patient in enumerate(PATIENTS):
        study_date = STUDY_DATES[p_idx]
        study_time = STUDY_TIMES[p_idx]
        study_uid  = _deterministic_uid(f"study-{p_idx}")

        for modality in MODALITIES:
            idx += 1
            series_uid = _deterministic_uid(f"series-{p_idx}-{modality}")
            sop_uid    = _deterministic_uid(f"sop-{p_idx}-{modality}")

            base    = GENERATORS[modality]()
            pixels  = _burn_phi(base, patient, study_date)

            ds = _build_dicom(
                pixels=pixels,
                modality=modality,
                patient=patient,
                study_date=study_date,
                study_time=study_time,
                study_uid=study_uid,
                series_uid=series_uid,
                sop_uid=sop_uid,
            )

            out_path = out_dir / patient["PatientID"] / modality / f"{sop_uid}.dcm"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pydicom.dcmwrite(str(out_path), ds)
            saved.append(out_path)
            name = patient["PatientName"].replace("^", " ")
            print(f"  [{idx:02d}/{total}] {modality:3s}  {name}  →  {out_path.name}")

    print(f"\nDone. {len(saved)} synthetic DICOM file(s) saved to:\n  {out_dir}")
    return saved


if __name__ == "__main__":
    generate()

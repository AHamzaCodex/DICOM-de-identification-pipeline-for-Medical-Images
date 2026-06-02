"""
Step 1: Discover DICOM studies
Walks through the input directory recursively and collects every .dcm file.Returns a flat list of Path objects for downstream steps"""

from pathlib import Path

def discover_dicoms(root_dir: str | Path) -> list[Path]:
    root = Path(root_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    dcm_files = sorted(root.rglob("*.dcm"))

    if not dcm_files:
        print(f"No .dcm files found under {root}")
    else:
        print(f"[Step 1] Discovered {len(dcm_files)} DICOM file(s) under {root}")

    return dcm_files


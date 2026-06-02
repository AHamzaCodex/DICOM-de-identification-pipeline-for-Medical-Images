"""Purpose: Download a set of real DICOM files from The Cancer Imaging Archive (TCIA).
Data source:
The Cancer Imaging Archive (TCIA) — https://www.cancerimagingarchive.net/
Collection used: TCGA-LUAD (Lung Adenocarcinoma CT scans"""

import sys
import requests
from pathlib import Path
from tqdm import tqdm

# Allow running as a standalone script from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1])) #parent[1] allows looking into 1 level above of this current file

from config import (
    TCIA_BASE_URL,
    TCIA_COLLECTION,
    TCIA_MAX_STUDIES,
    TCIA_MAX_SERIES,
    RAW_DICOM_DIR,
)
def _get(endpoint: str, params: dict = None) -> list:
    """GET from the TCIA REST API and return parsed JSON (always a list)."""
    url = f"{TCIA_BASE_URL}/{endpoint}"
    resp = requests.get(url, params=params or {}, timeout=60)
    resp.raise_for_status()
    return resp.json()

def get_studies(collection: str, max_studies: int) -> list[str]:
    """Return a list of StudyInstanceUIDs for the given collection"""
    print(f"[1/4] Fetching study list for collection '{collection}' ...")
    studies = _get("getPatientStudy", {"Collection": collection})
    uids = [s["StudyInstanceUID"] for s in studies[:max_studies]]
    print(f"{len(uids)} study/studies selected")
    return uids


def get_series(study_uid: str, max_series: int) -> list[dict]:
    """Return series metadata for a given study."""
    series = _get("getSeries", {"StudyInstanceUID": study_uid})
    return series[:max_series]


def download_series(series_uid: str, out_dir: Path) -> int:
    """
    Download all DICOM instances in a series into out_dir.
    Returns the number of files saved.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get instance SOPInstanceUIDs for the series
    instances = _get("getSOPInstanceUIDs", {"SeriesInstanceUID": series_uid})
    if not instances:
        print(f"No instances found for series {series_uid[:20]}")
        return 0

    saved = 0
    for inst in tqdm(instances, desc=f"Downloading {series_uid[:20]}...", unit="file"):
        sop_uid = inst["SOPInstanceUID"]
        resp = requests.get(
            f"{TCIA_BASE_URL}/getSingleImage",
            params={"SeriesInstanceUID": series_uid, "SOPInstanceUID": sop_uid},
            timeout=60,
        )
        resp.raise_for_status()

        out_path = out_dir / f"{sop_uid}.dcm"
        out_path.write_bytes(resp.content)
        saved += 1

    return saved


def download(
    collection: str = TCIA_COLLECTION,
    max_studies: int = TCIA_MAX_STUDIES,
    max_series: int = TCIA_MAX_SERIES,
    out_dir: Path = RAW_DICOM_DIR,
) -> list[Path]:
    """Download DICOM files from TCIA and return list of paths to saved files"""
    print("=" * 60)
    print("TCIA DICOM Downloader")
    print(f"Collection : {collection}")
    print(f"Max studies: {max_studies}  |  Max series/study: {max_series}")
    print(f"Output dir : {out_dir}")
    print("=" * 60)

    study_uids = get_studies(collection, max_studies)
    all_files: list[Path] = []

    for i, study_uid in enumerate(study_uids, 1):
        print(f"\n[Study {i}/{len(study_uids)}] UID: {study_uid[:40]}...")
        series_list = get_series(study_uid, max_series)

        if not series_list:
            print("No series found, skipping.")
            continue

        for j, series in enumerate(series_list, 1):
            series_uid = series["SeriesInstanceUID"]
            modality   = series.get("Modality", "UNKNOWN")
            series_dir = out_dir / study_uid[:20] / series_uid[:20]

            print(f" [Series {j}/{len(series_list)}] Modality={modality}")
            n = download_series(series_uid, series_dir)
            print(f" {n} file(s) saved → {series_dir}")
            all_files.extend(series_dir.glob("*.dcm"))

    print(f"\nDownload complete. Total files: {len(all_files)}")
    return all_files


if __name__ == "__main__":
    download()

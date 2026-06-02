"""
run_pipeline.py
---------------
Entry point for the full DICOM de-identification pipeline.

Usage
-----
    # Download free TCIA data + run the pipeline
    python run_pipeline.py

    # Run on a folder of DICOMs you already have
    python run_pipeline.py --input /path/to/your/dicoms --no-download

    # Dry-run (skip saving output files)
    python run_pipeline.py --dry-run
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from monai.data import Dataset, DataLoader

from config import RAW_DICOM_DIR, OUTPUT_DIR, LOG_DIR
from data.download_data import download
from pipeline.step1 import discover_dicoms
from transform.de_identify import DeidentifyTransform


def setup_logging() -> logging.Logger:
    log_file = LOG_DIR / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("deidentifier")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AI-powered DICOM de-identification pipeline for Medical Images"
    )
    p.add_argument(
        "--input", type=Path, default=RAW_DICOM_DIR,
        help="Directory containing input DICOM files",
    )
    p.add_argument(
        "--output", type=Path, default=OUTPUT_DIR,
        help="Directory for de-identified output files",
    )
    p.add_argument(
        "--no-download", action="store_true",
        help="Skip the TCIA download step",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Run OCR and classification without writing output files",
    )
    p.add_argument(
        "--num-workers", type=int, default=0,
        help="Number of DataLoader worker processes (default: 0)",
    )
    return p.parse_args()


def save_report(results: list[dict], logger: logging.Logger) -> None:
    total     = len(results)
    phi_total = sum(
        sum(1 for d in r.get("path_detections", []) if d.get("is_phi"))
        for r in results
    )
    saved = [r.get("path_output_path") for r in results if r.get("path_output_path")]

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Files processed   : {total}")
    logger.info(f"  Total PHI regions : {phi_total}")
    logger.info(f"  Files saved       : {len(saved)}")
    if saved:
        logger.info(f"  Output directory  : {OUTPUT_DIR}")

    report = []
    for r in results:
        dets = r.get("path_detections", [])
        report.append({
            "source":  str(r.get("path")),
            "output":  str(r.get("path_output_path")),
            "detections": [
                {
                    "text":       d["text"],
                    "is_phi":     d.get("is_phi"),
                    "phi_reason": d.get("phi_reason"),
                    "confidence": d.get("confidence"),
                }
                for d in dets
            ],
        })

    report_path = LOG_DIR / f"report_{datetime.now():%Y%m%d_%H%M%S}.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info(f"  Detailed report   : {report_path}")
    logger.info("=" * 60)


def main() -> None:
    args   = parse_args()
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("DICOM De-Identification Pipeline")
    logger.info("=" * 60)

    # Step 1a — download data if needed
    if not args.no_download:
        logger.info("Downloading DICOM data from TCIA ")
        download(out_dir=args.input)
    else:
        logger.info("Skipping download.")

    # Step 1b — discover files
    dcm_files = discover_dicoms(args.input)
    if not dcm_files:
        logger.error("No DICOM files found. Exiting.")
        sys.exit(1)

    # Build MONAI Dataset and DataLoader
    data_list = [{"path": str(p)} for p in dcm_files]

    transform = DeidentifyTransform(
        keys=("path",),
        input_root=args.input,
        output_root=args.output,
        save=not args.dry_run,
    )

    monai_ds = Dataset(data=data_list, transform=transform)
    loader   = DataLoader(
        monai_ds,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=lambda x: x[0],
    )

    logger.info(f"Processing {len(dcm_files)} file(s) ...\n")

    results: list[dict] = []
    loader_iter = iter(loader)
    for i in range(1, len(dcm_files) + 1):
        try:
            result = next(loader_iter)
            logger.info(f"[{i}/{len(dcm_files)}] Done: {result.get('path')}")
            results.append(result)
        except StopIteration:
            break
        except Exception as exc:
            logger.error(f"[{i}/{len(dcm_files)}] Error — {exc} — skipping.")

    save_report(results, logger)


if __name__ == "__main__":
    main()
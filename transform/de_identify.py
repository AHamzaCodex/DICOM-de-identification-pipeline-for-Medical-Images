"""Purpose: MONAI MapTransform that wraps all 7 pipeline steps into a single reusable transform compatible with MONAI Dataset / DataLoader"""

from __future__ import annotations
from pathlib import Path
from monai.transforms import MapTransform
# Import each pipeline step
from pipeline.step2 import load_dicom
from pipeline.step3 import run_ocr
from pipeline.step4 import classify_detections
from pipeline.step5 import redact_pixels
from pipeline.step6 import scrub_metadata
from pseudonymisation.step7 import save_deidentified
from config import RAW_DICOM_DIR, OUTPUT_DIR

class DeidentifyTransform(MapTransform):
    """MONAI MapTransform implementing the full 7-step de-identification pipeline"""
    def __init__(
        self,
        keys: tuple[str, ...] = ("path",),
        input_root: Path = RAW_DICOM_DIR,
        output_root: Path = OUTPUT_DIR,
        save: bool = True,
    ):
        super().__init__(keys)
        self.input_root  = Path(input_root)
        self.output_root = Path(output_root)
        self.save        = save

    def __call__(self, data: dict) -> dict:
        d = dict(data)   #shallow copy —-> don't mutate the input

        for key in self.keys:
            path = Path(d[key])
            print(f"\n{'='*60}")
            print(f"Processing: {path.name}")
            print(f"{'='*60}")

            #Step 2: Load
            loaded   = load_dicom(path)
            dataset  = loaded["dataset"]
            pixels   = loaded["pixels"]
            modality = str(dataset.get("Modality", "CT"))

            #Step 3: OCR
            detections = run_ocr(pixels, modality)

            #Step 4:Classify
            detections = classify_detections(detections)

            #Step 5: Redact pixels
            pixels_redacted = redact_pixels(pixels, detections)

            #Step 6: Scrub metadata
            dataset = scrub_metadata(dataset)

            #Step 7: Save
            output_path = None
            if self.save:
                output_path = save_deidentified(
                    ds=dataset,
                    redacted_pixels=pixels_redacted,
                    source_path=path,
                    input_root=self.input_root,
                    output_root=self.output_root,
                )
            # Enrich the data dict
            d[f"{key}_dataset"]       = dataset
            d[f"{key}_pixels_redacted"] = pixels_redacted
            d[f"{key}_detections"]    = detections
            d[f"{key}_output_path"]   = output_path
        return d

"""Remove PHI from DICOM metadata.
Implements the DICOM PS3.15 Annex E Basic Application Level (can be read at this link: https://www.aliza-dicom-viewer.com/manual/dicom-specification/application-level-confidentiality-profile-attributes)
Brief description of confidentiality profile:
  --> Replaces known PHI tags with anonymous value
  --> Deletes additional sensitive optional tag
  --> Removes all private (odd-group) tags which may contain vendor PHI.
Therefore, this code replaces Study/Series/SOP Instance UIDs with deterministic pseudonyms consistently which may enable longitudinal research while preserving privacy"""

from __future__ import annotations
import hashlib
import pydicom
from pydicom.dataset import FileDataset
from pydicom.sequence import Sequence

from config import TAGS_TO_ANONYMIZE, TAGS_TO_DELETE
# DICOM UIDs we generate will be rooted under this prefix
_UID_PREFIX = "2.25."

def _pseudonymise_uid(original_uid: str) -> str:
    digest = hashlib.sha256(original_uid.encode()).hexdigest()
    # Convert hex to a decimal integer string and prepend the prefix
    decimal = str(int(digest[:16], 16))          # 16 hex chars → ~19 digits
    new_uid = _UID_PREFIX + decimal
    return new_uid[:64]                           # DICOM UID max length = 64


# ── Recursive metadata scrubber ───────────────────────────────────────────────

def _scrub_sequence(seq: Sequence) -> None:
    """Recursively scrub nested DICOM sequences."""
    for item in seq:
        _scrub_dataset(item)


def _scrub_dataset(ds: FileDataset) -> None:
    """
    In-place scrub of a single DICOM dataset (or nested item).
    """
    # 1. Replace known PHI tags
    for keyword, replacement in TAGS_TO_ANONYMIZE.items():
        if keyword in ds:
            setattr(ds, keyword, replacement)

    # 2. Delete optional sensitive tags
    for keyword in TAGS_TO_DELETE:
        if keyword in ds:
            del ds[keyword]

    # 3. Remove all private (odd-group) tags
    ds.remove_private_tags()

    # 4. Pseudonymise UIDs
    for uid_tag in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID",
                    "ReferencedSOPInstanceUID", "FrameOfReferenceUID"):
        if uid_tag in ds:
            original = str(getattr(ds, uid_tag))
            setattr(ds, uid_tag, _pseudonymise_uid(original))

    # 5. Recurse into nested sequences
    for elem in ds:
        if elem.VR == "SQ" and elem.value:
            _scrub_sequence(elem.value)

def scrub_metadata(ds: FileDataset) -> FileDataset:
    """scrubing all PHI from a pydicom FileDataset"""
    _scrub_dataset(ds)
    # Mark the file as de-identified (DICOM standard tag)
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = (
        "DICOM PS3.15 Annex E — Basic Application Level Confidentiality Profile "
        "+ pixel redaction via EasyOCR + StanfordAIMI NER"
    )
    print("  [Step 6] Metadata scrubbed "
          f"({len(TAGS_TO_ANONYMIZE)} tags anonymised, "
          f"{len(TAGS_TO_DELETE)} tags deleted, private tags removed, UIDs pseudonymised)")
    return ds
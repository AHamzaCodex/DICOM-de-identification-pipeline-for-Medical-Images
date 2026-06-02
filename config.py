"""configuration for the DICOM de-identification pipeline"""

from pathlib import Path

BASE_DIR        = Path(__file__).parent
RAW_DICOM_DIR   = BASE_DIR / "data" / "raw"        # downloaded / input DICOMs
OUTPUT_DIR      = BASE_DIR / "data" / "deidentified"  # privacy-safe output
LOG_DIR         = BASE_DIR / "logs"

for _d in (RAW_DICOM_DIR, OUTPUT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TCIA_BASE_URL   = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
TCIA_COLLECTION = "Prostate-MRI-US-Biopsy" #"TCGA-LUAD" #I am using LUAD (lung adenocarcinoma) for this project; list of collections: https://www.cancerimagingarchive.net/collections/
TCIA_MAX_STUDIES = 3 #study = number of clinical visit by a patient
TCIA_MAX_SERIES  = 2 #series = number of mode of acquisition of image


OCR_LANGUAGES   = ["en"]
OCR_CONFIDENCE_THRESHOLD = 0.4

# Model card: https://huggingface.co/StanfordAIMI/stanford-deidentifier-base
NER_MODEL_NAME  = "StanfordAIMI/stanford-deidentifier-base"
#PHI: Protected Health Information (as defined by HIPAA)
PHI_ENTITY_LABELS = {
    # Generic labels
    "NAME", "DATE", "AGE", "ID", "LOCATION", "CONTACT", "PROFESSION", "USERNAME",
    # Stanford deidentifier-base specific labels
    "PATIENT", "DOCTOR", "HOSPITAL", "STAFF", "ORGANIZATION",
    "PHONE", "STREET", "CITY", "STATE", "ZIP", "COUNTRY",
    "MEDICALRECORD", "IDNUM", "BIOID", "DEVICE", "URL", "EMAIL",
}
#safe clinical terms (not to be redacted)
CLINICAL_ALLOWLIST = {"LEFT", "RIGHT", "LATERAL", "MEDIAL", "ANTERIOR", "POSTERIOR", "APICAL", "BASAL", "VIEW", "MODE", "GAIN", "DEPTH", "FREQ","B-MODE", "M-MODE", "PW", "CW", "LEFT VENTRICLE", "RIGHT VENTRICLE",
    "AORTA", "LIVER", "KIDNEY", "LUNG", "HEART", "ECG", "HR", "BP","FOV", "SLICE", "SERIES", "ECHO", "DOPPLER",}

REDACT_FILL_VALUE = 0 #pixel color value to fill in the redacted region

# DICOM metadata scrubbing
# Based on DICOM PS3.15 Annex E — Basic Application Level Confidentiality Profile (Read online for more details)
TAGS_TO_ANONYMIZE = {
    "PatientName":          "ANONYMIZED",
    "PatientID":            "ANON000000",
    "PatientBirthDate":     "",
    "PatientSex":           "",
    "PatientAge":           "",
    "PatientAddress":       "",
    "PatientTelephoneNumbers": "",
    "ReferringPhysicianName": "ANONYMIZED",
    "StudyDate":            "19000101",
    "SeriesDate":           "19000101",
    "AcquisitionDate":      "19000101",
    "ContentDate":          "19000101",
    "StudyTime":            "",
    "SeriesTime":           "",
    "AcquisitionTime":      "",
    "ContentTime":          "",
    "AccessionNumber":      "ANON",
    "StudyID":              "ANON",
    "InstitutionName":      "ANONYMIZED",
    "InstitutionAddress":   "",
    "StationName":          "ANONYMIZED",
    "OperatorsName":        "ANONYMIZED",
    "PerformingPhysicianName": "ANONYMIZED",
    "RequestingPhysician":  "ANONYMIZED",
}

TAGS_TO_DELETE = [
    "OtherPatientIDs",
    "OtherPatientNames",
    "MedicalRecordLocator",
    "EthnicGroup",
    "Occupation",
    "SmokingStatus",
    "PregnancyStatus",
    "SpecialNeeds",
    "MilitaryRank",
    "BranchOfService",
    "PatientInsurancePlanCodeSequence",
]

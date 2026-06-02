Hello, everyone.

In this project, I have tried to implement a 7-step medical image de-identification pipeline for clinical research. It removes Protected Health Information (PHI) from DICOM files, which include both burned-in pixel annotations and hidden metadata tags, and is aimed at producing privacy-safe images that comply with HIPAA's Safe Harbor method (checklist approach).

It has the following 3 components:
1 - EasyOCR to detect text burned into image pixels
2 - Stanford AIMI NER model to classify detected text as PHI or clinical label (along with regex-defined rules and a custom list)
3 - MONAI to wrap the full pipeline as a reusable medical imaging transform

## Data Source
I tried and played with publicly available TCIA data collections, LUAD and Prostate-MRI-US-Biopsy (PHI detection was negligible because per my reading, they come pre-anonymized, but metadata scrubbing was successful, and then made a custom synthetic data set of 15 DICOMs only (script included). Lastly, I also used data from the MIDI-B Validation set [1] (only partial due to limited storage).

## Tags
dicom  medical-imaging  de-identification  phi  hipaa  monai  pytorch  easyocr  clinical-ai  healthcare-ai  privacy  python


References: 
1 - Rutherford, M. W., Nolan, T., Pei, L., Wagner, U., Pan, Q., Farmer, P., Smith, K., Kopchick, B., Laura Opsahl-Ong, Sutton, G., Clunie, D. A., Farahani, K., & Prior, F. (2025). Data in Support of the MIDI-B Challenge (MIDI-B-Synthetic-Validation, MIDI-B-Curated-Validation, MIDI-B-Synthetic-Test, MIDI-B-Curated-Test) (Version 1) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/cf2p-aw56

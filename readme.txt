Hello, everyone.

In this project, I have tried to implement a 7-step medical image de-identification pipeline for clinical research. It removes Protected Health Information (PHI) from DICOM files, which include both burned-in pixel annotations and hidden metadata tags, and is aimed at producing privacy-safe images that comply with HIPAA's Safe Harbor method (checklist approach).

It has the following 3 components:
1 - EasyOCR to detect text burned into image pixels
2 - Stanford AIMI NER model to classify detected text as PHI or clinical label (along with regex-defined rules and a custom list)
3 - MONAI to wrap the full pipeline as a reusable medical imaging transform


## Tags
dicom  medical-imaging  de-identification  phi  hipaa  monai  pytorch  easyocr  clinical-ai  healthcare-ai  privacy  python

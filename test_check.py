""" This file is just made for checking if things work - a fancy print statement"""

import pydicom

cleaned = pydicom.dcmread(r"C:\Users\ammer\PycharmProjects\DICOM-Privacy\data\deidentified\PT-001234\CT\2.25.10713418310215654494.dcm")
original  = pydicom.dcmread(r"C:\Users\ammer\PycharmProjects\DICOM-Privacy\data\synthetic_phi\PT-001234\CT\2.25.10713418310215654494.dcm")

print("Raw data:")
print(f"  PatientName      : {original.get('PatientName', 'NOT FOUND')}")
print(f"  PatientBirthDate : {original.get('PatientBirthDate', 'NOT FOUND')}")
print(f"  PatientID        : {original.get('PatientID', 'NOT FOUND')}")
print(f"  InstitutionName  : {original.get('InstitutionName', 'NOT FOUND')}")

print("\nData after De-identification process:")
print(f"  PatientName      : {cleaned.get('PatientName', 'NOT FOUND')}")
print(f"  PatientBirthDate : {cleaned.get('PatientBirthDate', 'NOT FOUND')}")
print(f"  PatientID        : {cleaned.get('PatientID', 'NOT FOUND')}")
print(f"  InstitutionName  : {cleaned.get('InstitutionName', 'NOT FOUND')}")
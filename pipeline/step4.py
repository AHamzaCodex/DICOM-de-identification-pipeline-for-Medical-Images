"""each OCR detection as PHI or non-PHI terms.
Using a hybrid approach which consist of following:
  1. Clinical allowlist
  2. Regex heuristics
  3. Stanford NER model  — StanfordAIMI/stanford-deidentifier-base """

from __future__ import annotations
import re
from functools import lru_cache

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline as hf_pipeline,
)

from config import NER_MODEL_NAME, PHI_ENTITY_LABELS, CLINICAL_ALLOWLIST

# NER pipeline (lazy-loaded once)
@lru_cache(maxsize=1)
def _get_ner_pipeline():
    print(f"  [NER] Loading model '{NER_MODEL_NAME}' (first call only) ...")
    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
    model     = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
    return hf_pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",   # merge sub-word tokens into words
        device=-1,                       # CPU; change to 0 for CUDA
    )

# Regex heuristics

# "Pt: Smith John" / "Patient: Doe Jane" / "Dr: Brown David" — scanner overlays
_NAME_LABEL_RE = re.compile(
    r"^(?:Pt|Patient|Dr|Doctor|Physician|Ref|Operator|Tech)\s*[:\-]\s*"
    r"[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+",
)

_DATE_RE = re.compile(
    r"""
    \b(
        \d{1,2}[/-]\d{1,2}[/-]\d{2,4}          # 01/01/1980 or 1-1-80
      | \d{4}[/-]\d{1,2}[/-]\d{1,2}             # 1980-01-01
      | \d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|
                     Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4}  # 01 Jan 1980
      | (?:Jan|Feb|Mar|Apr|May|Jun|
           Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}  # Jan 01, 1980
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Medical Record Number
_MRN_RE = re.compile(
    r"""
    \b(
        MRN\s*[:#]?\s*\d+                        # MRN: 123456
      | ID\s*[:#]?\s*\w+                          # ID: ABC123
      | ACC\s*[:#]?\s*[\w-]+                      # ACC: 2024-001
      | \d{6,}                                    # bare 6+ digit number
      | [A-Z]{1,3}\d{4,}                          # e.g. PT00123
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Phone numbers
_PHONE_RE = re.compile(
    r"\b(\+?\d[\d\s\-().]{7,}\d)\b"
)


def _rule_based_phi(text: str) -> str | None:
    if _NAME_LABEL_RE.match(text):
        return "regex_name"
    if _DATE_RE.search(text):
        return "regex_date"
    if _MRN_RE.search(text):
        return "regex_id"
    if _PHONE_RE.search(text):
        return "regex_phone"
    return None

# Common medical label prefixes that precede a name in scanner overlays
_MEDICAL_PREFIX_RE = re.compile(
    r"^(?:Pt|Patient|Dr|Doctor|Physician|Ref(?:erring)?|Operator|Tech)\s*[:\-]\s*",
    re.IGNORECASE,
)


def _ner_phi(text: str) -> str | None:
    """Run the Stanford NER model and return "ner:<LABEL>" if any PHI entity is found, else None.
    Also checks the prefix-stripped version so 'Pt: Smith John' hits as a name."""
    ner = _get_ner_pipeline()

    candidates = [text]
    stripped = _MEDICAL_PREFIX_RE.sub("", text).strip()
    if stripped and stripped != text:
        candidates.append(stripped)

    for candidate in candidates:
        try:
            entities = ner(candidate)
        except Exception as exc:
            print(f"  [NER] Model error on '{candidate[:40]}' — {exc} — flagging as PHI for safety")
            return "ner:error"

        for ent in entities:
            label = ent.get("entity_group", "").upper()
            if label in PHI_ENTITY_LABELS:
                return f"ner:{label}"
    return None

def classify_detections(detections: list[dict]) -> list[dict]:
    """ Annotate each detection with 'is_phi' and 'phi_reason'"""
    phi_count = 0
    for det in detections:
        text = det["text"]
        # Rule 1: allowlist
        upper = text.upper()
        if any(term in upper for term in CLINICAL_ALLOWLIST):
            det["is_phi"]     = False
            det["phi_reason"] = "allowlist_safe"
            continue
        # Rule 2: regex heuristics
        reason = _rule_based_phi(text)
        if reason:
            det["is_phi"]     = True
            det["phi_reason"] = reason
            phi_count += 1
            continue
        # Rule 3: NER model
        reason = _ner_phi(text)
        if reason:
            det["is_phi"]     = True
            det["phi_reason"] = reason
            phi_count += 1
        else:
            det["is_phi"]     = False
            det["phi_reason"] = "ner:safe"

    print(f"[Step 4] {phi_count}/{len(detections)} detection(s) classified as PHI")
    return detections

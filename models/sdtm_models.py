# import pandas as pd
from pydantic import BaseModel
from typing import List, Optional

# --- Updated Pydantic Models ---


class Demographics(BaseModel):
    STUDYID: str
    USUBJID: str
    AGE: int
    SEX: str
    ARM: str
    # --- NEW COLUMNS ADDED ---
    # Made Optional in case some rows don't have them
    RACE: Optional[str] = None
    NAME: Optional[str] = None
    LOCATION: Optional[str] = None
    NO_OF_VISITS: Optional[int] = None
    DOCTOR: Optional[str] = None


class AdverseEvent(BaseModel):
    STUDYID: str
    USUBJID: str
    AETERM: str
    AESTDTC: str
    AESEV: str


class LabResult(BaseModel):
    STUDYID: str
    USUBJID: str
    LBTEST: str
    # Changed to str: Lab results can be non-numeric (e.g., "POSITIVE")
    LBORRES: str
    LBDTC: str


# --- NEW MODELS FOR NEW FILES ---


class VitalSign(BaseModel):
    STUDYID: str
    USUBJID: str
    VSTEST: str
    # Changed to str: Vital signs can also be non-numeric (e.g., "NOT DONE")
    VSORRES: str
    VSDTC: str


class MedicalHistory(BaseModel):
    STUDYID: str
    USUBJID: str
    MHTERM: str
    MHSTDTC: str


# --- Updated PatientData Model ---
# This model now holds all 5 data types


class PatientData(BaseModel):
    demographics: Optional[List[Demographics]]
    adverse_events: Optional[List[AdverseEvent]]
    lab_results: Optional[List[LabResult]]
    vital_signs: Optional[List[VitalSign]]  # <-- NEW
    medical_history: Optional[List[MedicalHistory]]  # <-- NEW

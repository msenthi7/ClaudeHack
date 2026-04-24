from pydantic import BaseModel
from typing import List, Optional


# Summary schemas
class DemographicsSummary(BaseModel):
    patient_id: str
    age: Optional[int]
    sex: Optional[str]
    arm: Optional[str]
    risk_commentary: Optional[str]
    summary: str


class AdverseEventSummary(BaseModel):
    patient_id: str
    num_events: int
    severe_events: int
    insight: str
    summary: str


class LabResultsSummary(BaseModel):
    patient_id: str
    abnormal_labs: int
    flagged_results: Optional[list]
    summary: str

import pandas as pd
from models.sdtm_models import (
    Demographics,
    AdverseEvent,
    LabResult,
    PatientData,
    VitalSign,
    MedicalHistory,
)


# def load_sdtm_data(patient_id: str):
#     # Load the SDTM CSV files
#     dm = pd.read_csv(r"D:\ACL Project\sdtm-data-process-ai\data\dm.csv")
#     ae = pd.read_csv(r"D:\ACL Project\sdtm-data-process-ai\data\ae.csv")
#     lb = pd.read_csv(
#         r"D:\ACL Project\sdtm-data-process-ai\data\lb.csv"
#     )  # Example USUBJID

#     # Filter and parse data into Pydantic models
#     dm_patient = [
#         Demographics(**row)
#         for row in dm[dm["USUBJID"] == patient_id].to_dict(orient="records")
#     ]
#     ae_patient = [
#         AdverseEvent(**row)
#         for row in ae[ae["USUBJID"] == patient_id].to_dict(orient="records")
#     ]
#     lb_patient = [
#         LabResult(**row)
#         for row in lb[lb["USUBJID"] == patient_id].to_dict(orient="records")
#     ]

#     # Combine all patient data
#     patient_data = PatientData(
#         demographics=dm_patient, adverse_events=ae_patient, lab_results=lb_patient
#     )
#     patient_data_dict = patient_data.model_dump()
#     demographics_list = patient_data_dict["demographics"]
#     adverse_events_list = patient_data_dict["adverse_events"]
#     lab_results_list = patient_data_dict["lab_results"]
#     return demographics_list, adverse_events_list, lab_results_list
# --- Updated Loading Function ---


def load_sdtm_data(patient_id: str):
    base_path = r"D:\ACL Project\sdtm-data-process-ai\data"

    try:
        # Load the SDTM CSV files
        dm = pd.read_csv(f"{base_path}\\dm.csv")
        ae = pd.read_csv(f"{base_path}\\ae.csv")
        lb = pd.read_csv(f"{base_path}\\lb.csv")
        vs = pd.read_csv(f"{base_path}\\vs.csv")
        mh = pd.read_csv(f"{base_path}\\mh.csv")

    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        return [], [], [], [], []

    # --- FIX 1: Explicitly convert LBORRES to string ---
    if "LBORRES" in lb.columns:
        lb["LBORRES"] = lb["LBORRES"].astype(str)

    # --- FIX 2: Explicitly convert VSORRES to string ---
    if "VSORRES" in vs.columns:
        vs["VSORRES"] = vs["VSORRES"].astype(str)

    # Filter and parse data into Pydantic models
    # ... (the rest of the filtering and parsing logic remains the same)

    # Demographics
    dm_patient = [
        Demographics(**row)
        for row in dm[dm["USUBJID"] == patient_id].to_dict(orient="records")
    ]

    # Adverse Events
    ae_patient = [
        AdverseEvent(**row)
        for row in ae[ae["USUBJID"] == patient_id].to_dict(orient="records")
    ]

    # Lab Results (now works because LBORRES is guaranteed to be a string)
    lb_patient = [
        LabResult(**row)
        for row in lb[lb["USUBJID"] == patient_id].to_dict(orient="records")
    ]

    # Vital Signs (now works because VSORRES is guaranteed to be a string)
    vs_patient = [
        VitalSign(**row)
        for row in vs[vs["USUBJID"] == patient_id].to_dict(orient="records")
    ]

    # Medical History
    mh_patient = [
        MedicalHistory(**row)
        for row in mh[mh["USUBJID"] == patient_id].to_dict(orient="records")
    ]

    # Combine all patient data
    patient_data = PatientData(
        demographics=dm_patient,
        adverse_events=ae_patient,
        lab_results=lb_patient,
        vital_signs=vs_patient,
        medical_history=mh_patient,
    )

    # Unpack the dictionary for the return
    patient_data_dict = patient_data.model_dump()

    demographics_list = patient_data_dict.get("demographics", [])
    adverse_events_list = patient_data_dict.get("adverse_events", [])
    lab_results_list = patient_data_dict.get("lab_results", [])
    vital_signs_list = patient_data_dict.get("vital_signs", [])
    medical_history_list = patient_data_dict.get("medical_history", [])

    return (
        demographics_list,
        adverse_events_list,
        lab_results_list,
        vital_signs_list,
        medical_history_list,
    )

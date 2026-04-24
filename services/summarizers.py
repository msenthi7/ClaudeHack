from .ollama_client import run_ollama_llm
from concurrent.futures import ThreadPoolExecutor
import subprocess


def summarize_demographics(data_list):
    if not data_list:
        return "No demographics data available."
    prompt = (
        "Summarize the following patient demographics data for a medical director:\n"
        f"{data_list}"
    )
    return run_ollama_llm(prompt)


def summarize_adverse_events(data_list):
    if not data_list:
        return "No adverse events data available."
    prompt = (
        "Summarize the adverse event details below for a medical director with clinical insights:\n"
        f"{data_list}"
    )
    return run_ollama_llm(prompt)


def summarize_lab_results(data_list):
    if not data_list:
        return "No lab results data available."
    prompt = (
        "Summarize the following lab results, highlighting clinically relevant abnormalities for a medical director:\n"
        f"{data_list}"
    )
    return run_ollama_llm(prompt)


def summarize_vital_signs(data_list):
    if not data_list:
        return "No vital signs data available."
    prompt = (
        "Summarize the following vital signs data, emphasizing any critical values for a medical director:\n"
        f"{data_list}"
    )
    return run_ollama_llm(prompt)


def summarize_medical_history(data_list):
    if not data_list:
        return "No medical history data available."
    prompt = (
        "Summarize the following medical history data, focusing on significant past conditions for a medical director:\n"
        f"{data_list}"
    )
    return run_ollama_llm(prompt)


def get_all_summaries(demo, ae, lab, vs, mh):
    with ThreadPoolExecutor() as executor:
        f1 = executor.submit(summarize_demographics, demo)
        f2 = executor.submit(summarize_adverse_events, ae)
        f3 = executor.submit(summarize_lab_results, lab)
        f4 = executor.submit(summarize_vital_signs, demo)
        f5 = executor.submit(summarize_medical_history, demo)
        return {
            "demographics_summary": f1.result(),
            "adverse_events_summary": f2.result(),
            "lab_results_summary": f3.result(),
            "vital_signs_summary": f4.result(),
            "medical_history_summary": f5.result(),
        }


def create_combined_summary(
    demo_summary: str,
    ae_summary: str,
    lab_summary: str,
    vs_summary: str,
    mh_summary: str,
) -> str:
    prompt = (
        "You are a medical director. Based on the following summaries, create a concise, "
        "clear, and clinically insightful overall narrative about the patient:\n\n"
        f"Demographics Summary:\n{demo_summary}\n\n"
        f"Adverse Events Summary:\n{ae_summary}\n\n"
        f"Lab Results Summary:\n{lab_summary}\n\n"
        f"Vital Signs Summary:\n{lab_summary}\n\n"
        f"Medical History Summary:\n{lab_summary}\n\n"
        "Provide the combined, professional summary:"
    )
    return run_ollama_llm(prompt)

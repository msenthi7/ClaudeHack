"""
Triage scoring engine.
Calculates a severity score for each patient based on labs, vitals,
adverse events, and medical history — so underserved clinics can
prioritize who needs attention first.
"""

# Normal reference ranges
LAB_RANGES = {
    "ALT":  (7,  56),    # U/L
    "AST":  (10, 40),    # U/L
    "CRP":  (0,  10),    # mg/L
}

VITAL_RANGES = {
    "SYSTOLIC BLOOD PRESSURE":  (90, 140),   # mmHg
    "DIASTOLIC BLOOD PRESSURE": (60, 90),    # mmHg
    "HEART RATE":               (60, 100),   # bpm
}

AE_SCORES = {"MILD": 1, "MODERATE": 2, "SEVERE": 4}

HIGH_RISK_CONDITIONS = {"HYPERTENSION", "DIABETES TYPE 2", "ASTHMA"}


def calculate_severity(patient_data: dict) -> dict:
    """
    Returns:
        score  (int)
        level  ("LOW" | "MODERATE" | "HIGH")
        flags  (list of human-readable alert strings)
    """
    score = 0
    flags = []

    # ── Adverse Events ────────────────────────────────────────────
    for ae in patient_data.get("adverse_events", []):
        sev = ae.get("AESEV", "MILD").upper()
        pts = AE_SCORES.get(sev, 1)
        score += pts
        if sev in ("MODERATE", "SEVERE"):
            flags.append(f"⚠️ {sev} adverse event: {ae.get('AETERM', 'Unknown')}")

    # ── Lab Results ───────────────────────────────────────────────
    for lab in patient_data.get("lab_results", []):
        test = lab.get("LBTEST", "").upper()
        if test not in LAB_RANGES:
            continue
        try:
            value = float(lab.get("LBORRES", 0))
        except (ValueError, TypeError):
            continue
        lo, hi = LAB_RANGES[test]
        if value < lo:
            score += 2
            flags.append(f"🔻 Low {test}: {value} (normal {lo}–{hi})")
        elif value > hi:
            score += 2
            flags.append(f"🔺 High {test}: {value} (normal {lo}–{hi})")

    # ── Vital Signs ───────────────────────────────────────────────
    for vs in patient_data.get("vital_signs", []):
        test = vs.get("VSTEST", "").upper()
        if test not in VITAL_RANGES:
            continue
        try:
            value = float(vs.get("VSORRES", 0))
        except (ValueError, TypeError):
            continue
        lo, hi = VITAL_RANGES[test]
        if value < lo:
            score += 2
            flags.append(f"🔻 Low {test}: {value}")
        elif value > hi:
            score += 2
            flags.append(f"🔺 High {test}: {value}")

    # ── Medical History (chronic burden) ─────────────────────────
    for mh in patient_data.get("medical_history", []):
        condition = mh.get("MHTERM", "").upper()
        score += 1
        if condition in HIGH_RISK_CONDITIONS:
            flags.append(f"📋 Chronic condition: {mh.get('MHTERM')}")

    # ── Severity Level ────────────────────────────────────────────
    if score >= 7:
        level = "HIGH"
    elif score >= 3:
        level = "MODERATE"
    else:
        level = "LOW"

    return {"score": score, "level": level, "flags": flags}

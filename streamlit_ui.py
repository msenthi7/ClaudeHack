"""
AI-Powered Triage Assistant for Underserved Clinics
====================================================
Helps doctors with limited resources quickly assess patient severity,
flag dangerous values, and get prescription safety guidance — without
needing expensive specialists.
"""

import os
import json
import pandas as pd
import streamlit as st
import ollama
import chromadb
from sentence_transformers import SentenceTransformer

from services.summarizers import get_all_summaries, create_combined_summary
from services.triage import calculate_severity

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClinIQ — AI Triage Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #f0f4f8; font-family: 'Segoe UI', sans-serif; }
    h1, h2, h3 { color: #1a2e4a; }

    .triage-high   { background:#fde8e8; border-left:5px solid #e53e3e; padding:10px; border-radius:6px; margin:4px 0; }
    .triage-mod    { background:#fef9e7; border-left:5px solid #d4ac0d; padding:10px; border-radius:6px; margin:4px 0; }
    .triage-low    { background:#e8f5e9; border-left:5px solid #38a169; padding:10px; border-radius:6px; margin:4px 0; }

    .flag-box      { background:#fff3cd; border:1px solid #ffc107; border-radius:5px; padding:8px 12px; margin:3px 0; font-size:0.9rem; }
    .metric-card   { background:white; border-radius:8px; padding:12px; box-shadow:0 1px 4px rgba(0,0,0,0.1); text-align:center; }
    .patient-plain { background:#e8f4fd; border-left:4px solid #3498db; padding:14px; border-radius:6px; font-size:0.95rem; }

    section[data-testid="stSidebar"] { background:#1a2e4a; }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    div.stButton > button { border-radius:6px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_PATH       = os.path.join(BASE_DIR, "data")
DB_PATH         = os.path.join(BASE_DIR, "db")
COLLECTION_NAME = "medical_knowledge"
MODEL_NAME      = "all-MiniLM-L6-v2"
LLM_MODEL       = "llama3.1"

LEVEL_EMOJI = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}

# ── Ollama availability check ─────────────────────────────────────────────────
def _ollama_available() -> bool:
    try:
        ollama.list()
        return True
    except Exception:
        return False

OLLAMA_READY = _ollama_available()
LEVEL_CSS   = {"HIGH": "triage-high", "MODERATE": "triage-mod", "LOW": "triage-low"}

# ── Data Loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_all_data():
    files = {"AE": "ae.csv", "DM": "dm.csv", "LB": "lb.csv", "MH": "mh.csv", "VS": "vs.csv"}
    dfs = {}
    for domain, fname in files.items():
        path = os.path.join(DATA_PATH, fname)
        if os.path.exists(path):
            dfs[domain] = pd.read_csv(path, low_memory=False)
        else:
            st.error(f"Missing file: {fname}")
            return None
    return dfs


def get_patient_dict(usubjid: str, dfs: dict) -> dict:
    def rows(domain):
        df = dfs.get(domain, pd.DataFrame())
        if "USUBJID" not in df.columns:
            return []
        return df[df["USUBJID"] == usubjid].to_dict(orient="records")

    return {
        "id":              usubjid,
        "demographics":    rows("DM"),
        "adverse_events":  rows("AE"),
        "lab_results":     rows("LB"),
        "vital_signs":     rows("VS"),
        "medical_history": rows("MH"),
    }


@st.cache_data
def build_triage_table(_dfs: dict) -> pd.DataFrame:
    dfs = _dfs
    dm = dfs["DM"]
    records = []
    for _, row in dm.iterrows():
        uid = row["USUBJID"]
        pdata = get_patient_dict(uid, dfs)
        result = calculate_severity(pdata)
        records.append({
            "ID":       uid,
            "Name":     row.get("NAME", "—"),
            "Age":      row.get("AGE", "—"),
            "Sex":      row.get("SEX", "—"),
            "Doctor":   row.get("DOCTOR", "—"),
            "Location": row.get("LOCATION", "—"),
            "Score":    result["score"],
            "Level":    result["level"],
            "Flags":    result["flags"],
        })
    df = pd.DataFrame(records).sort_values("Score", ascending=False).reset_index(drop=True)
    return df

# ── Knowledge Base Retrieval ──────────────────────────────────────────────────

@st.cache_resource
def get_kb():
    """Returns (embed_model, collection) or (None, None) if not ready."""
    try:
        embed_model = SentenceTransformer(MODEL_NAME)
        client      = chromadb.PersistentClient(path=DB_PATH)
        collection  = client.get_collection(COLLECTION_NAME)
        return embed_model, collection
    except Exception:
        return None, None


def retrieve_context(query: str, n: int = 4) -> str:
    """Retrieves top-n relevant chunks from the medical knowledge base."""
    embed_model, collection = get_kb()
    if embed_model is None:
        return ""
    try:
        vec     = embed_model.encode([query]).tolist()
        results = collection.query(query_embeddings=vec, n_results=n)
        docs    = results["documents"][0] if results["documents"] else []
        return "\n\n".join(f"[Source {i+1}]: {d}" for i, d in enumerate(docs))
    except Exception:
        return ""


# ── Prescription Engine ───────────────────────────────────────────────────────

@st.cache_resource
def get_engine():
    return PrescriptionEngine()


class PrescriptionEngine:
    def __init__(self):
        self.embed, self.col = get_kb()
        self.ready = self.embed is not None

    def _llm(self, prompt: str, fmt: str | None = None) -> str:
        kwargs = {
            "model":    LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "options":  {"temperature": 0},
        }
        if fmt:
            kwargs["format"] = fmt  # top-level param, not inside options
        r = ollama.chat(**kwargs)
        return r["message"]["content"]

    def _safe_json(self, text: str):
        try:
            return json.loads(text.replace("```json", "").replace("```", "").strip())
        except Exception:
            return None

    def gatekeeper(self, query: str) -> dict:
        prompt = f"""
        Is this a request for drug prescription, medical treatment or therapy?
        INPUT: "{query}"
        OUTPUT JSON ONLY: {{"is_prescription": true/false, "reason": "brief"}}
        """
        data = self._safe_json(self._llm(prompt, "json"))
        return data or {"is_prescription": False, "reason": "Parse error"}

    def retrieve(self, query: str):
        if not self.ready:
            return None, 0
        vec     = self.embed.encode([query]).tolist()
        results = self.col.query(query_embeddings=vec, n_results=4)
        if not results["documents"] or not results["documents"][0]:
            return None, 0
        context = "\n".join(f"- {d}" for d in results["documents"][0])
        prompt  = f"""
        You are a medical auditor. Rate ONLY using the provided context, not outside knowledge.
        QUERY: "{query}"
        CONTEXT FROM MEDICAL KNOWLEDGE BASE: "{context}"
        Rate 0-100 how well the context answers the query.
        OUTPUT JSON ONLY: {{"relevance_score": 0, "reason": "why"}}
        """
        data = self._safe_json(self._llm(prompt, "json"))
        score = data.get("relevance_score", 0) if data else 0
        return context, score

    def safety_check(self, treatment: str, patient: dict) -> dict:
        kb_context = retrieve_context(f"contraindications safety {treatment}")
        prompt = f"""
        You are a Clinical Safety Officer. Base your answer ONLY on the knowledge base context and patient data below.
        Do NOT use outside medical knowledge.

        MEDICAL KNOWLEDGE BASE CONTEXT:
        {kb_context if kb_context else "No relevant context found."}

        TREATMENT PROPOSED: {treatment}
        PATIENT DATA: {json.dumps(patient)}

        Check for contraindications using lab values (AST, ALT, CRP) and adverse events.
        If context is insufficient, flag as CONFLICT with reason "Insufficient knowledge base data."
        OUTPUT JSON ONLY: {{"status": "SAFE" or "CONFLICT", "reason": "clinical explanation"}}
        """
        data = self._safe_json(self._llm(prompt, "json"))
        return data or {"status": "CONFLICT", "reason": "Safety check failed"}

    def alternative(self, query: str, reason: str, patient: dict) -> str:
        kb_context = retrieve_context(f"alternative treatment {query}")
        prompt = f"""
        You are a Senior Medical Consultant. Use ONLY the knowledge base context below.
        Do NOT use outside medical knowledge. If the knowledge base lacks information, say so clearly.

        MEDICAL KNOWLEDGE BASE CONTEXT:
        {kb_context if kb_context else "No relevant context found in knowledge base."}

        REQUEST: "{query}"
        BLOCKED BECAUSE: {reason}
        PATIENT: {json.dumps(patient)}

        Suggest a safe, accessible alternative based ONLY on the above context. Use Markdown:
        ### ⚠️ Clinical Alert
        ### 🔄 Recommended Alternative
        ### 🧬 Why This Is Safe (cite knowledge base)
        ### 💊 Dosage & Monitoring
        """
        return self._llm(prompt)

# ── Tool Definitions for Chat ─────────────────────────────────────────────────

def get_current_patient():
    return st.session_state.get("patient_data", {})

TOOL_FNS = {
    "get_full_patient_summary":  lambda: json.dumps(get_current_patient()),
    "get_patient_demographics":  lambda: json.dumps(get_current_patient().get("demographics", [])),
    "get_adverse_events":        lambda: json.dumps(get_current_patient().get("adverse_events", [])),
    "get_lab_results":           lambda: json.dumps(get_current_patient().get("lab_results", [])),
    "get_vital_signs":           lambda: json.dumps(get_current_patient().get("vital_signs", [])),
    "get_medical_history":       lambda: json.dumps(get_current_patient().get("medical_history", [])),
}

TOOLS = [
    {"type": "function", "function": {
        "name": "get_full_patient_summary",
        "description": "Use for general overviews or 'tell me about the patient'.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "get_patient_demographics",
        "description": "Use for age, sex, race, or study ID questions.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "get_adverse_events",
        "description": "Use for side effects or symptom questions.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "get_lab_results",
        "description": "Use for lab values or test results.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "get_vital_signs",
        "description": "Use for blood pressure, heart rate, or other vitals.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "get_medical_history",
        "description": "Use for past conditions or chronic diseases.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
]

# ── UI: Triage Dashboard ──────────────────────────────────────────────────────

def show_triage_dashboard(dfs):
    with st.sidebar:
        st.markdown("# 🏥")
        st.title("ClinIQ")
        st.caption("AI Triage Assistant\nfor Underserved Clinics")
        st.markdown("---")
        st.info("Select a patient from the dashboard to begin analysis.")

    st.markdown("## 🏥 Patient Triage Dashboard")
    st.caption("Patients ranked by AI-calculated severity score. Red = immediate attention needed.")
    st.markdown("---")

    triage_df = build_triage_table(_dfs=dfs)

    # Summary metrics
    high  = len(triage_df[triage_df["Level"] == "HIGH"])
    mod   = len(triage_df[triage_df["Level"] == "MODERATE"])
    low   = len(triage_df[triage_df["Level"] == "LOW"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", len(triage_df))
    c2.metric("🔴 High Priority",     high)
    c3.metric("🟡 Moderate Priority", mod)
    c4.metric("🟢 Low Priority",      low)

    st.markdown("---")

    # ── Search & Filter ───────────────────────────────────────────
    col_search, col_filter = st.columns([3, 1])
    with col_search:
        search_term = st.text_input("🔍 Search patient by name or ID:", placeholder="e.g. Jane or SUBJ02")
    with col_filter:
        priority_filter = st.selectbox("Priority filter:", ["All", "🔴 High", "🟡 Moderate", "🟢 Low"])

    filtered_df = triage_df.copy()
    if search_term:
        mask = (
            filtered_df["Name"].astype(str).str.contains(search_term, case=False, na=False) |
            filtered_df["ID"].astype(str).str.contains(search_term, case=False, na=False)
        )
        filtered_df = filtered_df[mask]
    if priority_filter != "All":
        level_map = {"🔴 High": "HIGH", "🟡 Moderate": "MODERATE", "🟢 Low": "LOW"}
        filtered_df = filtered_df[filtered_df["Level"] == level_map[priority_filter]]

    if filtered_df.empty:
        st.warning("No patients match your search.")

    st.caption(f"Showing {len(filtered_df)} of {len(triage_df)} patients")
    st.markdown("---")

    for _, row in filtered_df.iterrows():
        css   = LEVEL_CSS[row["Level"]]
        emoji = LEVEL_EMOJI[row["Level"]]
        flags = row["Flags"]

        col_info, col_btn = st.columns([5, 1])
        with col_info:
            st.markdown(f"""
            <div class="{css}">
                <strong>{emoji} {row['ID']} — {row['Name']}</strong> &nbsp;|&nbsp;
                Age: {row['Age']} &nbsp;|&nbsp; {row['Sex']} &nbsp;|&nbsp;
                Doctor: {row['Doctor']} &nbsp;|&nbsp; Site: {row['Location']}<br>
                <small>Severity Score: <strong>{row['Score']}</strong> &nbsp;·&nbsp; {row['Level']}</small>
                {"<br><small>" + " &nbsp; ".join(flags[:2]) + ("..." if len(flags) > 2 else "") + "</small>" if flags else ""}
            </div>
            """, unsafe_allow_html=True)

        with col_btn:
            st.write("")
            if st.button("View →", key=f"btn_{row['ID']}"):
                st.session_state.patient_id   = row["ID"]
                st.session_state.patient_data = get_patient_dict(row["ID"], dfs)
                st.session_state.triage_info  = {"score": row["Score"], "level": row["Level"], "flags": flags}
                st.session_state.chat_history = []
                st.session_state.pop("summary_cache", None)
                st.rerun()

# ── UI: Patient Detail ────────────────────────────────────────────────────────

def show_patient_detail():
    uid         = st.session_state.patient_id
    pdata       = st.session_state.patient_data
    triage      = st.session_state.triage_info
    demos       = pdata["demographics"][0] if pdata["demographics"] else {}

    level = triage["level"]
    emoji = LEVEL_EMOJI[level]

    with st.sidebar:
        st.markdown("# 🏥")
        st.title("ClinIQ")
        st.markdown("---")
        st.markdown(f"**Active Patient**\n\n`{uid}`")
        st.markdown(f"**Priority:** {emoji} {level}")
        st.markdown(f"**Score:** {triage['score']}")
        st.markdown("---")
        if st.button("← Back to Triage", use_container_width=True):
            for k in ["patient_id", "patient_data", "triage_info", "chat_history", "summary_cache", "plain_summary"]:
                st.session_state.pop(k, None)
            st.rerun()
        st.markdown("---")
        if triage["flags"]:
            st.markdown("**Active Alerts**")
            for f in triage["flags"]:
                st.markdown(f"- {f}")

    # Hero
    st.markdown(f"## {emoji} Patient: {uid} — {demos.get('NAME', '')}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Age",      demos.get("AGE", "—"))
    c2.metric("Sex",      demos.get("SEX", "—"))
    c3.metric("Doctor",   demos.get("DOCTOR", "—"))
    c4.metric("Priority", f"{emoji} {level}")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🚨 Triage Alerts",
        "📄 Clinical Report",
        "🗣️ Patient Summary",
        "💊 Prescription Safety",
    ])

    # ── Tab 1: Triage Alerts ─────────────────────────────────────
    with tab1:
        st.markdown("#### Risk Flags")
        if triage["flags"]:
            for flag in triage["flags"]:
                st.markdown(f'<div class="flag-box">{flag}</div>', unsafe_allow_html=True)
        else:
            st.success("No critical flags detected for this patient.")

        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("**Lab Results**")
            if pdata["lab_results"]:
                lb_df = pd.DataFrame(pdata["lab_results"])[["LBTEST", "LBORRES", "LBDTC"]]
                lb_df.columns = ["Test", "Result", "Date"]
                st.dataframe(lb_df, use_container_width=True, hide_index=True)

        with col_b:
            st.markdown("**Vital Signs**")
            if pdata["vital_signs"]:
                vs_df = pd.DataFrame(pdata["vital_signs"])[["VSTEST", "VSORRES", "VSDTC"]]
                vs_df.columns = ["Test", "Result", "Date"]
                st.dataframe(vs_df, use_container_width=True, hide_index=True)

        with col_c:
            st.markdown("**Adverse Events**")
            if pdata["adverse_events"]:
                ae_df = pd.DataFrame(pdata["adverse_events"])[["AETERM", "AESEV", "AESTDTC"]]
                ae_df.columns = ["Event", "Severity", "Date"]
                st.dataframe(ae_df, use_container_width=True, hide_index=True)
            else:
                st.info("No adverse events recorded.")

    # ── Tab 2: Clinical Report ────────────────────────────────────
    with tab2:
        st.markdown("#### AI-Generated Clinical Summary")
        st.caption("Comprehensive report for the treating physician.")

        if not OLLAMA_READY:
            st.warning("⚠️ Ollama is not running. Start it with `ollama serve` then pull a model with `ollama pull llama3.1`.", icon="⚠️")

        if "summary_cache" not in st.session_state:
            if st.button("⚡ Generate Clinical Report", type="primary", disabled=not OLLAMA_READY):
                with st.spinner("Analysing patient records with AI..."):
                    try:
                        summaries = get_all_summaries(
                            pdata["demographics"],
                            pdata["adverse_events"],
                            pdata["lab_results"],
                            pdata["vital_signs"],
                            pdata["medical_history"],
                        )
                        combined = create_combined_summary(
                            summaries["demographics_summary"],
                            summaries["adverse_events_summary"],
                            summaries["lab_results_summary"],
                            summaries["vital_signs_summary"],
                            summaries["medical_history_summary"],
                        )
                        st.session_state["summary_cache"] = combined
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.success("Report ready")
            st.markdown(st.session_state["summary_cache"])
            if st.button("🔄 Regenerate"):
                del st.session_state["summary_cache"]
                st.rerun()

    # ── Tab 3: Patient-Friendly Summary ──────────────────────────
    with tab3:
        st.markdown("#### Plain-Language Patient Summary")
        st.caption("Translates clinical data into simple language the patient can understand — no jargon.")

        if not OLLAMA_READY:
            st.warning("⚠️ Ollama is not running. Start it with `ollama serve` then pull a model with `ollama pull llama3.1`.", icon="⚠️")

        if "plain_summary" not in st.session_state:
            if st.button("🗣️ Generate Patient Explanation", type="primary", disabled=not OLLAMA_READY):
                with st.spinner("Translating medical data into plain language..."):
                    try:
                        prompt = f"""
                        You are a compassionate doctor explaining a patient's health to them directly.
                        Use SIMPLE, warm language — no medical jargon. Explain what their numbers mean,
                        what they should watch out for, and one actionable next step.

                        Patient data:
                        - Demographics: {json.dumps(pdata['demographics'])}
                        - Lab Results: {json.dumps(pdata['lab_results'])}
                        - Vital Signs: {json.dumps(pdata['vital_signs'])}
                        - Adverse Events: {json.dumps(pdata['adverse_events'])}
                        - Medical History: {json.dumps(pdata['medical_history'])}

                        Write in second person ("Your blood pressure..."). Keep it under 200 words.
                        End with one clear action they should take.
                        """
                        resp = ollama.chat(
                            model=LLM_MODEL,
                            messages=[{"role": "user", "content": prompt}],
                            options={"temperature": 0.3},
                        )
                        st.session_state["plain_summary"] = resp["message"]["content"]
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.markdown(
                f'<div class="patient-plain">{st.session_state["plain_summary"]}</div>',
                unsafe_allow_html=True,
            )
            if st.button("🔄 Regenerate"):
                del st.session_state["plain_summary"]
                st.rerun()

    # ── Tab 4: Prescription Safety ────────────────────────────────
    with tab4:
        st.markdown("#### 💊 Prescription Safety Engine")
        st.caption("Checks drug orders against patient labs, vitals, and history using real FDA drug data.")

        engine = get_engine()
        if not engine.ready:
            st.warning(
                "⚠️ Knowledge base not found. "
                "Run `python scripts/seed_kb.py` first to load drug data.",
                icon="⚠️",
            )
            st.info("The prescription engine will still run safety checks using AI reasoning.")

        col_inp, col_btn = st.columns([4, 1])
        with col_inp:
            order = st.text_input(
                "Enter Medical Order:",
                placeholder="e.g., Prescribe ibuprofen 400mg for pain",
            )
        with col_btn:
            st.write("")
            st.write("")
            run = st.button("🚀 Check Safety", type="primary", use_container_width=True)

        if run:
            if not order:
                st.warning("Please enter a medical order.")
            else:
                st.divider()

                # Step 1: Gatekeeper
                gate = None
                with st.status("Step 1: Validating request type...", expanded=True) as s:
                    gate = engine.gatekeeper(order)
                    if not gate["is_prescription"]:
                        s.update(label="⛔ Not a prescription request", state="error")
                    else:
                        s.update(label="✅ Valid prescription request", state="complete")

                if not gate["is_prescription"]:
                    st.error(f"**Triage Gate:** {gate['reason']}")
                else:
                    # Step 2: Knowledge Base
                    context, score = None, 0
                    with st.status("Step 2: Searching FDA drug database...", expanded=True) as s:
                        context, score = engine.retrieve(order)
                        if score < 60 or not context:
                            s.update(label=f"⚠️ Low database match ({score}%) — using AI reasoning", state="complete")
                            context = f"AI reasoning only (no high-confidence protocol found for: {order})"
                        else:
                            s.update(label=f"✅ Protocol found ({score}% match)", state="complete")

                    if score >= 60 and context:
                        with st.expander("View Protocol Evidence"):
                            st.write(context)

                    # Step 3: Safety Check
                    safety = None
                    with st.status("Step 3: Running patient safety check...", expanded=True) as s:
                        safety = engine.safety_check(context, pdata)
                        if safety["status"] == "CONFLICT":
                            s.update(label="🛑 Safety conflict detected", state="error")
                        else:
                            s.update(label="✅ Safety check passed", state="complete")

                    if safety["status"] == "CONFLICT":
                        st.error(f"**⛔ SAFETY ALERT:** {safety['reason']}")
                        st.markdown("#### 🔄 Generating Safe Alternative...")
                        alt = engine.alternative(order, safety["reason"], pdata)
                        st.markdown(alt)
                    else:
                        st.success("✅ Prescription Authorized")
                        st.markdown(f"""
                        **Order:** {order}

                        **Safety Validation:** {safety['reason']}
                        """)

    # ── Floating Chat Assistant ───────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💬 Ask the AI About This Patient")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{
            "role":    "system",
            "content": (
                "You are a Clinical Assistant at an underserved clinic. "
                "STRICT RULES:\n"
                "1. For medical knowledge questions, you MUST ONLY use the MEDICAL KNOWLEDGE BASE CONTEXT injected into each user message. "
                "Do NOT use outside knowledge. If the answer is not in the context, say: "
                "'This information is not available in the medical knowledge base.'\n"
                "2. For patient-specific data (labs, vitals, history), use the provided tools.\n"
                "3. Be concise and clinically precise."
            ),
        }]

    chat_box = st.container(height=400)
    with chat_box:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.chat_message("user", avatar="🧑‍⚕️").write(msg["content"])
            elif msg["role"] == "assistant":
                st.chat_message("assistant", avatar="🩺").write(msg["content"])

    if not OLLAMA_READY:
        st.warning("⚠️ Ollama not running — chat disabled. Run `ollama serve` in a terminal.", icon="⚠️")

    if OLLAMA_READY and (prompt := st.chat_input("Ask about labs, vitals, history, or risks...")):
        # Retrieve KB context for this query and inject into the message
        kb_context = retrieve_context(prompt)
        augmented_content = prompt
        if kb_context:
            augmented_content = (
                f"{prompt}\n\n"
                f"--- MEDICAL KNOWLEDGE BASE CONTEXT (use ONLY this for medical facts) ---\n"
                f"{kb_context}\n"
                f"--- END CONTEXT ---"
            )

        # Show original prompt to user, send augmented version to LLM
        st.session_state.chat_history.append({"role": "user", "content": augmented_content})
        with chat_box:
            st.chat_message("user", avatar="🧑‍⚕️").write(prompt)
            with st.spinner("Analysing..."):
                try:
                    resp = ollama.chat(
                        model=LLM_MODEL,
                        messages=st.session_state.chat_history,
                        tools=TOOLS,
                        options={"temperature": 0.0},
                    )
                    if resp["message"].get("tool_calls"):
                        st.session_state.chat_history.append(resp["message"])
                        for tc in resp["message"]["tool_calls"]:
                            fn     = tc["function"]["name"]
                            result = TOOL_FNS[fn]()
                            st.session_state.chat_history.append({"role": "tool", "content": result})
                        final = ollama.chat(
                            model=LLM_MODEL,
                            messages=st.session_state.chat_history,
                            options={"temperature": 0.0},
                        )
                        reply = final["message"]["content"]
                        st.session_state.chat_history.append(final["message"])
                    else:
                        reply = resp["message"]["content"]
                        st.session_state.chat_history.append(resp["message"])
                    st.chat_message("assistant", avatar="🩺").write(reply)
                except Exception as e:
                    st.error(f"AI Error: {e}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if "all_data" not in st.session_state:
        with st.spinner("Loading patient data..."):
            st.session_state.all_data = load_all_data()
        if st.session_state.all_data is None:
            st.stop()

    if "patient_id" in st.session_state:
        show_patient_detail()
    else:
        show_triage_dashboard(st.session_state.all_data)


if __name__ == "__main__":
    main()

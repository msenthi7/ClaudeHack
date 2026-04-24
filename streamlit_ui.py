import streamlit as st
import pandas as pd
import os
import json
import time
import sys
from typing import TypedDict, Annotated, Literal

# AI / Backend Imports
import ollama
import chromadb
from sentence_transformers import SentenceTransformer

# NOTE: Ensure these modules exist in your project folder as per your previous code
from services.summarizers import get_all_summaries, create_combined_summary

# ============================================================================
# STREAMLIT PAGE CONFIGURATION & CUSTOM STYLING
# ============================================================================

st.set_page_config(
    page_title="SDTM Clinical Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a Professional Clinical Look
st.markdown(
    """
<style>
    /* Main Background and Text */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 600;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Buttons */
    div.stButton > button:first-child {
        border-radius: 8px;
        font-weight: 600;
    }
    div.stButton > button:active {
        background-color: #0056b3;
    }

    /* Cards/Expanders */
    .streamlit-expanderHeader {
        background-color: #ffffff;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
    }
    
    /* Chat Interface */
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    
    /* Custom Metric Box */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #007bff;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# PART 0: CONFIGURATION & CONSTANTS
# ============================================================================

# !!! IMPORTANT: Update paths to match your local system !!!
DATA_PATH = r"D:\ACL Project\sdtm-data-process-ai\data"
DB_PATH = r"D:\ACL Project\sdtm-data-process-ai\db_creator\db_creator"

COLLECTION_NAME = "medical_knowledge"
MODEL_NAME = "all-MiniLM-L6-v2"

# ============================================================================
# PART 1: DATA LOADING (FROM UI CODE)
# ============================================================================


@st.cache_data
def load_all_data(base_path):
    """Loads all SDTM data from the specified folder and returns a dictionary of DataFrames."""
    files = {
        "AE": "ae.csv",
        "DM": "dm.csv",
        "LB": "lb.csv",
        "MH": "mh.csv",
        "VS": "vs.csv",
    }
    dataframes = {}

    if not os.path.isdir(base_path):
        st.error(f"Error: The specified data folder does not exist: {base_path}")
        return None

    all_files_found = True
    for domain, file_name in files.items():
        full_path = os.path.join(base_path, file_name)
        if os.path.exists(full_path):
            dataframes[domain] = pd.read_csv(full_path, low_memory=False)
        else:
            st.error(f"Error: Could not find the file: {full_path}")
            all_files_found = False

    if not all_files_found:
        return None

    if "DM" not in dataframes or dataframes["DM"].empty:
        st.error("Error: Failed to load critical 'DM' (Demographics) data.")
        return None

    return dataframes


def fetch_patient_data(usubjid, all_data):
    """Fetch all patient data across SDTM domains as DataFrames"""
    patient_data = {}
    for domain, df in all_data.items():
        if "USUBJID" in df.columns:
            patient_data[domain] = df[df["USUBJID"] == usubjid]
    return patient_data


def reset_patient_specific_data():
    """Reset all patient-specific session state variables"""
    keys_to_reset = [
        "chat_messages",
        "current_patient_data",  # The merged data structure
        "chat_history",  # For the UI chat
        "prescription_history",
        "summary_cache",  # <--- ADDED THIS: Clears the summary when switching
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state.patient_confirmed = False
    if "confirmed_patient" in st.session_state:
        del st.session_state["confirmed_patient"]


# ============================================================================
# PART 2: BACKEND LOGIC CLASSES
# ============================================================================


@st.cache_resource
def get_clinical_engine():
    """Initializes and caches the heavy DB connection"""
    return ClinicalDecisionEngine()


class ClinicalDecisionEngine:
    def __init__(self):
        try:
            self.embed_model = SentenceTransformer(MODEL_NAME)
            self.db_client = chromadb.PersistentClient(path=DB_PATH)
            self.collection = self.db_client.get_collection(COLLECTION_NAME)
            print("Medical Knowledge Base Connected")
        except Exception as e:
            st.error(f"Database Startup Error: {e}")

    def parse_json_safe(self, text_output):
        try:
            clean_text = text_output.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except json.JSONDecodeError:
            return None

    def gatekeeper_check(self, user_input):
        prompt = f"""
        You are a Clinical Triage System.
        INPUT: "{user_input}"
        TASK: Is this a request for medical treatment, drug prescription, or clinical therapy?
        RULES:
        - "What is the weather?" -> false
        - "Prescribe medication for headache" -> true
        - "Is Paracetamol safe?" -> true
        OUTPUT JSON ONLY. NO MARKDOWN.
        Format: {{"is_prescription": true/false, "reason": "brief explanation"}}
        """
        response = ollama.chat(
            model="llama3.1",
            messages=[{"role": "user", "content": prompt}],
            options={"format": "json", "temperature": 0},
        )
        data = self.parse_json_safe(response["message"]["content"])
        if not data:
            return {
                "is_prescription": False,
                "reason": "System Error: Could not parse intent.",
            }
        return data

    def retrieve_and_evaluate_context(self, query):
        query_vec = self.embed_model.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_vec, n_results=3)

        # Check if documents exist
        if not results["documents"] or not results["documents"][0]:
            return None, 0

        context_text = "\n".join([f"- {doc}" for doc in results["documents"][0]])

        prompt = f"""
        You are a Medical Data Auditor.
        USER QUERY: "{query}"
        RETRIEVED DB CONTEXT: "{context_text}"
        TASK: Rate how well this context answers the specific medical question. Scale: 0 to 100.
        OUTPUT JSON ONLY. NO MARKDOWN. Format: {{"relevance_score": 0, "reason": "why"}}
        """
        response = ollama.chat(
            model="llama3.1",
            messages=[{"role": "user", "content": prompt}],
            options={"format": "json", "temperature": 0},
        )
        data = self.parse_json_safe(response["message"]["content"])
        if not data:
            return context_text, 0
        return context_text, data.get("relevance_score", 0)

    def patient_safety_check(self, treatment_plan, patient_data):
        prompt = f"""
        You are a Clinical Safety Officer.
        1. PROPOSED TREATMENT: {treatment_plan}
        2. PATIENT DATA: {json.dumps(patient_data)}
        TASK: Check for CONTRAINDICATIONS.
        - Look specifically at Lab Results (e.g., AST, ALT, Creatinine).
        - Look at Adverse Events.
        OUTPUT JSON ONLY. NO MARKDOWN.
        Format: {{ "status": "SAFE" or "CONFLICT", "reason": "clinical explanation using the lab values" }}
        """
        response = ollama.chat(
            model="llama3.1",
            messages=[{"role": "user", "content": prompt}],
            options={"format": "json", "temperature": 0},
        )
        data = self.parse_json_safe(response["message"]["content"])
        if not data:
            return {
                "status": "CONFLICT",
                "reason": "System Error: Could not verify safety.",
            }
        return data

    def generate_alternative(self, original_query, failure_reason, patient_data):
        prompt = f"""
        You are a Senior Medical Consultant.
        SITUATION:
        - User requested: "{original_query}"
        - SYSTEM BLOCKED REQUEST BECAUSE: {failure_reason}
        - Patient Profile: {json.dumps(patient_data)}
        TASK: Suggest a CLINICAL ALTERNATIVE that is safe for this specific patient.
        OUTPUT FORMAT (Markdown): "### ⚠️ Clinical Alert... ### 🔄 Recommended Alternative... ### 🧬 Why this is Safe..."
        """
        response = ollama.chat(
            model="llama3.1",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
        )
        return response["message"]["content"]


# ============================================================================
# PART 3: TOOL DEFINITIONS
# ============================================================================


def get_current_patient():
    """Helper to safely get data from session state"""
    if "current_patient_data" not in st.session_state:
        return {}
    return st.session_state.current_patient_data


def get_full_patient_summary():
    return json.dumps(get_current_patient())


def get_patient_demographics():
    return json.dumps(get_current_patient().get("demographics", []))


def get_adverse_events():
    return json.dumps(get_current_patient().get("adverse_events", []))


def get_lab_results():
    return json.dumps(get_current_patient().get("lab_results", []))


def get_vital_signs():
    return json.dumps(get_current_patient().get("vital_signs", []))


def get_medical_history():
    return json.dumps(get_current_patient().get("medical_history", []))


available_functions = {
    "get_full_patient_summary": get_full_patient_summary,
    "get_patient_demographics": get_patient_demographics,
    "get_adverse_events": get_adverse_events,
    "get_lab_results": get_lab_results,
    "get_vital_signs": get_vital_signs,
    "get_medical_history": get_medical_history,
}

my_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_full_patient_summary",
            "description": 'Use this when the user asks for a GENERAL OVERVIEW, a summary, or says "tell me about the patient". It retrieves everything.',
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_patient_demographics",
            "description": "Use ONLY for specific questions about age, sex, race, or study ID.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_adverse_events",
            "description": "Use ONLY for specific questions about side effects or symptoms.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_lab_results",
            "description": "Use ONLY for specific questions about lab values or test scores.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_vital_signs",
            "description": "Use ONLY for specific questions about vital signs like blood pressure, heart rate, etc.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_medical_history",
            "description": "Use ONLY for specific questions about past medical conditions or history.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# ============================================================================
# PART 4: MAIN UI FUNCTIONS
# ============================================================================


def display_patient_search():
    """Display patient search interface with filters"""
    st.markdown("## 🔍 Patient Data Discovery")

    col_main, col_metrics = st.columns([3, 1])

    with col_metrics:
        st.info("System Ready")

    with col_main:
        st.write(
            "Use the filters below to locate a patient record from the SDTM database."
        )

    st.markdown("---")

    all_data = st.session_state.all_data
    dm_df = all_data["DM"]

    # Sidebar for Filters
    with st.sidebar:
        st.header("🔎 Filter Criteria")
        st.write("Refine your patient search:")

        search_term = st.text_input(
            "Name or ID Search:", placeholder="e.g., John or SUBJ01"
        )

        # Get unique values for filters
        unique_sex = dm_df["SEX"].unique() if "SEX" in dm_df.columns else []
        unique_race = dm_df["RACE"].unique() if "RACE" in dm_df.columns else []
        unique_arm = dm_df["ARM"].unique() if "ARM" in dm_df.columns else []

        sex_filter = st.multiselect("Sex:", options=unique_sex, default=[])
        race_filter = st.multiselect("Race:", options=unique_race, default=[])
        arm_filter = st.multiselect("Study Arm:", options=unique_arm, default=[])

    # Filtering Logic
    filtered_df = dm_df.copy()
    if search_term:
        search_condition = pd.Series(
            [False] * len(filtered_df), index=filtered_df.index
        )
        if "NAME" in filtered_df.columns:
            search_condition |= (
                filtered_df["NAME"]
                .astype(str)
                .str.contains(search_term, case=False, na=False)
            )
        if "USUBJID" in filtered_df.columns:
            search_condition |= (
                filtered_df["USUBJID"]
                .astype(str)
                .str.contains(search_term, case=False, na=False)
            )
        filtered_df = filtered_df[search_condition]

    if sex_filter and "SEX" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["SEX"].isin(sex_filter)]
    if arm_filter and "ARM" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["ARM"].isin(arm_filter)]
    if race_filter and "RACE" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["RACE"].isin(race_filter)]

    # Display Results
    col_res1, col_res2 = st.columns([2, 1])
    with col_res1:
        st.subheader(f"Results ({len(filtered_df)})")

    if not filtered_df.empty:
        display_cols = ["USUBJID", "NAME", "AGE", "SEX", "ARM", "RACE"]
        available_cols = [col for col in display_cols if col in filtered_df.columns]

        # Use a container for the table
        with st.container():
            st.dataframe(
                filtered_df[available_cols], use_container_width=True, height=300
            )

        st.markdown("### 📋 Select Patient Record")

        col_sel, col_btn = st.columns([3, 1])

        # Create list for selectbox
        if "NAME" in filtered_df.columns:
            patient_list = (
                filtered_df["USUBJID"].astype(str)
                + " ("
                + filtered_df["NAME"].astype(str)
                + ")"
            )
        else:
            patient_list = filtered_df["USUBJID"].astype(str)

        with col_sel:
            selected_patient_str = st.selectbox(
                "Choose a patient to analyze:", options=patient_list.unique()
            )

        if selected_patient_str:
            selected_usubjid = selected_patient_str.split(" ")[0]

            # Fetch DF data for preview
            patient_full_data_df = fetch_patient_data(selected_usubjid, all_data)

            # Preview section
            with st.expander(
                "📄 View Domain Data Preview (Click to Expand)", expanded=False
            ):
                tabs = st.tabs(list(patient_full_data_df.keys()))
                for i, (domain, df) in enumerate(patient_full_data_df.items()):
                    with tabs[i]:
                        st.dataframe(df, use_container_width=True)

            # CONFIRMATION BUTTON
            with col_btn:
                st.write("")  # Spacer
                st.write("")  # Spacer
                if st.button(
                    "✅ Load Patient", type="primary", use_container_width=True
                ):
                    # !!! CRITICAL: CONVERT DF TO DICT FOR AI ENGINE !!!
                    current_patient_dict = {
                        "id": selected_usubjid,
                        "demographics": patient_full_data_df.get(
                            "DM", pd.DataFrame()
                        ).to_dict(orient="records"),
                        "adverse_events": patient_full_data_df.get(
                            "AE", pd.DataFrame()
                        ).to_dict(orient="records"),
                        "lab_results": patient_full_data_df.get(
                            "LB", pd.DataFrame()
                        ).to_dict(orient="records"),
                        "vital_signs": patient_full_data_df.get(
                            "VS", pd.DataFrame()
                        ).to_dict(orient="records"),
                        "medical_history": patient_full_data_df.get(
                            "MH", pd.DataFrame()
                        ).to_dict(orient="records"),
                    }

                    # Store in session state
                    reset_patient_specific_data()
                    st.session_state.current_patient_data = current_patient_dict
                    st.session_state.confirmed_patient = selected_usubjid
                    st.session_state.patient_confirmed = True
                    st.rerun()

    else:
        st.warning("⚠️ No patients match the current filter criteria.")


def display_medical_dashboard():
    patient_data = st.session_state.current_patient_data
    patient_id = st.session_state.confirmed_patient

    # Extract basic info for the "Hero Card"
    demos = patient_data.get("demographics", [{}])[0]
    p_age = demos.get("AGE", "N/A")
    p_sex = demos.get("SEX", "N/A")
    p_race = demos.get("RACE", "N/A")

    # --- TOP NAV BAR ---
    with st.sidebar:
        st.title("🏥 Clinical Assistant")
        st.success(f"**Active ID:** {patient_id}")
        st.markdown("---")
        if st.button("🔙 Switch Patient", use_container_width=True):
            reset_patient_specific_data()
            st.rerun()
        st.info("System Online\nRAG Engine: Connected\nSafety: Active")

    # --- HERO CARD ---
    st.markdown(f"## 🗂️ Patient Dashboard: {patient_id}")

    # Create a nice banner using columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Subject ID", patient_id)
    col2.metric("Age", f"{p_age} Years")
    col3.metric("Sex", p_sex)
    col4.metric("Race", p_race)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(
        ["📄 Clinical Summaries", "💬 AI Consultant", "💊 Prescription Engine"]
    )

    # --- TAB 1: SUMMARIES ---
    with tab1:
        st.markdown("#### Comprehensive Patient Report")

        col_gen, col_display = st.columns([1, 4])

        with col_gen:
            st.write(
                "Generate a full natural language report based on all SDTM domains."
            )
            if st.button("⚡ Generate Report", use_container_width=True):
                with st.spinner("Processing LLM Summaries..."):
                    try:
                        p = get_all_summaries(
                            patient_data["demographics"],
                            patient_data["adverse_events"],
                            patient_data["lab_results"],
                            patient_data["vital_signs"],
                            patient_data["medical_history"],
                        )
                        combined_summary = create_combined_summary(
                            p["demographics_summary"],
                            p["adverse_events_summary"],
                            p["lab_results_summary"],
                            p["vital_signs_summary"],
                            p["medical_history_summary"],
                        )
                        st.session_state["summary_cache"] = combined_summary
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")

        with col_display:
            if "summary_cache" in st.session_state:
                st.success("Report Generated Successfully")
                st.markdown(st.session_state["summary_cache"])
            else:
                st.info("Click 'Generate Report' to analyze patient data.")

    # --- TAB 2: CHAT ASSISTANT ---
    with tab2:
        st.markdown("#### 🤖 Clinical Director Assistant")

        chat_container = st.container(height=500)

        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {
                    "role": "system",
                    "content": "You are an expert Medical Director's Assistant. Analyze patient data provided via tools.",
                }
            ]
            # Initial bot message
            with chat_container:
                st.chat_message("assistant", avatar="🩺").write(
                    f"Hello. I have loaded the records for **{patient_id}**. How can I assist with the clinical review?"
                )

        # Display history
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.chat_message("user", avatar="🧑‍⚕️").write(msg["content"])
                elif msg["role"] == "assistant":
                    st.chat_message("assistant", avatar="🩺").write(msg["content"])
                elif msg["role"] == "tool":
                    with st.expander(f"🔍 Analyzed Data for Request", expanded=False):
                        st.code(msg["content"], language="json")

        # Chat Input
        if prompt := st.chat_input("Ask about medical history, labs, or vitals..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with chat_container:
                st.chat_message("user", avatar="🧑‍⚕️").write(prompt)

                with st.spinner("👩‍⚕️ Analyzing records..."):
                    try:
                        # First Call
                        response = ollama.chat(
                            model="llama3.1",
                            messages=st.session_state.chat_history,
                            tools=my_tools,
                            options={"temperature": 0.0},
                        )

                        # Check for tool calls
                        if response["message"].get("tool_calls"):
                            st.session_state.chat_history.append(response["message"])
                            tool_calls = response["message"]["tool_calls"]

                            for tool in tool_calls:
                                fname = tool["function"]["name"]
                                func = available_functions[fname]
                                result = func()

                                # Add tool result to history
                                st.session_state.chat_history.append(
                                    {"role": "tool", "content": result}
                                )

                                # Visual indicator of tool use
                                with st.expander(
                                    f"🛠️ Tool Used: {fname}", expanded=False
                                ):
                                    st.json(
                                        json.loads(result)
                                        if isinstance(result, str)
                                        else result
                                    )

                            # Second Call (Final Answer)
                            final_response = ollama.chat(
                                model="llama3.1",
                                messages=st.session_state.chat_history,
                                options={"temperature": 0.0},
                            )
                            bot_reply = final_response["message"]["content"]
                            st.session_state.chat_history.append(
                                final_response["message"]
                            )
                        else:
                            bot_reply = response["message"]["content"]
                            st.session_state.chat_history.append(response["message"])

                        st.chat_message("assistant", avatar="🩺").write(bot_reply)

                    except Exception as e:
                        st.error(f"Ollama Error: {e}")

    # --- TAB 3: PRESCRIPTION ENGINE ---
    with tab3:
        st.markdown("#### 🔬 Clinical Decision & Prescription Engine")
        st.caption("Powered by RAG Knowledge Base & Real-time Safety Checks")

        engine = get_clinical_engine()

        col_input, col_process = st.columns([3, 1])
        with col_input:
            user_order = st.text_input(
                "Enter Medical Order:",
                placeholder="e.g., Prescribe Paracetamol for headache",
            )

        with col_process:
            st.write("")  # Spacer
            st.write("")
            process_btn = st.button(
                "🚀 Process Order", use_container_width=True, type="primary"
            )

        if process_btn:
            if not user_order:
                st.warning("Please enter an order.")
            else:
                st.divider()
                st.markdown("##### ⚙️ Engine Workflow")

                # 1. Gatekeeper
                with st.status("Checking Clinical Intent...", expanded=True) as status:
                    gate = engine.gatekeeper_check(user_order)
                    if not gate["is_prescription"]:
                        status.update(
                            label="⛔ Request Rejected by Triage", state="error"
                        )
                        st.error(f"**Gatekeeper Alert:** {gate['reason']}")
                        return
                    status.update(
                        label="✅ Valid Prescription Request", state="complete"
                    )

                # 2. Knowledge Base
                context = ""  # Initialize context to ensure it exists
                score = 0
                with st.status(
                    "Querying Medical Knowledge Base (RAG)...", expanded=True
                ) as status:
                    context, score = engine.retrieve_and_evaluate_context(user_order)
                    if score < 60:
                        status.update(label="❌ Context Low Relevance", state="error")
                        st.error(
                            f"Context Relevance Low ({score}%). Generating Alternative..."
                        )
                        alt = engine.generate_alternative(
                            user_order, "Knowledge Base Relevance < 60%", patient_data
                        )
                        st.markdown(alt)
                        return
                    status.update(
                        label=f"✅ Protocol Found (Match: {score}%)", state="complete"
                    )

                # FIXED: This Expander is now OUTSIDE the st.status block
                with st.expander("View Protocol Evidence"):
                    st.write(context)

                # 3. Safety Check
                with st.status(
                    "Running Patient Safety Checks (Labs & History)...", expanded=True
                ) as status:
                    safety = engine.patient_safety_check(context, patient_data)
                    if safety["status"] == "CONFLICT":
                        status.update(
                            label="🛑 Safety Contraindication Found", state="error"
                        )
                        st.error(f"**SAFETY ALERT:** {safety['reason']}")
                        st.markdown("#### 🔄 Calculating Safe Alternative...")
                        alt = engine.generate_alternative(
                            user_order, safety["reason"], patient_data
                        )
                        st.markdown(alt)
                        return
                    status.update(label="✅ Safety Check Passed", state="complete")

                # 4. Final Authorization
                st.success("✅ Prescription Authorized")
                st.balloons()

                with st.container():
                    st.markdown(
                        f"""
                    <div style="background-color: #d4edda; color: #155724; padding: 15px; border-radius: 5px; border: 1px solid #c3e6cb;">
                        <strong>📡 Order Summary:</strong><br>
                        {user_order}<br><br>
                        <strong>✅ Safety Validation:</strong><br>
                        {safety["reason"]}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )


# ============================================================================
# MAIN APPLICATION
# ============================================================================


def main():
    # Initialize session state for data
    if "all_data" not in st.session_state:
        with st.spinner("🔄 Loading SDTM Clinical Data..."):
            st.session_state.all_data = load_all_data(DATA_PATH)
            if st.session_state.all_data is None:
                st.stop()

    if "patient_confirmed" not in st.session_state:
        st.session_state.patient_confirmed = False

    # Router
    if not st.session_state.patient_confirmed:
        display_patient_search()
    else:
        display_medical_dashboard()


if __name__ == "__main__":
    main()

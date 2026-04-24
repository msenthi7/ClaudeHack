from utils.data_loader import load_sdtm_data
from services.summarizers import get_all_summaries, create_combined_summary
from rich.console import Console
from rich.markdown import Markdown
import ollama
import json
import chromadb
from sentence_transformers import SentenceTransformer
from rich.panel import Panel
from rich.status import Status
from rich.layout import Layout
import time
import sys

# ==========================================
# 1. GLOBAL SETUP & DATA LOADING
# ==========================================
console = Console()
patient_id = "SUBJ02"  # Example USUBJID

console.print(
    Panel(f"[bold blue]Loading Data for Patient: {patient_id}...[/bold blue]")
)

(
    demographics_list,
    adverse_events_list,
    lab_results_list,
    vital_signs_list,
    medical_history_list,
) = load_sdtm_data(patient_id)

# Print Raw Data Check (As per your original code)
print("-" * 100)
print(demographics_list)
print("-" * 100)
print(adverse_events_list)
print("-" * 100)
print(lab_results_list)
print("-" * 100)
print(vital_signs_list)
print("-" * 100)
print(medical_history_list)
print("-" * 100)

# Global Patient Object for the Engines
CURRENT_PATIENT = {
    "id": patient_id,
    "demographics": demographics_list,
    "adverse_events": adverse_events_list,
    "lab_results": lab_results_list,
    "vital_signs": vital_signs_list,
    "medical_history": medical_history_list,
}

# Configuration for Vector DB
DB_PATH = r"D:\ACL Project\sdtm-data-process-ai\db_creator\db_creator"
COLLECTION_NAME = "medical_knowledge"
MODEL_NAME = "all-MiniLM-L6-v2"


# ==========================================
# 2. MODE A: SUMMARIZATION GENERATOR
# ==========================================
def run_summary_mode():
    console.print("[bold magenta]Generating Clinical Summaries...[/bold magenta]")

    with console.status("[bold magenta]Processing LLM Summaries...[/bold magenta]"):
        p = get_all_summaries(
            demographics_list,
            adverse_events_list,
            lab_results_list,
            vital_signs_list,
            medical_history_list,
        )

        # You can uncomment these if you want individual prints
        # print(p["demographics_summary"])
        # print(p["adverse_events_summary"])
        # print(p["lab_results_summary"])

        combined_summary = create_combined_summary(
            p["demographics_summary"],
            p["adverse_events_summary"],
            p["lab_results_summary"],
            p["vital_signs_summary"],
            p["medical_history_summary"],
        )

    console.print(
        Panel(Markdown(combined_summary), title="Comprehensive Patient Report")
    )
    input("\nPress Enter to return to Main Menu...")


# ==========================================
# 3. MODE B: CHAT ASSISTANT (WITH TOOLS)
# ==========================================
# --- Tool Definitions ---
def get_full_patient_summary():
    return json.dumps(CURRENT_PATIENT)


def get_patient_demographics():
    return json.dumps(demographics_list)


def get_adverse_events():
    return json.dumps(adverse_events_list)


def get_lab_results():
    return json.dumps(lab_results_list)


def get_vital_signs():
    return json.dumps(vital_signs_list)


def get_medical_history():
    return json.dumps(medical_history_list)


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


def run_chat_mode():
    system_prompt = """
    You are an expert Medical Director's Assistant.
    Your goal is to analyze patient data and provide medical insights.

    CRITICAL RULES FOR TOOL USAGE:
    1. **General Questions:** If the user asks "Tell me about the patient", call `get_full_patient_summary`.
    2. **Specific Data:** If the user asks for specific values (e.g., "What is the age?", "What is the Lab Result?"), call the specific tool.
    3. **Definitions/Explanations:** If the user asks "What is AST?" or "Explain this symptom", DO NOT CALL A TOOL. Use your internal medical knowledge to explain the term.
    4. **Chatting:** Do not call tools for "Hello", "Thank you", or "Goodbye".

    TONE:
    - Be concise, professional, and clinical.
    - Do not repeat data unnecessarily.
    """

    messages = [{"role": "system", "content": system_prompt}]
    console.print(
        Panel(
            "[bold blue]🩺 CLINICAL DIRECTOR ASSISTANT (Chat Mode)[/bold blue]\nType 'back' to return to menu.",
            style="blue",
        )
    )

    while True:
        try:
            user_input = input("\nMedical Director (Chat): ")
            if user_input.lower() in ["back", "exit", "quit"]:
                break

            messages.append({"role": "user", "content": user_input})

            # FIRST CALL
            response = ollama.chat(
                model="llama3.1",
                messages=messages,
                tools=my_tools,
                options={"temperature": 0.0},
            )

            # CHECK FOR TOOL CALLS
            if response["message"].get("tool_calls"):
                tool_calls = response["message"]["tool_calls"]
                console.print(
                    f"[dim italic]   -> Querying Clinical Database... ({len(tool_calls)} request)[/dim italic]"
                )

                messages.append(response["message"])

                for tool in tool_calls:
                    fname = tool["function"]["name"]
                    func = available_functions[fname]
                    result = func()
                    messages.append({"role": "tool", "content": result})

                # SECOND CALL (Final Answer)
                final_response = ollama.chat(
                    model="llama3.1", messages=messages, options={"temperature": 0.0}
                )

                console.print(
                    f"\n[bold blue]AI Assistant:[/bold blue] {final_response['message']['content']}"
                )
                messages.append(final_response["message"])

            else:
                # NO TOOL NEEDED
                console.print(
                    f"\n[bold blue]AI Assistant:[/bold blue] {response['message']['content']}"
                )
                messages.append(response["message"])

        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")


# ==========================================
# 4. MODE C: PRESCRIPTION ENGINE
# ==========================================
class ClinicalDecisionEngine:
    def __init__(self):
        console.print(
            Panel(
                "[bold white]Initializing Clinical Workflow Engine...[/bold white]",
                style="green",
            )
        )
        try:
            self.embed_model = SentenceTransformer(MODEL_NAME)
            self.db_client = chromadb.PersistentClient(path=DB_PATH)
            self.collection = self.db_client.get_collection(COLLECTION_NAME)
            console.print("[dim]✅ Medical Knowledge Base Connected[/dim]")
        except Exception as e:
            console.print(f"[bold red]Startup Error:[/bold red] {e}")

    def parse_json_safe(self, text_output):
        try:
            clean_text = text_output.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except json.JSONDecodeError:
            console.print(
                f"[bold red]JSON Parse Warning.[/bold red] Raw Output: {text_output[:50]}..."
            )
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
        if not results["documents"][0]:
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

    def run(self, user_input):
        print("\n" + "─" * 60)
        with console.status(
            "[bold white]1. Checking Prescription Intent...[/bold white]"
        ):
            gate = self.gatekeeper_check(user_input)

        if not gate["is_prescription"]:
            console.print(
                f"[yellow]⛔ Gatekeeper:[/yellow] Input rejected. {gate['reason']}"
            )
            return

        console.print("[green]✅ Valid Prescription Request.[/green]")

        with console.status("[bold cyan]2. Analyzing Knowledge Base...[/bold cyan]"):
            context, score = self.retrieve_and_evaluate_context(user_input)

        if score < 60:
            console.print(f"[bold red]❌ Context Relevance Low ({score}%).[/bold red]")
            with console.status(
                "[magenta]🔄 Generating Alternative Solution...[/magenta]"
            ):
                alt = self.generate_alternative(
                    user_input, "Knowledge Base Relevance < 60%", CURRENT_PATIENT
                )
            console.print(
                Panel(
                    Markdown(alt), title="Alternative Care Plan", border_style="magenta"
                )
            )
            return

        console.print(
            f"[green]✅ Database Context Matched ({score}% Relevance).[/green]"
        )

        with console.status(
            "[bold yellow]3. Evaluating Patient Safety (Labs & History)...[/bold yellow]"
        ):
            safety = self.patient_safety_check(context, CURRENT_PATIENT)

        if safety["status"] == "CONFLICT":
            console.print(f"[bold red]🛑 SAFETY ALERT: {safety['reason']}[/bold red]")
            with console.status(
                "[magenta]🔄 Calculating Safe Alternative...[/magenta]"
            ):
                alt = self.generate_alternative(
                    user_input, safety["reason"], CURRENT_PATIENT
                )
            console.print(
                Panel(
                    Markdown(alt),
                    title="Safety Intervention & Alternative",
                    border_style="red",
                )
            )
            return

        console.print("[bold green]✅ Safety Check Passed.[/bold green]")
        final_output = f"""### ⚕️ Prescription Authorized\n**Request:** {user_input}\n**Protocol Evidence:** {context[:200]}...\n**Safety Validation:** {safety["reason"]}"""
        console.print(
            Panel(
                Markdown(final_output),
                title="Final Medical Order",
                border_style="green",
            )
        )


def run_prescription_mode():
    engine = ClinicalDecisionEngine()
    console.print(
        Panel(
            "[bold green]Prescription Engine Active[/bold green]\nType 'back' to return to menu.",
            style="green",
        )
    )
    while True:
        try:
            user_in = input("\n[Medical Director] Enter Order: ")
            if user_in.lower() in ["back", "exit", "quit"]:
                break
            engine.run(user_in)
        except Exception as e:
            console.print(f"[bold red]Runtime Error:[/bold red] {e}")


# ==========================================
# 5. MAIN APPLICATION LOOP
# ==========================================
def main_menu():
    while True:
        console.clear()
        console.print(
            Panel.fit(
                "[bold cyan]🏥 PATIENT DATA SYSTEM[/bold cyan]\n\n"
                "1. [bold magenta]📄 View Combined Summary[/bold magenta]\n"
                "2. [bold blue]💬 Chat with Patient Data[/bold blue]\n"
                "3. [bold green]💊 Prescription Engine[/bold green]\n"
                "4. ❌ Exit",
                title=f"User: {patient_id}",
                border_style="white",
            )
        )

        choice = input("Select an option (1-4): ")

        if choice == "1":
            run_summary_mode()
        elif choice == "2":
            run_chat_mode()
        elif choice == "3":
            run_prescription_mode()
        elif choice == "4":
            console.print("[bold]Exiting System. Goodbye.[/bold]")
            sys.exit()
        else:
            input("Invalid choice. Press Enter to try again...")


if __name__ == "__main__":
    # Data is already loaded at global scope
    main_menu()

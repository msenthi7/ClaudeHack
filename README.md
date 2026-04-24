# ClinIQ — AI Triage Assistant for Underserved Clinics

> **Anthropic Hackathon — Track #1: Health and Wellbeing**

ClinIQ helps doctors at under-resourced clinics instantly prioritize patients by severity, get AI-generated clinical summaries, and safely check prescriptions — all powered by a local LLM grounded strictly in a medical knowledge base.

---

## Devpost Submission

**Project Name:** ClinIQ — AI Triage Assistant for Underserved Clinics

**Tagline:** Instant patient triage and prescription safety for clinics that can't afford to wait.

**Track:** Health and Wellbeing

**GitHub:** https://github.com/msenthi7/ClaudeHack

---

### Who we built this for and why they need it

Doctors at underserved and under-resourced clinics see dozens of patients a day with no specialists to consult, no clinical decision support tools, and no time. They must decide — often within minutes — who needs attention first, whether a drug is safe for a specific patient, and how to explain a diagnosis in plain language to a patient who may not be health-literate.

ClinIQ is built for those doctors. It gives them an AI-powered triage layer that automatically ranks patients by clinical severity, surfaces dangerous lab values and vital sign anomalies instantly, and checks prescription safety against a real medical knowledge base — all running locally, with no subscription fees or internet dependency for core features.

---

### How we used AI in this project

AI is used across four core workflows:

1. **Triage scoring** — A rule-based engine scores every patient across adverse events, lab results, vital signs, and chronic condition burden. This produces an objective severity ranking updated in real time.

2. **Clinical summaries** — A local LLM (llama3.1 via Ollama) generates structured, physician-facing narrative reports by summarizing all five SDTM data domains in parallel using a thread pool.

3. **Plain-language patient summaries** — The same LLM rewrites clinical findings in simple, jargon-free language directed at the patient, so they can understand their own health.

4. **Prescription safety engine (RAG pipeline)** — When a doctor enters a drug order, the system:
   - Checks whether it is a valid prescription request (gatekeeper)
   - Retrieves relevant drug protocols from a ChromaDB knowledge base seeded with the Gale Encyclopedia of Medicine and real FDA drug labels
   - Runs a patient-specific safety check against their labs, vitals, and adverse events
   - Flags contraindications and generates a safe alternative if needed

5. **Strict RAG chat** — The AI chat assistant retrieves relevant context from the medical knowledge base before every response and is instructed to answer only from that context — preventing hallucinated medical advice.

---

### What could go wrong and how we addressed it

| Risk | Mitigation |
|---|---|
| LLM hallucinating medical facts | Strict RAG: every AI answer is grounded in retrieved KB chunks. System prompt explicitly forbids outside knowledge. |
| Prescription engine authorizing unsafe drugs | 3-step pipeline with a dedicated patient safety check against actual lab values before any order is authorized |
| JSON parsing failures from LLM | `_safe_json()` wrapper with fallback defaults on every LLM call that expects structured output |
| Knowledge base not loaded | App checks KB availability at startup; AI buttons are disabled with a clear message if KB is missing |
| Wrong patient data shown after switching patients | All patient-specific session state (summaries, chat, plain summary) is cleared on patient switch |
| ChromaDB hashing errors with Streamlit cache | Used `_dfs` underscore prefix to skip unhashable DataFrame dict from Streamlit's cache hash |

---

### What we'd build next with more time

1. **Claude API integration** — Replace local Ollama with Claude claude-sonnet-4-6 for significantly better clinical reasoning and faster responses
2. **Real patient data pipeline** — Connect to MIMIC-III or HL7 FHIR feeds for real de-identified clinical data
3. **Nurse/patient portal** — A separate view where patients can read their own summary and ask questions in their language
4. **Alert notifications** — Push urgent triage alerts to the doctor's phone when a HIGH-priority patient is checked in
5. **Expanded knowledge base** — Ingest clinical guidelines (WHO, CDC, NICE) and drug interaction databases (DrugBank, RxNorm)
6. **Audit trail** — Log every AI recommendation with the knowledge base sources used, for clinical accountability

---

## The Problem

Healthcare is expensive, inaccessible, and confusing. Medical knowledge is locked behind jargon and paywalls. Billions lack quality care. Doctors at underserved clinics must make critical decisions alone, without specialists, with limited time per patient.

## The Solution

ClinIQ provides:
- **Automatic triage** — ranks all patients by risk score so the most critical cases get seen first
- **AI clinical summaries** — instant, structured reports for physicians
- **Plain-language patient summaries** — translates medical data into words patients can understand
- **Prescription safety engine** — checks drug orders against patient labs, vitals, and a real medical knowledge base before authorizing

---

## Features

### 🔴🟡🟢 Triage Dashboard
- All patients ranked by AI-calculated severity score
- Risk factors flagged: abnormal labs, high blood pressure, SEVERE adverse events, chronic conditions
- Search by name or ID
- Filter by priority level (High / Moderate / Low)

### 🚨 Triage Alerts (per patient)
- Specific clinical flags with values and normal ranges
- Lab results, vital signs, and adverse event tables side by side

### 📄 Clinical Report
- AI-generated narrative summary for the treating physician
- Covers demographics, labs, vitals, adverse events, and medical history

### 🗣️ Patient Summary
- Plain-language explanation of the patient's health — no jargon
- Written directly to the patient in second person

### 💊 Prescription Safety Engine
- 3-step pipeline: Intent check → Knowledge base retrieval → Patient safety check
- Grounded in real FDA drug label data and the Gale Encyclopedia of Medicine
- Flags contraindications and suggests safe alternatives

### 💬 AI Chat Assistant
- Ask anything about the patient
- Strictly answers from the medical knowledge base — no hallucinated medical facts
- Uses tools to pull patient-specific data (labs, vitals, history, etc.)

---

## Knowledge Base

Two sources, stored locally in ChromaDB:
1. **Gale Encyclopedia of Medicine** (637 pages) — primary medical reference
2. **openFDA drug labels** — warnings, contraindications, interactions for 12 common drugs

All AI answers are grounded in this knowledge base. The model will not use outside knowledge to answer medical questions.

---

## Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| LLM | Ollama (llama3.1) — runs locally |
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Vector DB | ChromaDB (local, persistent) |
| Data models | Pydantic |
| Patient data | SDTM-format CSV files |
| Drug data | openFDA API |

---

## Project Structure

```
ClinIQ/
├── data/                   # SDTM patient data (CSV)
│   ├── dm.csv              # Demographics
│   ├── ae.csv              # Adverse Events
│   ├── lb.csv              # Lab Results
│   ├── vs.csv              # Vital Signs
│   └── mh.csv              # Medical History
├── db/                     # ChromaDB vector store (auto-created)
├── models/
│   └── sdtm_models.py      # Pydantic data models
├── services/
│   ├── triage.py           # Severity scoring engine
│   ├── summarizers.py      # LLM summarization functions
│   └── ollama_client.py    # Ollama wrapper
├── utils/
│   └── data_loader.py      # SDTM CSV loader
├── scripts/
│   └── seed_kb.py          # Seeds ChromaDB from PDF + openFDA
├── streamlit_ui.py         # Main Streamlit application
└── main2.py                # CLI version
```

---

## Setup & Running

### Prerequisites
- Python 3.12+
- [Ollama](https://ollama.com) installed and running

### 1. Install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install chromadb ollama pandas pydantic rich sentence-transformers streamlit requests pymupdf
```

### 2. Start Ollama and pull the model
```bash
ollama serve          # in a separate terminal
ollama pull llama3.1
```

### 3. Seed the knowledge base (run once)
Place `Medical_book.pdf` in `~/Downloads/` then:
```bash
python3 scripts/seed_kb.py
```

### 4. Launch the app
```bash
streamlit run streamlit_ui.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## Triage Scoring

| Signal | Score |
|---|---|
| SEVERE adverse event | +4 |
| MODERATE adverse event | +2 |
| MILD adverse event | +1 |
| Abnormal lab value (AST, ALT, CRP) | +2 each |
| Abnormal vital sign (BP, HR) | +2 each |
| Chronic condition (Hypertension, Diabetes, Asthma) | +1 each |

| Score | Priority |
|---|---|
| 7+ | 🔴 HIGH |
| 3–6 | 🟡 MODERATE |
| 0–2 | 🟢 LOW |

---

## Built With

- [Streamlit](https://streamlit.io)
- [Ollama](https://ollama.com)
- [ChromaDB](https://www.trychroma.com)
- [SentenceTransformers](https://www.sbert.net)
- [openFDA](https://open.fda.gov)

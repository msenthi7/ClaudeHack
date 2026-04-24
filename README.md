# ClinIQ — AI Triage Assistant for Underserved Clinics

> **Anthropic Hackathon — Track #1: Health and Wellbeing**

ClinIQ helps doctors at under-resourced clinics instantly prioritize patients by severity, get AI-generated clinical summaries, and safely check prescriptions — all powered by a local LLM grounded strictly in a medical knowledge base.

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
- Grounded in real FDA drug label data and the medical knowledge base
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

Patients are scored automatically based on:

| Signal | Score |
|---|---|
| SEVERE adverse event | +4 |
| MODERATE adverse event | +2 |
| MILD adverse event | +1 |
| Abnormal lab value (AST, ALT, CRP) | +2 each |
| Abnormal vital sign (BP, HR) | +2 each |completely |
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

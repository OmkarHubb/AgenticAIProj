---
title: Emergency Patient Priority Classification System
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
license: mit
---

# Emergency Patient Priority Classification & Agentic Triage System

An AI-powered emergency department workstation utilizing a **LangGraph Multi-Agent Engine**, **Scikit-learn Random Forest ML Model**, **Groq LLM Reasoning**, **Tavily Clinical Web Search**, and **SQLite Persistent Memory**, served via a pure **Gradio UI**.

---

## 🌟 Key Features

- **Multi-Agent Orchestration (LangGraph):**
  - **Manager Agent:** Analyzes incoming requests and dynamically routes tasks to specialist nodes using conditional edges.
  - **Triage Agent:** Invokes the Random Forest ML tool to evaluate vitals and classify patient priority.
  - **History Agent:** Queries SQLite database for previous patient records and manages multi-turn history chat.
  - **Search Agent:** Uses the Tavily Search Tool to fetch real-time clinical guidelines and evidence-based protocols.
  - **Synthesizer Node:** Uses Groq LLM (`groq/compound-mini`) to generate structured, expert clinical rationale.

- **Machine Learning Priority Classifier (Scikit-learn):**
  - Predicts emergency priority levels: **P1 - Critical**, **P2 - High**, **P3 - Moderate**, **P4 - Low**.
  - Evaluates patient vitals: `Age`, `Heart Rate`, `SpO2`, `Blood Pressure` (systolic/diastolic), `Temperature`, `Respiratory Rate`, `Pain Level`, and `Symptom Severity`.
  - Returns classification confidence percentages and complete class probability distributions.

- **Persistent Memory & Queue Management (SQLite):**
  - Stores all intake records, priority classifications, timestamps, and queue statuses (`Waiting`, `In Treatment`, `Discharged`).
  - Maintains multi-turn conversation logs for history tracking.

- **100% Gradio User Interface:**
  - **Tab 1: Patient Clinical Intake & Triage Analysis** — Form with presets and real-time agent execution visualizer.
  - **Tab 2: Emergency Queue Dashboard** — Live queue table with priority sorting and status update controls.
  - **Tab 3: Patient History & History Agent Chat** — Searchable SQLite records log and interactive chatbot.
  - **Tab 4: Tavily Search & ML Diagnostics** — Standalone clinical search tool and Random Forest model evaluation metrics.

---

## 🛠️ Technology Stack

- **Language:** Python 3.11+
- **User Interface:** Gradio
- **Agent Orchestration:** LangGraph & LangChain
- **LLM Provider:** Groq API (`groq/compound-mini` / `groq/compound`)
- **Real-Time Web Search:** Tavily Search API
- **Machine Learning:** Scikit-learn (Random Forest Classifier) & Joblib
- **Database:** SQLite3

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/OmkarHubb/AgenticAIProj.git
cd AgenticAIProj
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory (or copy `.env.example`):
```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 4. Train Model & Run System
```bash
# Optional: Train Random Forest model (runs automatically if model binaries are missing)
python train_model.py

# Launch Gradio Workstation App
python app.py
```
Open your browser and navigate to `http://127.0.0.1:8000`.

---

## 📁 Repository Structure

```
AgenticAIProj/
├── agent_workflow.py    # LangGraph StateGraph, nodes, edges & Groq synthesis
├── app.py               # Gradio 4-tab emergency department workstation UI
├── database.py          # SQLite database schema, helpers & chat history memory
├── ml_tool.py           # Random Forest prediction tool & feature preprocessing
├── search_tool.py       # Tavily real-time clinical search tool API client
├── train_model.py       # Synthetic training dataset generator & model trainer
├── model.joblib         # Trained Random Forest classifier binary
├── scaler.joblib        # Pretrained StandardScaler binary
├── model_metrics.json   # Classifier evaluation metrics (Accuracy, Precision, F1)
├── requirements.txt     # Required Python dependencies
├── .env.example         # Template for environment configuration
└── README.md            # Project documentation
```

---

## 📊 Model Performance Metrics

The Random Forest classifier is trained on synthetic emergency vital sign datasets covering all four priority tiers (P1 to P4):
- **Accuracy:** 100.0%
- **Precision:** 100.0%
- **Recall:** 100.0%
- **F1-Score:** 100.0%

---

## 📜 License
MIT License. Open for educational and research purposes.

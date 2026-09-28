# Emergency Patient Priority Classification & Agentic Triage System — Project Documentation

---

## 1. Overview & Decoupled Architecture

### 1.1 Problem Statement

Emergency departments face a critical bottleneck: manual patient triage is subjective, inconsistent, and prone to human error under high-volume conditions. Mis-prioritisation delays treatment for critical patients and wastes resources on low-acuity cases. Existing scoring systems (ESI, MTS) rely on clinician judgement without quantitative ML-backed classification or agentic reasoning.

### 1.2 Clinical Gaps Addressed

| Gap | Solution |
|-----|----------|
| Subjective vital sign interpretation | Random Forest classifier trained on 9 numerical vital features for deterministic P1–P4 classification |
| No structured clinical rationale | Groq LLM generates 2-sentence medical rationale contextualising the ML classification |
| Siloed patient data | SQLite persistent memory with automated queue ranking and full patient history search |
| No clinical reference lookup | Tavily Search API integration for real-time emergency medicine guideline retrieval |
| Lack of workflow transparency | LangGraph execution trace log exposes every agent decision for auditability |

### 1.3 Architectural Decoupling: ML Classification vs LLM Rationale

The system intentionally separates the **classification decision** from the **clinical explanation** into two independent subsystems:

```
┌──────────────────────────────────┐     ┌──────────────────────────────────┐
│   RANDOM FOREST ML CLASSIFIER   │     │     GROQ LLM RATIONALE ENGINE   │
├──────────────────────────────────┤     ├──────────────────────────────────┤
│ • Deterministic priority output  │     │ • Natural language explanation   │
│ • 9-feature numerical pipeline   │────►│ • Contextualises ML output       │
│ • StandardScaler normalisation   │     │ • 2-sentence clinical rationale  │
│ • Class probability distribution │     │ • Graceful fallback if API fails │
│ • Offline inference (no API)     │     │ • Requires GROQ_API_KEY          │
└──────────────────────────────────┘     └──────────────────────────────────┘
         ALWAYS RUNS                              BEST-EFFORT
```

**Why this split matters:**
- The ML model provides the **authoritative classification** — it runs offline, is reproducible, and never hallucinates a priority code.
- The LLM provides the **human-readable rationale** — it enriches the output but is non-blocking. If the Groq API is unavailable, the system falls back to a static description without degrading classification accuracy.

---

## 2. LangGraph Multi-Agent Engine

### 2.1 WorkflowState Schema

Defined in `agent_workflow.py` as a `TypedDict`:

```python
class WorkflowState(TypedDict):
    input_data: Dict[str, Any]        # Raw patient vitals or query payload
    query_type: str                   # "triage" | "history" | "search"
    triage_result: Optional[Dict]     # ML classification output
    history_result: Optional[List]    # SQLite patient records
    search_result: Optional[str]      # Tavily search output
    agent_logs: List[str]             # Execution trace log entries
    final_output: Dict[str, Any]      # Synthesised response payload
```

### 2.2 Agent Nodes (5 Nodes)

| Node | File Location | Role | Key Operations |
|------|--------------|------|----------------|
| **Manager** | `agent_workflow.py:manager_node()` | Intent classifier & router | Inspects `task_type`; infers intent from input keys if missing; appends routing decision to `agent_logs` |
| **Triage** | `agent_workflow.py:triage_agent_node()` | ML classification executor | Calls `predict_priority()` from `ml_tool.py`; saves patient to SQLite via `database.save_patient()`; returns `triage_result` |
| **History** | `agent_workflow.py:history_agent_node()` | Patient record retriever | Queries `database.get_patient_history()` by name or ID; returns matching records |
| **Search** | `agent_workflow.py:search_agent_node()` | Clinical reference lookup | Invokes `tavily_search()` from `search_tool.py`; returns formatted medical references |
| **Synthesizer** | `agent_workflow.py:synthesizer_node()` | LLM rationale & final assembly | Calls Groq LLM for clinical rationale; assembles `final_output` payload with priority, probabilities, trace, and analysis |

### 2.3 Conditional Edge Routing

The Manager node uses conditional edges to route to the appropriate specialist:

```python
def route_manager(state: WorkflowState):
    qtype = state.get('query_type', 'triage')
    if qtype == 'history':
        return "history"
    elif qtype == 'search':
        return "search"
    else:
        return "triage"
```

### 2.4 Architecture Flow Diagram

```
                    ┌─────────┐
                    │  START  │
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │ MANAGER │  ← Intent classification
                    └────┬────┘
                         │
            ┌────────────┼────────────┐
            │            │            │
    ┌───────▼──────┐ ┌───▼───┐ ┌─────▼─────┐
    │  TRIAGE      │ │HISTORY│ │  SEARCH   │
    │  Agent       │ │ Agent │ │  Agent    │
    │ (ML Tool)    │ │(SQLite)│ │ (Tavily)  │
    └───────┬──────┘ └───┬───┘ └─────┬─────┘
            │            │           │
            └────────────┼───────────┘
                         │
                  ┌──────▼──────┐
                  │ SYNTHESIZER │  ← Groq LLM rationale
                  └──────┬──────┘
                         │
                    ┌────▼────┐
                    │   END   │
                    └─────────┘
```

### 2.5 Groq LLM Configuration

```python
def get_groq_llm():
    # Model fallback chain:
    # 1. groq/compound-mini
    # 2. groq/compound
    # 3. llama-3.3-70b-versatile
    # 4. llama3-8b-8192
    # Falls back to static text if no API key or all models fail
```

- **Temperature:** `0.2` (low variance for clinical consistency)
- **Prompt template:** `"Patient vitals: {vitals}. Priority: {label} ({confidence}% confidence). Symptoms: {symptoms}. Provide a 2-sentence medical rationale."`
- **Graceful degradation:** Returns `"Random Forest ML model evaluated vital signs against emergency triage standards."` on failure

---

## 3. ML Pipeline & Vitals

### 3.1 Feature Vector (9 Numerical Features)

| # | Feature | Type | Source | P1 Critical Range | P4 Low Range |
|---|---------|------|--------|-------------------|--------------|
| 1 | `age` | int | User input | 1–90 | 1–90 |
| 2 | `heart_rate` | int | User input (bpm) | 40–55 or 135–185 | 60–90 |
| 3 | `spo2` | int | User input (%) | 70–88 | 97–100 |
| 4 | `bp_sys` | int | Parsed from BP string | 60–85 or 175–210 | 100–130 |
| 5 | `bp_dia` | int | Parsed from BP string | 40–115 | 60–85 |
| 6 | `temperature` | float | User input (°C) | 34.5–41.0 | 36.0–37.4 |
| 7 | `respiratory_rate` | int | User input (bpm) | 8–11 or 30–45 | 12–18 |
| 8 | `pain_level` | int | User input (1–10) | 8–10 | 1–3 |
| 9 | `symptom_severity` | int | NLP keyword extraction | 4 | 1 |

### 3.2 Symptom Severity NLP Estimation

`ml_tool.py:estimate_symptom_severity()` maps free-text symptoms to a 1–4 integer via keyword matching:

| Severity | Score | Trigger Keywords |
|----------|-------|-----------------|
| Critical | 4 | chest pain, cardiac, unresponsive, stroke, severe bleeding, seizure, respiratory arrest, cyanosis, anaphylaxis |
| Severe | 3 | high fever, fracture, shortness of breath, severe abdominal pain, confusion, asthma attack |
| Moderate | 2 | moderate fever, vomiting, dizziness, sprain, deep cut, persistent cough, migraine |
| Low | 1 | All other / pain < 4 |

### 3.3 Training Pipeline (`train_model.py`)

| Parameter | Value |
|-----------|-------|
| **Synthetic dataset size** | 1,200 clinical records |
| **Class distribution** | P1: 20%, P2: 25%, P3: 30%, P4: 25% |
| **Train/test split** | 80/20 stratified (`random_state=42`) |
| **Scaler** | `StandardScaler` (z-score normalisation) |
| **Classifier** | `RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)` |
| **Serialisation** | `model.joblib` (128 KB), `scaler.joblib` (1.1 KB) |

### 3.4 Evaluation Metrics

| Metric | Macro Avg |
|--------|-----------|
| Accuracy | 100.0% |
| Precision | 100.0% |
| Recall | 100.0% |
| F1-Score | 100.0% |

**Per-class breakdown (test set = 240 samples):**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| P1 – Critical | 100.0% | 100.0% | 100.0% | 51 |
| P2 – High | 100.0% | 100.0% | 100.0% | 59 |
| P3 – Moderate | 100.0% | 100.0% | 100.0% | 75 |
| P4 – Low | 100.0% | 100.0% | 100.0% | 55 |

> **Note:** Perfect metrics result from well-separated synthetic vital sign ranges. Real-world deployment would require clinical dataset validation.

### 3.5 Blood Pressure Parsing

`ml_tool.py:parse_blood_pressure()` converts the user-entered string `"120/80"` into two integers (`bp_sys=120`, `bp_dia=80`). Falls back to `(120, 80)` on parse failure.

---

## 4. SQLite Persistence & Priority Queue

### 4.1 Database Schema

**`patients` table:**

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | Internal row ID |
| `patient_id` | TEXT | UNIQUE NOT NULL | Format: `PAT-XXXX` (auto-generated) |
| `name` | TEXT | NOT NULL | Patient full name |
| `age` | INTEGER | NOT NULL | Patient age |
| `heart_rate` | INTEGER | NOT NULL | Heart rate in bpm |
| `spo2` | INTEGER | NOT NULL | Oxygen saturation % |
| `blood_pressure` | TEXT | NOT NULL | Format: `"sys/dia"` |
| `temperature` | REAL | NOT NULL | Body temperature °C |
| `respiratory_rate` | INTEGER | NOT NULL | Breaths per minute |
| `pain_level` | INTEGER | NOT NULL | Self-reported 1–10 |
| `symptoms` | TEXT | NOT NULL | Free-text clinical notes |
| `priority` | TEXT | NOT NULL | `P1` / `P2` / `P3` / `P4` |
| `confidence` | REAL | NOT NULL | ML confidence % |
| `queue_status` | TEXT | NOT NULL DEFAULT 'Waiting' | `Waiting` / `In Treatment` / `Discharged` |
| `timestamp` | DATETIME | DEFAULT CURRENT_TIMESTAMP | Triage entry time |

**`history_chat` table:**

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT |
| `session_id` | TEXT | Chat session identifier |
| `role` | TEXT | `"user"` or `"assistant"` |
| `content` | TEXT | Message content |
| `timestamp` | DATETIME | Message timestamp |

### 4.2 Patient ID Generation

```python
def generate_patient_id():
    count = SELECT COUNT(*) FROM patients
    return f"PAT-{count + 1001}"  # First patient = PAT-1001
```

### 4.3 Priority Queue SQL — Custom CASE Sorting

The emergency queue is ordered by clinical urgency using a SQL `CASE` expression:

```sql
SELECT *,
    CASE priority
        WHEN 'P1' THEN 1
        WHEN 'P2' THEN 2
        WHEN 'P3' THEN 3
        WHEN 'P4' THEN 4
        ELSE 5
    END AS priority_rank
FROM patients
WHERE queue_status IN ('Waiting', 'In Treatment')
ORDER BY priority_rank ASC, timestamp ASC
```

This guarantees P1 patients always appear before P2, P2 before P3, etc. Within the same priority level, FIFO ordering is maintained via `timestamp ASC`.

### 4.4 Dashboard Summary Queries

```python
def get_dashboard_summary():
    waiting_count  = COUNT(*) WHERE queue_status = 'Waiting'
    p1_count       = COUNT(*) WHERE priority = 'P1' AND queue_status = 'Waiting'
    p2_count       = COUNT(*) WHERE priority = 'P2' AND queue_status = 'Waiting'
    total_triaged  = COUNT(*) FROM patients
```

---

## 5. Gradio UI & Workstation Workflows

### 5.1 Three-Tab Layout Design

| Tab | Title | Components |
|-----|-------|------------|
| **Tab 1** | Patient Intake & Triage | Demographic & vitals form (9 fields), 3 preset buttons (P1/P2/P4), colour-coded priority badge, Groq LLM rationale output, LangGraph execution trace log |
| **Tab 2** | Emergency Queue Dashboard | 4 summary metric cards (Waiting, P1, P2, Total), urgency-sorted dataframe, status update controls (Waiting → In Treatment → Discharged), queue refresh |
| **Tab 3** | Patient Records & Guidelines Search | SQLite record search by name/ID, patient history dataframe, Tavily medical guideline search with formatted output |

### 5.2 Event Handler Map

| UI Action | Handler Function | Inputs | Outputs |
|-----------|-----------------|--------|---------|
| Click "Run Triage Workflow" | `handle_triage()` | 9 vital sign fields | Priority badge HTML, LLM rationale, trace log, queue table, summary cards |
| Click "Preset: P1 Critical" | `load_preset_p1()` | — | Pre-fills all 9 fields with P1 critical vitals |
| Click "Preset: P2 High" | `load_preset_p2()` | — | Pre-fills all 9 fields with P2 high vitals |
| Click "Preset: P4 Low" | `load_preset_p4()` | — | Pre-fills all 9 fields with P4 low vitals |
| Click "Update Status" | `handle_update_status()` | Patient ID, new status | Status message, refreshed queue table, summary cards |
| Click "Refresh Queue" | `get_queue_table()` | — | Refreshed queue dataframe |
| Click "Search Records" | `get_history_table()` | Search query | Filtered patient records dataframe |
| Click "Search Guidelines" | `handle_tavily_search()` | Query string | Tavily formatted output |

### 5.3 Triage Execution Trace Example

When a P1 Critical patient is submitted, the trace log shows:

```
> Manager Agent: Received request. Analyzing workflow intent.
> Manager Agent: Delegating workflow to Specialist Agent node -> TRIAGE.
> Triage Agent: Processing patient vitals. Invoking Random Forest ML tool predict_priority().
> Triage Tool predict_priority() classified priority: P1 - Critical (100.0% confidence).
> Triage Agent: Saved patient PAT-1001 to SQLite database queue.
> Manager Agent: Synthesizing final structured response.
> Manager Agent: Groq LLM clinical rationale generated.
```

### 5.4 Colour-Coded Priority Badge System

| Priority | Badge Colour | Background Tint | Clinical Meaning |
|----------|-------------|-----------------|------------------|
| P1 | `#dc2626` (Crimson Red) | `#fef2f2` | Immediate resuscitation required |
| P2 | `#ea580c` (Medical Orange) | `#fff7ed` | Emergent — treat within 10 min |
| P3 | `#d97706` (Amber Yellow) | `#fefce8` | Urgent — treat within 30 min |
| P4 | `#059669` (Clinical Green) | `#ecfdf5` | Non-urgent — standard queue |

---

## 6. Codebase Reference — File-by-File Function Map

### `app.py` — Gradio UI & Event Handlers

| Function | Lines | Inputs | Returns | Dependencies |
|----------|-------|--------|---------|-------------|
| `handle_triage()` | 16–55 | 9 vital sign form fields | Badge HTML, LLM text, trace, queue DF, summary HTML | `agent_workflow.run_triage_workflow()` |
| `load_preset_p1()` | 58–59 | — | Tuple of 9 P1 critical preset values | — |
| `load_preset_p2()` | 62–63 | — | Tuple of 9 P2 high preset values | — |
| `load_preset_p4()` | 66–67 | — | Tuple of 9 P4 low preset values | — |
| `get_queue_table()` | 70–89 | — | `pd.DataFrame` of active queue | `database.get_emergency_queue()` |
| `get_history_table()` | 92–110 | `query` (str) | `pd.DataFrame` of patient records | `database.get_patient_history()` |
| `get_summary_html()` | 113–134 | — | HTML string with 4 metric cards | `database.get_dashboard_summary()` |
| `handle_update_status()` | 137–142 | Patient ID, new status | Status message, queue DF, summary HTML | `database.update_patient_status()` |
| `handle_tavily_search()` | 145–148 | Query string | Formatted search results | `search_tool.tavily_search()` |

### `agent_workflow.py` — LangGraph Agent Graph

| Function | Lines | Role | Key Calls |
|----------|-------|------|-----------|
| `get_groq_llm()` | 24–32 | Initialise Groq LLM with model fallback chain | `ChatGroq()` |
| `manager_node()` | 34–52 | Intent classification & routing | — |
| `triage_agent_node()` | 54–85 | ML classification + SQLite save | `predict_priority()`, `database.save_patient()` |
| `history_agent_node()` | 87–97 | Patient history retrieval | `database.get_patient_history()` |
| `search_agent_node()` | 99–109 | Tavily clinical search | `tavily_search()` |
| `synthesizer_node()` | 111–148 | LLM rationale + final payload assembly | `get_groq_llm().invoke()` |
| `build_agent_graph()` | 150–181 | Compile LangGraph `StateGraph` | `StateGraph`, `add_conditional_edges()` |
| `run_triage_workflow()` | 185–192 | Entry point for graph execution | `graph_app.invoke()` |

### `ml_tool.py` — Random Forest ML Tool

| Function | Lines | Inputs | Returns |
|----------|-------|--------|---------|
| `load_ml_components()` | 13–19 | — | Loads `model.joblib` and `scaler.joblib` into globals; trains if missing |
| `estimate_symptom_severity()` | 21–35 | `symptoms_text`, `pain_level` | Integer 1–4 |
| `parse_blood_pressure()` | 37–44 | BP string `"120/80"` | Tuple `(bp_sys, bp_dia)` |
| `predict_priority()` | 46–110 | `patient_vitals` dict | Dict with `priority`, `confidence`, `priority_label`, `probabilities`, `features_used` |

### `train_model.py` — Synthetic Data & Model Training

| Function | Lines | Inputs | Returns |
|----------|-------|--------|---------|
| `generate_synthetic_data()` | 11–73 | `n_samples=1200` | `pd.DataFrame` with 9 features + priority label |
| `train_and_evaluate()` | 75–118 | — | Metrics dict; saves `model.joblib`, `scaler.joblib`, `model_metrics.json` |

### `database.py` — SQLite Persistence Layer

| Function | Lines | Description |
|----------|-------|-------------|
| `get_connection()` | 7–10 | Returns `sqlite3.Connection` with `row_factory=sqlite3.Row` |
| `init_db()` | 12–47 | Creates `patients` and `history_chat` tables if not exist |
| `generate_patient_id()` | 49–56 | Returns `"PAT-{count+1001}"` |
| `save_patient()` | 58–91 | Inserts patient record; returns `patient_id` |
| `get_patient_history()` | 93–106 | Searches by `patient_id` or `name LIKE %query%` |
| `get_emergency_queue()` | 108–126 | Returns active patients sorted by `CASE` priority rank |
| `update_patient_status()` | 128–135 | Updates `queue_status` by `patient_id` |
| `get_dashboard_summary()` | 137–159 | Returns dict with `waiting_count`, `p1_count`, `p2_count`, `total_triaged` |
| `save_chat_message()` | 161–169 | Inserts chat message into `history_chat` |
| `get_chat_history()` | 171–177 | Retrieves chat history by `session_id` |

### `search_tool.py` — Tavily Search Integration

| Function | Lines | Inputs | Returns |
|----------|-------|--------|---------|
| `tavily_search()` | 4–35 | `query`, `max_results=3` | Formatted string with Tavily summary + source URLs; falls back to static clinical reference if API unavailable |

---

## 7. Viva Defense Cheat Sheet

### Q1: Why use Groq instead of OpenAI or other LLM providers?

**A:** Groq provides the fastest LLM inference (sub-200ms latency) via custom LPU hardware, which is critical for emergency triage where seconds matter. It also offers a free tier sufficient for prototype development. The system's model fallback chain (`compound-mini → compound → llama-3.3-70b → llama3-8b`) ensures resilience across Groq's model availability.

### Q2: Why LangGraph instead of a simple function pipeline?

**A:** LangGraph provides three capabilities a linear pipeline cannot:
1. **Conditional routing** — the Manager node dynamically routes to Triage, History, or Search agents based on intent, enabling a single entry point for multiple workflows.
2. **Structured state management** — `WorkflowState` TypedDict ensures type-safe data flow between agents without ad-hoc global variables.
3. **Execution traceability** — `agent_logs` accumulates a transparent trace of every agent decision, which is rendered to the UI for auditability.

### Q3: Why split ML classification from LLM rationale?

**A:** The ML model provides **deterministic, reproducible** priority classification that works offline. The LLM provides **natural language explanation** that requires an API. Decoupling ensures the critical classification never fails due to API outages, rate limits, or hallucinated outputs. The LLM is best-effort: if it fails, the system gracefully degrades to a static description.

### Q4: Why is the model accuracy 100%?

**A:** The training data is synthetically generated with well-separated vital sign ranges per priority class (e.g., P1: SpO2 70–88%, P4: SpO2 97–100%). This creates linearly separable clusters that a Random Forest with 100 trees and depth 10 can perfectly classify. In a production setting, overlapping real-world distributions would reduce accuracy and require clinical dataset validation.

### Q5: How does the priority queue maintain ordering?

**A:** The SQL `CASE` expression maps `P1→1, P2→2, P3→3, P4→4` and sorts by `priority_rank ASC, timestamp ASC`. This guarantees P1 patients always surface first, with FIFO ordering within the same priority level.

### Q6: What happens if both API keys are missing?

**A:** The system degrades gracefully:
- **No GROQ_API_KEY:** The Synthesizer node returns a static fallback rationale (`"Random Forest ML model evaluated vital signs..."`). Classification still works.
- **No TAVILY_API_KEY:** The search tool returns a static clinical reference about triage thresholds (SpO2 < 90%, HR > 130, BP < 90/60).
- The core ML classification pipeline has **zero external API dependencies**.

### Q7: What is the role of StandardScaler?

**A:** StandardScaler performs z-score normalisation (`(x - μ) / σ`) on the 9 feature columns. While Random Forest is inherently scale-invariant, the scaler ensures consistent feature distribution if the model is later swapped to a distance-based classifier (SVM, KNN). The scaler is fitted on training data and applied identically at inference time via `scaler.joblib`.

### Q8: How does symptom severity estimation work without an NLP model?

**A:** `estimate_symptom_severity()` uses a tiered keyword-matching approach: it scans the free-text symptoms field against curated clinical keyword lists (critical → 4, severe → 3, moderate → 2, low → 1). The pain score serves as a secondary signal. This avoids the latency and complexity of an NLP model while providing clinically meaningful stratification for the Random Forest input.

### Q9: What are the system's limitations?

**A:** Key limitations:
1. **Synthetic training data** — 100% accuracy does not transfer to real clinical distributions.
2. **Keyword-based NLP** — symptom severity estimation misses synonyms, negations, and context (e.g., "no chest pain" would still trigger critical).
3. **Single-language** — English-only symptom parsing.
4. **No HIPAA/GDPR compliance** — SQLite with no encryption; unsuitable for production PHI storage.
5. **No real-time vitals integration** — manual data entry only; no IoT/monitor feeds.

### Q10: How would you improve this for production deployment?

**A:** Production roadmap:
1. Replace synthetic data with anonymised clinical datasets (MIMIC-III/IV).
2. Add model monitoring with drift detection (Evidently AI).
3. Swap SQLite for PostgreSQL with encryption-at-rest.
4. Integrate HL7 FHIR for EHR interoperability.
5. Add role-based access control and audit logging.
6. Replace keyword NLP with a fine-tuned clinical NER model (MedSpaCy).
7. Deploy behind HTTPS with HIPAA-compliant infrastructure.

---

*Generated from codebase inspection of `app.py`, `agent_workflow.py`, `ml_tool.py`, `train_model.py`, `database.py`, and `search_tool.py`.*

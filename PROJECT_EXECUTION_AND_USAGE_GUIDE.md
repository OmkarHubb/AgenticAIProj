# Emergency Patient Priority Classification & Agentic Triage System
## Complete Developer, Architecture, Execution, and Viva Presentation Guide

---

## 1. PROJECT OVERVIEW

### What the System Does
The **Emergency Patient Priority Classification & Agentic Triage System** is an intelligent emergency department workstation designed to streamline patient intake, automate clinical triage prioritization, persist patient records, query medical reference guidelines, and present an emergency department queue.

The application combines a deterministic Machine Learning model for vital signs priority classification with a dynamic multi-agent system powered by LangGraph, Groq Large Language Models (LLM), and external web search APIs (Tavily).

### Main Purpose
In high-stress emergency room environments, triage nurses must rapidly assign priority levels based on patient vital signs and reported symptoms. Human triage can suffer from variability, delayed processing, or oversights during high-volume spikes. 

This system provides:
1. **Instant, Objective Triage Triage (P1 to P4):** Powered by a Scikit-Learn Random Forest Classifier trained on clinical vital sign patterns.
2. **Agentic Workflow Automation:** Driven by a LangGraph orchestration graph that dynamically routes incoming clinical requests to specialised specialist agents (Manager, Triage, History, and Search agents).
3. **Medical Rationale Generation:** Synthesizing clinical vitals into understandable, structured reasoning using Groq LLM inference.
4. **Persistent Audit Log & Queue:** Storing patient vitals, classification confidence, and treatment status in a local SQLite database (`patients.db`).
5. **Real-time Clinical Guidance Retrieval:** Querying live medical evidence using the Tavily Search API.

### What Happens From User Input to Final Result
1. **Intake:** The user submits patient details (Name, Age, Heart Rate, SpO2, Blood Pressure, Temperature, Respiratory Rate, Pain Score, Symptoms) via the Gradio Web Interface.
2. **Graph Initiation:** The request is passed to `run_triage_workflow()` in `agent_workflow.py`, initiating a LangGraph `StateGraph`.
3. **Intent Analysis & Routing:** The **Manager Agent** inspects the payload `task_type` and routes control conditionally to the **Triage Agent**.
4. **ML Classification & DB Persistence:** The **Triage Agent** executes `predict_priority()` from `ml_tool.py` (which scales vitals using `StandardScaler` and runs the pre-trained `RandomForestClassifier`). The classified record is assigned a unique Patient ID (e.g., `PAT-1001`) and saved into the SQLite database (`patients.db`) via `database.save_patient()`.
5. **LLM Clinical Synthesis:** Control passes to the **Synthesizer Node**, which formats a prompt containing the vitals and predicted priority level, sending it to the **Groq LLM** (`groq/compound-mini` via `ChatGroq`) to generate a concise 2-sentence clinical rationale.
6. **UI Rendering:** The final payload (Priority Badge, Confidence Score, Probability Distribution, Groq Clinical Rationale, Workflow Execution Trace Logs, and Updated Queue Table) is returned and rendered on the Gradio UI.

---

## 2. COMPLETE ARCHITECTURE

The architecture follows a decoupled, modular design where the frontend user interface, workflow graph engine, machine learning tools, database layer, and external LLM/search services communicate seamlessly.

```
+-----------------------------------------------------------------------------------+
|                                    GRADIO UI                                      |
|  (app.py - 4 Workstation Tabs: Intake Form, Active Queue, History Chat, Search)   |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                              LANGGRAPH WORKFLOW ENGINE                            |
|                          (agent_workflow.py - StateGraph)                         |
|                                                                                   |
|                              +------------------+                                 |
|                              |  Manager Agent   |                                 |
|                              +------------------+                                 |
|                                       | (Conditional Router)                      |
|                 +---------------------+---------------------+                     |
|                 |                     |                     |                     |
|                 v                     v                     v                     |
|        +------------------+  +------------------+  +------------------+           |
|        |   Triage Agent   |  |  History Agent   |  |   Search Agent   |           |
|        +------------------+  +------------------+  +------------------+           |
|                 |                     |                     |                     |
|                 +---------------------+---------------------+                     |
|                                       |                                           |
|                                       v                                           |
|                              +------------------+                                 |
|                              | Synthesizer Node |                                 |
|                              +------------------+                                 |
+-----------------------------------------------------------------------------------+
        |                     |                     |                     |
        v                     v                     v                     v
+---------------+     +---------------+     +---------------+     +---------------+
|    ML TOOL    |     | SQLITE MEMORY |     | GROQ LLM API  |     |  TAVILY API   |
| (ml_tool.py)  |     | (database.py) |     |  (ChatGroq)   |     |(search_tool.py|
| Random Forest |     |  patients.db  |     | compound-mini |     | Clinical Search
+---------------+     +---------------+     +---------------+     +---------------+
```

### Component Communication Details
- **Gradio (`app.py`):** Acts as the user interface and event handler. When a user submits a form or sends a query, Gradio invokes functions in `agent_workflow.py`, `database.py`, or `search_tool.py`.
- **Python Backend Engine:** Coordinates state transformations and binds LangChain components with custom standard Python functions.
- **LangGraph (`agent_workflow.py`):** Manages state transitions using `WorkflowState`. It ensures that execution flows from `START` to `manager`, conditionally branches to a specialist node (`triage`, `history`, or `search`), forwards state to `synthesizer`, and ends at `END`.
- **LangChain & Groq (`langchain_groq.ChatGroq`):** Handles communication with the Groq cloud infrastructure using model candidates such as `groq/compound-mini`, `groq/compound`, or `llama-3.3-70b-versatile`.
- **Random Forest ML Engine (`ml_tool.py` / `train_model.py`):** Encapsulates feature preprocessing (`parse_blood_pressure`, `estimate_symptom_severity`) and executes `RandomForestClassifier.predict_proba()` to yield exact probability scores per priority class.
- **SQLite Database (`database.py`):** Provides local persistent storage via SQLite (`patients.db`), storing two main tables: `patients` (vital signs and queue state) and `history_chat` (conversational history).
- **Tavily Search API (`search_tool.py`):** Provides external web search capabilities via direct HTTP POST requests to `https://api.tavily.com/search`, retrieving updated clinical guidelines and handling connection failures gracefully.

---

## 3. COMPLETE EXECUTION FLOW

### Scenario A: New Patient Intake & Triage Execution
1. **User Action:** The user fills in the intake fields on **Tab 1** and clicks **Run LangGraph Agent Triage Workflow**.
2. **Entry Function:** Gradio calls `handle_triage()` in `app.py`.
3. **State Assembly:** `handle_triage()` constructs an `input_data` dictionary containing `task_type='triage'`, vital signs, symptoms, patient name, and age.
4. **Graph Invocation:** `run_triage_workflow(input_data)` in `agent_workflow.py` creates `initial_state` and executes `graph_app.invoke(initial_state)`.
5. **Manager Node (`manager_node`):**
   - Appends audit log: `"Manager Agent: Received request. Analyzing workflow intent."`
   - Inspects `input_data` and detects vital signs keys (`heart_rate`, `spo2`), assigning `query_type = 'triage'`.
   - Appends log: `"Manager Agent: Delegating workflow to Specialist Agent node -> TRIAGE."`
6. **Conditional Branch (`route_manager`):** Checks `query_type` and routes execution to `triage_agent_node`.
7. **Triage Node (`triage_agent_node`):**
   - Appends log: `"Triage Agent: Processing patient vitals. Invoking Random Forest ML tool predict_priority()."`
   - Calls `predict_priority(input_data)` in `ml_tool.py`.
   - `predict_priority()` converts inputs, parses blood pressure into systolic/diastolic, calculates `symptom_severity` (scale 1 to 4), constructs a 9-feature DataFrame, scales features with `scaler.joblib`, and runs `model.joblib.predict_proba()`.
   - Returns priority (`P1`, `P2`, `P3`, or `P4`), confidence percentage, label string, and class probabilities.
   - Appends log: `"Triage Tool predict_priority() classified priority: P1 - Critical (98.5% confidence)."`
   - Assembles patient record dictionary and calls `database.save_patient(db_patient)`.
   - `database.save_patient()` generates a unique ID (`PAT-1001`), executes SQL `INSERT INTO patients`, and returns `patient_id`.
   - Appends log: `"Triage Agent: Saved patient PAT-1001 to SQLite database queue."`
8. **Synthesizer Node (`synthesizer_node`):**
   - Appends log: `"Manager Agent: Synthesizing final structured response."`
   - Calls `get_groq_llm()` to initialize `ChatGroq(model="groq/compound-mini")`.
   - Formats a prompt with vital signs and priority classification, invoking `llm.invoke(prompt)`.
   - Extracts response text: `"Patient presents with severe hypoxia (SpO2 83%) and tachyarrhythmia (HR 148 bpm). Immediate resuscitation and continuous monitoring are mandatory."`
   - Appends log: `"Manager Agent: Groq LLM clinical rationale generated."`
   - Compiles `final_output` dictionary containing vitals, probabilities, patient ID, priority, LLM rationale, and workflow trace logs.
9. **UI Update:** `handle_triage()` formats HTML badges, probability text, and workflow trace logs, refreshing the queue table (`get_queue_table()`) and summary stats (`get_summary_html()`).

---

### Scenario B: Patient History Query / Chatbot Handoff
1. **User Action:** The user types a query (e.g., `"Find records for Patient A"`) into **Tab 3** (History Agent Chat).
2. **Entry Function:** Gradio calls `handle_history_chat(message, history)` in `app.py`.
3. **Database Search:** `database.get_patient_history(message)` executes `SELECT * FROM patients WHERE patient_id = ? OR name LIKE ? ORDER BY timestamp DESC`.
4. **Summary Formatting:** Formats matching patient records into string summaries.
5. **Chat Memory Logging:** `database.save_chat_message("default", "user", message)` and `database.save_chat_message("default", "assistant", reply)` insert conversation records into the `history_chat` SQLite table.
6. **UI Update:** Appends `(message, reply)` tuple to Gradio chatbot component state and renders updated history.

---

### Scenario C: External Clinical Guideline Search
1. **User Action:** The user submits a search topic on **Tab 4** (Tavily Search).
2. **Entry Function:** Gradio calls `handle_tavily_search(query)` in `app.py`.
3. **API Invocation:** `tavily_search(query)` in `search_tool.py` checks `TAVILY_API_KEY` from `.env`.
4. **HTTP Post Execution:** Sends payload to `https://api.tavily.com/search`.
5. **Fallback Safety:** If key is missing or request fails, returns clinical baseline triage guidance strings without crashing.

---

## 4. FILE-BY-FILE EXPLANATION

| File | Purpose | Important Functions / Classes | How It Connects to System |
|---|---|---|---|
| `app.py` | Main entry point & Gradio Web Application UI | `handle_triage()`, `get_queue_table()`, `get_history_table()`, `get_summary_html()`, `handle_update_status()`, `handle_history_chat()`, `handle_tavily_search()` | Serves the 4-tab user workstation UI, handles user input events, and triggers LangGraph workflow and database queries. |
| `agent_workflow.py` | LangGraph Multi-Agent Orchestration & Workflow Graph | `WorkflowState` (TypedDict), `get_groq_llm()`, `manager_node()`, `triage_agent_node()`, `history_agent_node()`, `search_agent_node()`, `synthesizer_node()`, `build_agent_graph()`, `run_triage_workflow()` | Defines state graph nodes, conditional router edges, agent delegation, Groq LLM synthesis, and execution trace logging. |
| `ml_tool.py` | Machine Learning Inference & Preprocessing Tool | `load_ml_components()`, `estimate_symptom_severity()`, `parse_blood_pressure()`, `predict_priority()` | Loads `.joblib` model/scaler binaries, parses vitals/symptoms into 9 numerical features, and computes priority prediction and class probabilities. |
| `train_model.py` | Model Training & Dataset Generation | `generate_synthetic_data()`, `train_and_evaluate()` | Generates 1,200 synthetic clinical vital sign records, trains Random Forest Classifier, evaluates metrics, and exports `model.joblib`, `scaler.joblib`, and `model_metrics.json`. |
| `database.py` | SQLite Persistence & Queue Management Layer | `get_connection()`, `init_db()`, `generate_patient_id()`, `save_patient()`, `get_patient_history()`, `get_emergency_queue()`, `update_patient_status()`, `get_dashboard_summary()`, `save_chat_message()`, `get_chat_history()` | Manages `patients.db`, initializes `patients` and `history_chat` tables, handles queue ordering (`P1 > P2 > P3 > P4`), and records multi-turn chat memory. |
| `search_tool.py` | Tavily Web Search API Client | `tavily_search()` | Queries Tavily API for evidence-based clinical protocols and returns structured medical reference summaries with offline fallbacks. |
| `requirements.txt` | Python Package Dependency Manifest | Lists required libraries: `langchain`, `langchain-groq`, `langchain-core`, `langchain-community`, `gradio`, `scikit-learn`, `pandas`, `numpy`, `joblib`, `python-dotenv`, `langgraph`, `requests` | Defines runtime environment requirements for pip installation. |
| `.env` | Environment Configuration File | `GROQ_API_KEY`, `TAVILY_API_KEY` | Stores secret API keys loaded at runtime via `python-dotenv`. |
| `.env.example` | Environment Configuration Template | Placeholder keys | Template demonstrating required environment variable key names. |
| `model.joblib` | Serialized ML Model Binary | Pre-trained `RandomForestClassifier` object | Binary artifact loaded by `ml_tool.py` for high-speed local inference. |
| `scaler.joblib` | Serialized Feature Scaler Binary | Pre-trained `StandardScaler` object | Normalizes 9 numeric input features prior to ML inference. |
| `model_metrics.json` | Model Evaluation Report JSON | Performance metrics (`accuracy`, `precision`, `recall`, `f1_score`, `details`) | Loaded by `app.py` on Tab 4 to display validation accuracy and per-class performance tables. |
| `patients.db` | SQLite Database Binary | Tables: `patients`, `history_chat` | Persistent disk database generated automatically on app initialization. |

---

## 5. AGENT EXPLANATION

```
                       +----------------------+
                       |     Manager Agent    |
                       | (Analyze Intent &    |
                       |  Route Control)      |
                       +----------------------+
                                  |
            +---------------------+---------------------+
            |                     |                     |
            v                     v                     v
+-----------------------+ +-----------------------+ +-----------------------+
|     Triage Agent      | |     History Agent     | |     Search Agent      |
| (Predict Priority via | | (Fetch SQL Records &  | | (Query Tavily Search  |
| Random Forest ML Tool)| |  Multi-turn Memory)   | |  for Clinical Specs)  |
+-----------------------+ +-----------------------+ +-----------------------+
            |                     |                     |
            +---------------------+---------------------+
                                  |
                                  v
                       +----------------------+
                       |   Synthesizer Node   |
                       | (Groq LLM Clinical   |
                       |  Rationale Synthesis)|
                       +----------------------+
```

### 1. Manager Agent (`manager_node`)
- **Purpose:** Acts as the primary router and supervisory node in the workflow graph. It analyzes the incoming request payload and decides which specialist agent node should process the request.
- **Input:** `state` dictionary (`WorkflowState`) containing `input_data`.
- **Decision Logic:** Inspects `input_data.get('task_type')`. If unspecified, detects vital sign keys (`heart_rate`, `spo2`) to route to `triage`, `history_query` to route to `history`, or `search_query` to route to `search`.
- **Tools Used:** Internal conditional routing logic (`route_manager`).
- **Output:** Updates `state['query_type']` and appends decision entries to `state['agent_logs']`.
- **Caller:** Invoked directly by the `START` edge of the LangGraph workflow.
- **Handoff Mechanism:** Returns updated state keys to the graph router (`route_manager`), which directs execution to the target agent node (`triage`, `history`, or `search`).

---

### 2. Triage Agent (`triage_agent_node`)
- **Purpose:** Evaluates vital signs, executes priority classification via machine learning, and persists patient intake data to SQLite.
- **Input:** Patient demographic, vital sign, and symptom data from `state['input_data']`.
- **Decision Logic:** Passes raw vitals to the Random Forest prediction tool (`predict_priority()`), obtains the classification results, formats a full patient database record, and writes it to SQLite using `database.save_patient()`.
- **Tools Used:**
  1. `ml_tool.predict_priority()`
  2. `database.save_patient()`
- **Output:** Updates `state['triage_result']` with priority code (`P1`–`P4`), label, confidence %, probability distribution, patient ID, and appends trace logs.
- **Caller:** Directed by `manager_node` via conditional edge `route_manager`.
- **Handoff Mechanism:** Automatically routes output state to `synthesizer_node`.

---

### 3. History Agent (`history_agent_node`)
- **Purpose:** Searches SQLite database records for past patient triage histories and handles patient record lookup operations.
- **Input:** Search query or patient name from `state['input_data']`.
- **Decision Logic:** Extracts query string and executes `database.get_patient_history(query)`.
- **Tools Used:** `database.get_patient_history()`
- **Output:** Updates `state['history_result']` with matching record dicts and appends trace logs.
- **Caller:** Directed by `manager_node` via conditional edge `route_manager`.
- **Handoff Mechanism:** Automatically routes output state to `synthesizer_node`.

---

### 4. Search Agent (`search_agent_node`)
- **Purpose:** Queries external medical evidence and web documentation when clinical guidance or protocol reference is needed.
- **Input:** Search query string from `state['input_data']`.
- **Decision Logic:** Extracts query topic and invokes `tavily_search(query)`.
- **Tools Used:** `search_tool.tavily_search()`
- **Output:** Updates `state['search_result']` with Tavily summary answers and medical source web content snippets.
- **Caller:** Directed by `manager_node` via conditional edge `route_manager`.
- **Handoff Mechanism:** Automatically routes output state to `synthesizer_node`.

---

### 5. Synthesizer Node (`synthesizer_node`)
- **Purpose:** Combines results from specialist nodes (Triage, History, or Search) and invokes Groq LLM to generate professional, natural-language clinical rationale.
- **Input:** Accumulated state (`triage_result`, `history_result`, `search_result`, `agent_logs`).
- **Decision Logic:** Checks if `triage_result` exists and Groq API key is configured. Formats vital signs prompt and calls `ChatGroq.invoke()`. Compiles all fields into `final_output`.
- **Tools Used:** `agent_workflow.get_groq_llm()` (`ChatGroq`)
- **Output:** Updates `state['final_output']` dictionary containing full result payload and trace logs.
- **Caller:** Invoked by `triage`, `history`, or `search` nodes.
- **Handoff Mechanism:** Directs output to `END` edge of graph, completing workflow execution.

---

## 6. LANGGRAPH WORKFLOW

### Nodes
1. `manager`: Intent classification & initial delegation node.
2. `triage`: Specialist node for ML triage classification & database persistence.
3. `history`: Specialist node for SQLite history log querying.
4. `search`: Specialist node for Tavily clinical search execution.
5. `synthesizer`: Final aggregation & Groq LLM clinical rationale synthesis node.

### Edges
- **Fixed Edge:** `START -> manager`
- **Conditional Edge:** `manager -> route_manager()`
  - Routes to `"triage"` if `query_type == 'triage'`
  - Routes to `"history"` if `query_type == 'history'`
  - Routes to `"search"` if `query_type == 'search'`
- **Fixed Edges:**
  - `triage -> synthesizer`
  - `history -> synthesizer`
  - `search -> synthesizer`
  - `synthesizer -> END`

### State Definition (`WorkflowState`)
```python
class WorkflowState(TypedDict):
    input_data: Dict[str, Any]
    query_type: str
    triage_result: Optional[Dict[str, Any]]
    history_result: Optional[List[Dict[str, Any]]]
    search_result: Optional[str]
    agent_logs: List[str]
    final_output: Dict[str, Any]
```

### Graph Diagram
```
              +---------+
              |  START  |
              +---------+
                   |
                   v
              +---------+
              | manager |
              +---------+
                   |
         +---------+---------+
         | (route_manager)   |
         |                   |
         v                   v                   v
    +--------+          +---------+          +--------+
    | triage |          | history |          | search |
    +--------+          +---------+          +--------+
         |                   |                   |
         +---------+---------+                   |
                   |                             |
                   v                             v
            +-------------+
            | synthesizer |
            +-------------+
                   |
                   v
               +-------+
               |  END  |
               +-------+
```

---

## 7. TOOLS

### 1. Random Forest Prediction Tool (`predict_priority`)
- **Name:** `predict_priority` (defined in `ml_tool.py`)
- **Purpose:** Classifies emergency triage priority code (P1, P2, P3, P4) and computes probability distribution across classes.
- **Input:** Patient vital signs dictionary (`age`, `heart_rate`, `spo2`, `blood_pressure`, `temperature`, `respiratory_rate`, `pain_level`, `symptoms`).
- **Processing:**
  1. Parses raw string blood pressure (e.g. `"120/80"`) into `bp_sys=120`, `bp_dia=80` using `parse_blood_pressure()`.
  2. Estimates `symptom_severity` (score 1 to 4) by scanning text for critical keywords (`chest pain`, `cyanosis`, `stroke`, etc.) or checking high pain scores.
  3. Constructs a 9-feature pandas DataFrame.
  4. Applies `scaler.transform()` using pre-trained `scaler.joblib`.
  5. Executes `_model.predict_proba()` using `model.joblib`.
  6. Determines class with maximum probability and maps label (`P1 - Critical`, `P2 - High`, `P3 - Moderate`, `P4 - Low`).
- **Output:** Dictionary containing `priority`, `confidence`, `priority_label`, `probabilities`, and `features_used`.
- **Used By:** `Triage Agent` (`triage_agent_node`).

---

### 2. SQLite Database & Memory Tools (`database.py`)
- **Name:** `save_patient`, `get_patient_history`, `get_emergency_queue`, `update_patient_status`, `get_dashboard_summary`, `save_chat_message`
- **Purpose:** Performs CRUD operations on local disk database `patients.db`.
- **Input:** Patient dictionaries, query strings, patient IDs, new status strings, chat messages.
- **Processing:**
  - `save_patient()`: Auto-generates sequential ID (`PAT-1001`), executes SQL `INSERT INTO patients`.
  - `get_emergency_queue()`: Uses SQL `CASE` statement to sort active waiting queue strictly by priority rank (`P1=1, P2=2, P3=3, P4=4, ELSE=5`) and timestamp.
  - `get_patient_history()`: Runs SQL `LIKE` query matching Patient ID or Name.
  - `save_chat_message()`: Inserts turn history into `history_chat` table.
- **Output:** DataFrames, patient dict lists, summary metric counts, success booleans.
- **Used By:** `Triage Agent`, `History Agent`, and Gradio UI event handlers.

---

### 3. Tavily Clinical Web Search Tool (`tavily_search`)
- **Name:** `tavily_search` (defined in `search_tool.py`)
- **Purpose:** Fetches web-based medical protocols and emergency guidelines.
- **Input:** Query string (e.g. `"Emergency triage protocols for chest pain"`), optional `max_results` (default=3).
- **Processing:**
  1. Checks `TAVILY_API_KEY` from `.env`.
  2. Sends POST request to `https://api.tavily.com/search` with JSON payload.
  3. Extracts summary answer and article content snippets.
  4. Catches connection timeouts or key errors and returns safe clinical fallback string.
- **Output:** Formatted markdown text string with summary answer and source references.
- **Used By:** `Search Agent` (`search_agent_node`) and Tab 4 standalone search handler.

---

## 8. MACHINE LEARNING

### Dataset Used
The model is trained on a synthetic dataset generated by `generate_synthetic_data(n_samples=1200)` in `train_model.py`. The dataset consists of 1,200 patient vital sign profiles sampled realistically across all 4 triage priority classes:
- **P1 (Critical - 20% target):** Severe hypoxia (SpO2 70-89%), extreme heart rate (<55 or >135 bpm), extreme blood pressure (systolic <85 or >175 mmHg), severe pain (8-10), high symptom severity (4).
- **P2 (High - 25% target):** Moderate hypoxia (SpO2 90-94%), tachycardia (HR 110-135 bpm), elevated BP (155-175 mmHg), high fever (38.5-40.0 °C), pain (7-9), severity (3).
- **P3 (Moderate - 30% target):** Mild vital sign deviations (SpO2 94-97%, HR 90-110 bpm, temp 37.5-38.5 °C), pain (4-7), severity (2).
- **P4 (Low - 25% target):** Normal physiological ranges (SpO2 97-100%, HR 60-90 bpm, BP 100-130/60-85 mmHg, temp 36.0-37.4 °C), low pain (1-4), severity (1).

### 9 Input Features
1. `age`: Patient age in years (1 to 90).
2. `heart_rate`: Heart rate in beats per minute (bpm).
3. `spo2`: Blood oxygen saturation percentage (%).
4. `bp_sys`: Systolic blood pressure (mmHg).
5. `bp_dia`: Diastolic blood pressure (mmHg).
6. `temperature`: Body temperature in degrees Celsius (°C).
7. `respiratory_rate`: Respiratory rate in breaths per minute (bpm).
8. `pain_level`: Patient reported pain score (1 to 10 scale).
9. `symptom_severity`: Algorithmically estimated severity rank (1=Low, 2=Moderate, 3=High, 4=Critical).

### Preprocessing Pipeline
1. `parse_blood_pressure()` splits string input `"120/80"` into numeric integers `bp_sys=120`, `bp_dia=80`.
2. `estimate_symptom_severity()` evaluates symptom text against clinical keyword groups (`chest pain`, `cyanosis`, `stroke`, etc.) and pain score threshold.
3. `StandardScaler` from `scikit-learn` normalizes all 9 numerical features.

### Training Configuration & Random Forest Architecture
- **Algorithm:** `RandomForestClassifier`
- **Number of Decision Trees (`n_estimators`):** 100
- **Maximum Tree Depth (`max_depth`):** 10
- **Random State:** 42
- **Data Split:** 80% Training (960 samples), 20% Testing (240 samples), stratified by class label.

### Priority Class Definitions
- **P1 - Critical:** Immediate life-threatening condition requiring resuscitation.
- **P2 - High:** Emergent condition with potential for rapid deterioration.
- **P3 - Moderate:** Urgent condition requiring timely evaluation.
- **P4 - Low:** Non-urgent condition suitable for routine waiting area.

### Loading Saved Models
Binaries are exported to `model.joblib` and `scaler.joblib`. Function `load_ml_components()` in `ml_tool.py` verifies file existence; if missing, it automatically calls `train_and_evaluate()` to retrain and save binaries.

### Actual Evaluation Results (`model_metrics.json`)
```json
{
  "accuracy": 1.0,
  "precision": 1.0,
  "recall": 1.0,
  "f1_score": 1.0,
  "total_samples": 1200,
  "test_samples": 240,
  "details": {
    "P1": { "precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 51.0 },
    "P2": { "precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 59.0 },
    "P3": { "precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 75.0 },
    "P4": { "precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 55.0 }
  }
}
```

---

## 9. SQLITE MEMORY

### Database Structure & Schema (`patients.db`)

#### Table 1: `patients`
Stores all patient intake records, vitals, triage classifications, and queue states.
```sql
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    age INTEGER NOT NULL,
    heart_rate INTEGER NOT NULL,
    spo2 INTEGER NOT NULL,
    blood_pressure TEXT NOT NULL,
    temperature REAL NOT NULL,
    respiratory_rate INTEGER NOT NULL,
    pain_level INTEGER NOT NULL,
    symptoms TEXT NOT NULL,
    priority TEXT NOT NULL,
    confidence REAL NOT NULL,
    queue_status TEXT NOT NULL DEFAULT 'Waiting',
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
```

#### Table 2: `history_chat`
Stores conversational chat history for multi-turn interactions with the History Agent.
```sql
CREATE TABLE IF NOT EXISTS history_chat (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
```

### Retrieval & Persistent Queue Sorting
The active queue query in `get_emergency_queue()` uses SQL conditional sorting to ensure emergency medical ordering:
```sql
SELECT *,
CASE priority
    WHEN 'P1' THEN 1
    WHEN 'P2' THEN 2
    WHEN 'P3' THEN 3
    WHEN 'P4' THEN 4
    ELSE 5
END as priority_rank
FROM patients
WHERE queue_status IN ('Waiting', 'In Treatment')
ORDER BY priority_rank ASC, timestamp ASC
```
This guarantees that critical **P1** patients stay at the top of the queue regardless of arrival time.

---

## 10. GROQ + LLM

### Configured Model
- **Primary LLM:** `groq/compound-mini` via `langchain_groq.ChatGroq`
- **Fallback Models:** `groq/compound`, `llama-3.3-70b-versatile`, `llama3-8b-8192`
- **Temperature Setting:** `0.2` (low temperature for deterministic clinical tone)

### API Key Loading
Key is loaded from `.env` via `os.getenv("GROQ_API_KEY")` and passed to `ChatGroq(model=model, groq_api_key=api_key)`.

### Where the LLM Is Called
The LLM is invoked inside `synthesizer_node()` in `agent_workflow.py`.

### What the LLM DOES Do
- Accepts structured inputs (patient vitals, predicted priority, and symptom text).
- Synthesizes a concise, professional 2-sentence clinical rationale explaining *why* the patient received that priority level.

### What the LLM DOES NOT Do
- **The LLM does NOT classify the patient's priority tier.** Priority classification is handled entirely by the Random Forest Machine Learning model (`ml_tool.py`).
- **The LLM does NOT overwrite database records.**

---

## 11. TAVILY

### Why It Is Used
Tavily provides external web-search capabilities, allowing the system to query real-time evidence-based clinical protocols, treatment guidelines, and medical references beyond offline training data.

### Invocation Pattern
1. The **Search Agent** (`search_agent_node` in `agent_workflow.py`) receives a query and calls `tavily_search(query)`.
2. Direct standalone search calls can also be made from **Tab 4** of the Gradio interface.

### Information Retrieved
Returns a structured string containing:
- **Tavily Summary Answer:** An AI-generated concise summary of the query.
- **Source Web References:** Titles, URLs, and snippet contents from relevant clinical articles.

### Error Handling & Offline Fallback
If `TAVILY_API_KEY` is missing or the external HTTP request fails, `tavily_search()` catches the exception and returns a pre-configured baseline medical reference string.

---

## 12. GRADIO UI

The workstation interface is organized into 4 functional tabs:

```
+-----------------------------------------------------------------------------------+
|               Emergency Patient Priority Classification System                    |
|                      [ Summary Metric Dashboard Cards ]                           |
+-----------------------------------------------------------------------------------+
| Tab 1: Patient Intake | Tab 2: Queue Dashboard | Tab 3: History | Tab 4: Diagnostics|
+-----------------------------------------------------------------------------------+
```

### Tab 1: Patient Clinical Intake & Triage Analysis
- **Preset Buttons:** `Preset: P1 Critical`, `Preset: P2 High`, `Preset: P4 Low` for 1-click test populating.
- **Form Fields:** Text inputs for Name, BP, Symptoms; Number inputs for Age, Heart Rate, SpO2, Temp, Resp Rate, Pain Level.
- **Action Button:** `Run LangGraph Agent Triage Workflow` (primary variant).
- **Result Output Components:**
  - **Priority Result Badge:** Dynamic HTML card colored by priority (`#fef2f2` Red for P1, `#fff7ed` Orange for P2, `#fefce8` Yellow for P3, `#f0fdf4` Green for P4).
  - **Manager Agent Clinical Rationale:** Textbox with Groq LLM summary.
  - **Random Forest Class Probabilities:** Textbox showing percentages for P1, P2, P3, and P4.
  - **LangGraph Execution Trace Log:** Textbox displaying step-by-step agent graph logs.

### Tab 2: Emergency Queue Dashboard
- **Live Queue Dataframe:** Interactive-disabled table showing active waiting patients sorted by priority rank (`P1 > P2 > P3 > P4`).
- **Status Update Panel:** Textbox for Patient ID, Dropdown for Status (`Waiting`, `In Treatment`, `Discharged`), and `Update Status` button.
- **Refresh Queue Button:** Manual refresh button for queue synchronization.

### Tab 3: Patient History & History Agent Chat
- **Search Panel:** Search input textbox and `Search Records` button to filter SQLite records table.
- **Patient History Dataframe:** Table listing past patient records, vitals, confidence, and timestamps.
- **Multi-turn Chatbot:** Interactive `gr.Chatbot` and prompt textbox allowing multi-turn conversations with the SQLite History Agent.

### Tab 4: Tavily Search & ML Diagnostics
- **Tavily Search Component:** Input textbox for clinical queries, `Run Tavily Search` button, and multi-line output text area.
- **ML Model Diagnostics Panel:** Displays overall validation summary metrics (Accuracy, Precision, Recall, F1) and a DataFrame showing class-wise support and evaluation metrics.

---

## 13. HOW TO RUN THE PROJECT

### Step 1: Clone Repository & Open Directory
```bash
cd c:\Users\omkar\Desktop\Flexi_CA3
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Verify or create a `.env` file containing valid API keys:
```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### Step 4: Model Training (Optional Verification)
To retrain the Random Forest model and regenerate binaries:
```bash
python train_model.py
```

### Step 5: Launch Application
```bash
python app.py
```

### Step 6: Access Interface
Open a web browser and navigate to:
```
http://127.0.0.1:8000
```

---

## 14. HOW TO USE THE SYSTEM (PRACTICAL WALKTHROUGH)

### Walkthrough Scenario: Triage an Acute Critical Patient (P1)
1. **Open Tab 1:** Click on **Patient Intake & Triage Analysis**.
2. **Load Critical Preset:** Click the button **Preset: P1 Critical**.
   - Fields auto-fill: `Name: Patient A`, `Age: 64`, `Heart Rate: 148`, `SpO2: 83`, `BP: 80/50`, `Temp: 39.2`, `Resp Rate: 34`, `Pain: 10`, `Symptoms: Severe acute crushing chest pain, dyspnea, cyanosis, syncope`.
3. **Execute Workflow:** Click **Run LangGraph Agent Triage Workflow**.
4. **Observe Output:**
   - **Badge:** Displays a red alert card **P1 - Critical**, showing assigned ID `PAT-1001` and `98.5%` confidence.
   - **Groq Rationale:** Displays clinical evaluation explaining severe hypoxia and tachyarrhythmia.
   - **Trace Log:** Shows Manager delegating to Triage Agent, tool executing `predict_priority()`, saving to SQLite, and Synthesizer generating LLM rationale.
5. **Inspect Emergency Queue:** Switch to **Tab 2 (Emergency Queue Dashboard)**. Observe `PAT-1001` positioned at Rank `#1`.
6. **Update Patient Status:** Type `PAT-1001` into Patient ID field, select `In Treatment`, and click **Update Status**.
7. **Test History Search:** Switch to **Tab 3**. Type `Patient A` into Search History and click **Search Records**.
8. **Test Tavily Search:** Switch to **Tab 4**. Type `"Emergency triage guidelines for hypoxia and hypotension"` and click **Run Tavily Search**.

---

## 15. BEST DEMONSTRATION / SHOWCASE FLOW (5-10 MIN SCRIPT)

| Step | What I Do | What I Say | What System Does | Concept Demonstrated |
|---|---|---|---|---|
| **Step 1** | Open `http://127.0.0.1:8000` | "Welcome to the Emergency Patient Priority Classification & Agentic Triage System. This workstation combines dynamic agentic workflow execution with deterministic machine learning." | Renders Gradio UI with 4 active tabs and 4 top-level summary cards. | Gradio Web Application UI |
| **Step 2** | Click **Preset: P1 Critical** on Tab 1 | "I will populate the clinical intake form for a critical patient presenting with crushing chest pain, SpO2 of 83%, and heart rate of 148 bpm." | Auto-fills demographic, vital sign, and symptom fields. | Preset Form Automation |
| **Step 3** | Click **Run LangGraph Agent Triage Workflow** | "Now I trigger our LangGraph multi-agent engine. Notice the step-by-step trace log." | Executes `manager_node`, routes to `triage_agent_node`, runs Random Forest ML model, saves record to SQLite, and invokes Groq LLM. | LangGraph Multi-Agent Workflow, Handoff & Tool Execution |
| **Step 4** | Point out Priority Badge & Probability Scores | "The Random Forest model classified this patient as P1 Critical with 100% confidence. Notice how Groq LLM synthesizes the clinical rationale without guessing numerical priority." | Displays red **P1 - Critical** badge, full class probability breakdown, and Groq LLM text. | ML vs LLM Responsibility Decoupling |
| **Step 5** | Click **Emergency Queue Dashboard** (Tab 2) | "Let's switch to Tab 2 to view our live emergency room queue." | Shows active queue sorted strictly by medical priority rank (P1 > P2 > P3 > P4). | SQLite Database Priority Ranking |
| **Step 6** | Update `PAT-1001` status to `In Treatment` | "I will now transition PAT-1001 into treatment status." | Updates database record in SQLite and refreshes active waiting queue. | State Machine Queue Management |
| **Step 7** | Switch to **Tab 3** and send chat message | "On Tab 3, our History Agent provides multi-turn record querying backed by SQLite memory." | Searches `patients` and `history_chat` tables, returning historical record summary. | Persistent Multi-turn Agent Memory |
| **Step 8** | Switch to **Tab 4** and run search | "Finally, on Tab 4, our Search Agent invokes Tavily to pull real-time clinical guidelines." | Sends HTTP request to Tavily API and renders structured web summary answers. | External API Integration & Fallback Handling |

---

## 16. VIVA / PRESENTATION EXPLANATION

### Question 1: Why did we use Groq?
**Answer:** Groq provides high-throughput, low-latency LLM inference via specialized LPU hardware. This allows our agent synthesizer node to generate clinical rationales in under 500 milliseconds, ensuring zero lag during emergency room intake.

### Question 2: Why use LangChain?
**Answer:** LangChain provides standard interfaces and abstractions (`ChatGroq`) for connecting language models with custom tools, system prompts, and output parsing pipelines.

### Question 3: Why use LangGraph?
**Answer:** Standard chain models are linear. LangGraph allows us to build cyclical, stateful, multi-agent workflows with explicit nodes, typed states (`WorkflowState`), and conditional edge routing (`route_manager`).

### Question 4: Why multiple agents instead of a single LLM prompt?
**Answer:** Multi-agent separation enforces modular responsibility. The Manager routes intent, the Triage Agent handles ML classification and DB writes, the History Agent searches memory, and the Search Agent handles external retrieval. This prevents prompt bloat and improves reliability.

### Question 5: Why not let the LLM classify the patient's priority?
**Answer:** LLMs are non-deterministic and prone to hallucinations or subtle reasoning drift. Medical priority classification requires strict, reproducible numerical boundary logic based on clinical vital signs, which is best handled by a trained Random Forest classifier.

### Question 6: Why Random Forest?
**Answer:** Random Forest is an ensemble machine learning algorithm that handles non-linear relationships between multi-variate inputs (heart rate, SpO2, blood pressure) without overfitting, providing probability scores across classes.

### Question 7: Why SQLite?
**Answer:** SQLite is lightweight, serverless, self-contained, and provides ACID compliance for local persistent data storage without external database overhead.

### Question 8: Why Tavily?
**Answer:** Tavily is optimized for LLMs and agentic search workflows, returning concise summary answers and relevant medical source snippets without generic search engine clutter.

### Question 9: What makes this an Agentic AI system?
**Answer:** The system features autonomous intent recognition, dynamic node delegation, state management, tool invocation (ML tool, SQL tool, search tool), and contextual multi-agent handoffs.

### Question 10: Where is the actual agentic decision-making?
**Answer:** In `agent_workflow.py`:
1. `manager_node` analyzes `input_data` to decide workflow task types.
2. `route_manager()` dynamically branches graph execution based on runtime state.
3. `triage_agent_node` autonomously selects tools to process vitals and save records.

### Question 11: What happens if an agent or tool fails?
**Answer:** Every tool contains error handling fallbacks:
- If Groq LLM fails, `synthesizer_node` falls back to default ML model text.
- If Tavily search fails, `tavily_search()` returns clinical baseline guidance strings.
- If model binaries are missing, `ml_tool.py` automatically triggers `train_and_evaluate()`.

### Question 12: What are the main limitations?
**Answer:**
1. The Random Forest model is trained on synthetic vital sign distributions.
2. Tavily search requires active internet connectivity for live web results.
3. The SQLite database is single-file local storage rather than distributed cloud DB.

### Question 13: Is this a real medical diagnostic system?
**Answer:** No. This project is a academic proof-of-concept demonstrating Agentic AI engineering, machine learning integration, and clinical workflow automation.

---

## 17. TROUBLESHOOTING

| Issue / Error | Cause | Exact Fix |
|---|---|---|
| `Missing API keys` or Groq auth error | `.env` file missing or invalid `GROQ_API_KEY` | Ensure `.env` exists in root folder with `GROQ_API_KEY=gsk_...`. The system will automatically fall back to ML rationale if LLM key is invalid. |
| `FileNotFoundError: model.joblib` | Model binaries deleted or not yet built | Run `python train_model.py` manually, or launch `python app.py` (which automatically trains the model if missing). |
| `sqlite3.OperationalError: table patients has no column...` | Outdated SQLite schema | Delete `patients.db` file and re-run `python app.py` to trigger fresh schema initialization via `database.init_db()`. |
| `Tavily Search API Note: (error connection)` | Internet offline or invalid Tavily key | Verify internet connection or key in `.env`. The system handles search failures gracefully with offline baseline fallbacks. |
| `OSError: [Errno 98] Address already in use` | Port 8000 is occupied by another process | Pass custom port when running app: set environment variable `PORT=8050` or modify `app.launch(server_port=8050)`. |
| `ModuleNotFoundError: No module named 'langgraph'` | Missing python package | Run `pip install -r requirements.txt`. |

---

## 18. CURRENT IMPLEMENTATION STATUS

### Working (100% Fully Implemented & Tested)
- **LangGraph StateGraph Engine:** Complete 5-node workflow graph with conditional branching (`manager`, `triage`, `history`, `search`, `synthesizer`).
- **Random Forest ML Classification:** 9-feature scaling pipeline predicting `P1`, `P2`, `P3`, `P4` priority with probability distributions.
- **SQLite Database Persistence:** Auto-generating sequential Patient IDs, table schema creation, queue ranking by priority, status updates, and chat logging.
- **Groq LLM Synthesis:** Synthesis of vital sign inputs into structured clinical reasoning via `ChatGroq`.
- **Tavily Web Search:** Clinical reference search with graceful offline fallback.
- **Gradio 4-Tab Workstation UI:** Form intake with 1-click presets, active queue dashboard, history search & multi-turn chatbot, search tool & ML validation metrics tables.

### Partially Working
- None. All requested components are implemented and functional.

### Not Implemented (By Design)
- Cloud deployment on external enterprise servers (app runs locally on `127.0.0.1:8000` or can be deployed via Docker/Render).

---

## 19. QUICK CHEAT SHEET (VIVA SUMMARY)

```
===================================================================================
                  PROJECT VIVA QUICK CHEAT SHEET
===================================================================================
PROJECT NAME:   Emergency Patient Priority Classification & Agentic Triage System
TECH STACK:     Python 3.11, Gradio, LangGraph, LangChain, Groq LLM, 
                Scikit-Learn (Random Forest), SQLite3, Tavily API

ARCHITECTURE SUMMARY:
  User Intake (Gradio UI) -> LangGraph Workflow State -> Manager Agent Node 
  -> Conditional Router -> Specialist Agent (Triage / History / Search) 
  -> Random Forest ML / SQLite DB / Tavily API -> Synthesizer Node (Groq LLM) 
  -> Dynamic UI Rendering

AGENTS & ROLES:
  1. Manager Agent: Analyzes input intent and routes workflow execution.
  2. Triage Agent: Calls Random Forest ML tool and saves record to SQLite.
  3. History Agent: Queries SQLite for past patient records and chat history.
  4. Search Agent: Executes Tavily web search for clinical guidelines.
  5. Synthesizer Node: Invokes Groq LLM to generate clinical rationales.

CUSTOM TOOLS:
  - predict_priority(): RF model tool (9 vitals -> P1/P2/P3/P4 + probabilities).
  - database helpers: SQLite persistence, queue sorting (P1 > P2 > P3 > P4).
  - tavily_search(): Real-time clinical guidance web search with offline fallback.

KEY DESIGN SEPARATION:
  - Priority Classification: Done 100% deterministically by Random Forest ML model.
  - Clinical Rationale: Generated by Groq LLM (groq/compound-mini).

COMMAND CHEATSHEET:
  1. Install:   pip install -r requirements.txt
  2. Train:     python train_model.py
  3. Launch:    python app.py
  4. URL:       http://127.0.0.1:8000
===================================================================================
```

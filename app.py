import os
import json
import pandas as pd
import gradio as gr

import database
from ml_tool import load_ml_components
from agent_workflow import run_triage_workflow
from search_tool import tavily_search

database.init_db()
load_ml_components()

def handle_triage(name, age, heart_rate, spo2, blood_pressure, temperature, respiratory_rate, pain_level, symptoms):
    input_data = {
        'task_type': 'triage',
        'name': name or "Anonymous Patient",
        'age': int(age),
        'heart_rate': int(heart_rate),
        'spo2': int(spo2),
        'blood_pressure': str(blood_pressure),
        'temperature': float(temperature),
        'respiratory_rate': int(respiratory_rate),
        'pain_level': int(pain_level),
        'symptoms': str(symptoms)
    }
    
    res = run_triage_workflow(input_data)
    
    p_code = res.get('priority', 'P4')
    p_label = res.get('priority_label', 'P4 - Low')
    conf = res.get('confidence', 100.0)
    pid = res.get('patient_id', 'N/A')
    analysis = res.get('llm_analysis', 'Random Forest model evaluated vital signs.')
    trace = "\n".join([f"> {line}" for line in res.get('workflow_trace', [])])
    
    badge_html = f"""
    <div style="padding:16px; border-radius:6px; background-color:{'#fef2f2' if p_code=='P1' else '#fff7ed' if p_code=='P2' else '#fefce8' if p_code=='P3' else '#f0fdf4'}; border:1px solid {'#fca5a5' if p_code=='P1' else '#fdba74' if p_code=='P2' else '#fde68a' if p_code=='P3' else '#99f6e4'};">
        <div style="display:flex; align-items:center; gap:14px;">
            <div style="width:48px; height:48px; border-radius:6px; background-color:{'#b91c1c' if p_code=='P1' else '#c2410c' if p_code=='P2' else '#b45309' if p_code=='P3' else '#0f766e'}; color:#fff; font-size:22px; font-weight:800; display:flex; align-items:center; justify-content:center;">
                {p_code}
            </div>
            <div>
                <h3 style="margin:0; font-size:16px; font-weight:700; color:#0f172a;">{p_label}</h3>
                <div style="font-size:12px; color:#475569; margin-top:2px;">
                    Assigned Patient ID: <strong>{pid}</strong> | ML Classification Confidence: <strong>{conf}%</strong>
                </div>
            </div>
        </div>
    </div>
    """
    
    probs = res.get('probabilities', {})
    prob_str = f"P1 (Critical): {probs.get('P1', 0)}%  |  P2 (High): {probs.get('P2', 0)}%  |  P3 (Moderate): {probs.get('P3', 0)}%  |  P4 (Low): {probs.get('P4', 0)}%"
    
    return badge_html, analysis, prob_str, trace, get_queue_table(), get_summary_html()

def load_preset_p1():
    return "Patient A", 64, 148, 83, "80/50", 39.2, 34, 10, "Severe acute crushing chest pain, dyspnea, cyanosis, syncope"

def load_preset_p2():
    return "Patient B", 47, 122, 91, "165/100", 38.7, 26, 8, "High fever, acute right lower quadrant abdominal pain, persistent vomiting"

def load_preset_p4():
    return "Patient C", 28, 72, 99, "118/75", 36.7, 14, 2, "Minor superficial cut on right index finger, routine check"

def get_queue_table():
    queue = database.get_emergency_queue()
    if not queue:
        return pd.DataFrame(columns=["Rank", "Patient ID", "Name", "Age", "Priority", "SpO2", "HR", "BP", "Status", "Timestamp"])
        
    data = []
    for idx, p in enumerate(queue):
        data.append({
            "Rank": f"#{idx + 1}",
            "Patient ID": p['patient_id'],
            "Name": p['name'],
            "Age": p['age'],
            "Priority": p['priority'],
            "SpO2": f"{p['spo2']}%",
            "HR": f"{p['heart_rate']} bpm",
            "BP": p['blood_pressure'],
            "Status": p['queue_status'],
            "Timestamp": p['timestamp']
        })
    return pd.DataFrame(data)

def get_history_table(query=""):
    records = database.get_patient_history(query)
    if not records:
        return pd.DataFrame(columns=["Patient ID", "Name", "Age", "Priority", "Confidence", "SpO2 / HR / BP", "Symptoms", "Status", "Timestamp"])
        
    data = []
    for p in records:
        data.append({
            "Patient ID": p['patient_id'],
            "Name": p['name'],
            "Age": p['age'],
            "Priority": p['priority'],
            "Confidence": f"{p['confidence']}%",
            "SpO2 / HR / BP": f"{p['spo2']}% / {p['heart_rate']} bpm / {p['blood_pressure']}",
            "Symptoms": p['symptoms'],
            "Status": p['queue_status'],
            "Timestamp": p['timestamp']
        })
    return pd.DataFrame(data)

def get_summary_html():
    summary = database.get_dashboard_summary()
    return f"""
    <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:12px; margin-bottom:12px;">
        <div style="background:#fff; border:1px solid #cbd5e1; border-radius:6px; padding:10px 14px;">
            <div style="font-size:10px; font-weight:700; color:#64748b; text-transform:uppercase;">Active Waiting Queue</div>
            <div style="font-size:20px; font-weight:800; color:#0f172a;">{summary.get('waiting_count', 0)}</div>
        </div>
        <div style="background:#fff; border:1px solid #cbd5e1; border-top:3px solid #b91c1c; border-radius:6px; padding:10px 14px;">
            <div style="font-size:10px; font-weight:700; color:#64748b; text-transform:uppercase;">P1 Critical</div>
            <div style="font-size:20px; font-weight:800; color:#b91c1c;">{summary.get('p1_count', 0)}</div>
        </div>
        <div style="background:#fff; border:1px solid #cbd5e1; border-top:3px solid #c2410c; border-radius:6px; padding:10px 14px;">
            <div style="font-size:10px; font-weight:700; color:#64748b; text-transform:uppercase;">P2 High</div>
            <div style="font-size:20px; font-weight:800; color:#c2410c;">{summary.get('p2_count', 0)}</div>
        </div>
        <div style="background:#fff; border:1px solid #cbd5e1; border-radius:6px; padding:10px 14px;">
            <div style="font-size:10px; font-weight:700; color:#64748b; text-transform:uppercase;">Total Triaged Patients</div>
            <div style="font-size:20px; font-weight:800; color:#0284c7;">{summary.get('total_triaged', 0)}</div>
        </div>
    </div>
    """

def handle_update_status(patient_id, new_status):
    if not patient_id:
        return "Please enter a valid Patient ID.", get_queue_table(), get_summary_html()
    success = database.update_patient_status(patient_id.strip(), new_status)
    msg = f"Patient {patient_id} status updated to '{new_status}'." if success else f"Patient ID {patient_id} not found."
    return msg, get_queue_table(), get_summary_html()

def handle_history_chat(message, history):
    if not message or not message.strip():
        return "", history
        
    records = database.get_patient_history(message)
    if records:
        rec_summary = []
        for r in records[:3]:
            rec_summary.append(f"Patient {r['patient_id']} ({r['name']}): Priority {r['priority']}, SpO2 {r['spo2']}%, HR {r['heart_rate']} bpm, BP {r['blood_pressure']}, Symptoms: {r['symptoms']}, Status: {r['queue_status']}.")
        reply = "SQLite History Tool Results:\n" + "\n".join(rec_summary)
    else:
        reply = f"SQLite History Agent: No historical triage records found matching query '{message}'."
        
    history = history or []
    history.append((message, reply))
    database.save_chat_message("default", "user", message)
    database.save_chat_message("default", "assistant", reply)
    return "", history

def handle_tavily_search(query):
    if not query or not query.strip():
        return "Please enter a search topic."
    return tavily_search(query)

def get_metrics_data():
    if os.path.exists("model_metrics.json"):
        with open("model_metrics.json", "r") as f:
            data = json.load(f)
            acc = data.get("accuracy", 1.0)
            prec = data.get("precision", 1.0)
            rec = data.get("recall", 1.0)
            f1 = data.get("f1_score", 1.0)
            
            metrics_summary = f"Accuracy: {acc*100:.1f}%  |  Precision: {prec*100:.1f}%  |  Recall: {rec*100:.1f}%  |  F1-Score: {f1*100:.1f}%"
            
            details = data.get("details", {})
            rows = []
            for cls in ['P1', 'P2', 'P3', 'P4']:
                if cls in details:
                    r = details[cls]
                    rows.append({
                        "Class": cls,
                        "Precision": f"{r.get('precision', 0)*100:.1f}%",
                        "Recall": f"{r.get('recall', 0)*100:.1f}%",
                        "F1-Score": f"{r.get('f1-score', 0)*100:.1f}%",
                        "Support": r.get('support', 0)
                    })
            return metrics_summary, pd.DataFrame(rows)
    return "Metrics file not found.", pd.DataFrame()

# Gradio Theme & Custom CSS for Emergency Department Workstation Styling
custom_css = """
body { font-family: 'Inter', -apple-system, sans-serif; background-color: #f1f5f9; color: #0f172a; }
.header-box { background-color: #0f172a; color: #ffffff; padding: 12px 18px; border-radius: 6px; margin-bottom: 12px; }
.header-box h1 { margin: 0; font-size: 18px; font-weight: 700; color: #ffffff; }
.header-box p { margin: 2px 0 0 0; font-size: 12px; color: #94a3b8; }
.gr-button-primary { background-color: #0284c7 !important; border: none !important; color: #ffffff !important; }
.gr-button-primary:hover { background-color: #0369a1 !important; }
"""

with gr.Blocks(title="Emergency Patient Priority Classification System", css=custom_css, theme=gr.themes.Soft()) as app:
    gr.HTML("""
    <div class="header-box">
        <h1>Emergency Patient Priority Classification System</h1>
        <p>LangGraph Agentic Multi-Agent Workflow Engine | Random Forest ML Tool | Tavily Search Tool | SQLite Persistent Memory</p>
    </div>
    """)
    
    summary_html = gr.HTML(value=get_summary_html())
    
    with gr.Tabs():
        # TAB 1: Patient Intake & Triage Analysis
        with gr.TabItem("Patient Intake & Triage Analysis"):
            with gr.Row():
                # Left Column: Vitals Form
                with gr.Column(scale=1):
                    gr.Markdown("### Patient Clinical Intake Form")
                    
                    with gr.Row():
                        preset_p1 = gr.Button("Preset: P1 Critical", size="sm")
                        preset_p2 = gr.Button("Preset: P2 High", size="sm")
                        preset_p4 = gr.Button("Preset: P4 Low", size="sm")
                        
                    with gr.Row():
                        name_in = gr.Textbox(label="Full Patient Name", value="Patient A", scale=2)
                        age_in = gr.Number(label="Age", value=64, precision=0, scale=1)
                        
                    with gr.Row():
                        hr_in = gr.Number(label="Heart Rate (bpm)", value=148, precision=0)
                        spo2_in = gr.Number(label="SpO2 (%)", value=83, precision=0)
                        bp_in = gr.Textbox(label="Blood Pressure", value="80/50")
                        
                    with gr.Row():
                        temp_in = gr.Number(label="Temp (°C)", value=39.2)
                        rr_in = gr.Number(label="Resp Rate (bpm)", value=34, precision=0)
                        pain_in = gr.Number(label="Pain Score (1-10)", value=10, precision=0)
                        
                    symptoms_in = gr.Textbox(label="Symptoms & Clinical Notes", value="Severe acute crushing chest pain, dyspnea, cyanosis, syncope", lines=3)
                    
                    btn_triage = gr.Button("Run LangGraph Agent Triage Workflow", variant="primary")
                    
                # Right Column: Analysis & Agent Output
                with gr.Column(scale=1):
                    gr.Markdown("### LangGraph Agent Triage Result")
                    result_badge = gr.HTML(value="""<div style="padding:16px; border:1px dashed #cbd5e1; border-radius:6px; text-align:center; color:#64748b;">Submit patient intake form to execute Manager, Triage, and Specialist agent workflow graph.</div>""")
                    llm_analysis_out = gr.Textbox(label="Manager Agent Clinical Rationale", lines=3)
                    probs_out = gr.Textbox(label="Random Forest Class Probabilities")
                    trace_out = gr.Textbox(label="LangGraph Workflow Execution Trace Log", lines=6)
                    
        # TAB 2: Emergency Queue Dashboard
        with gr.TabItem("Emergency Queue Dashboard"):
            gr.Markdown("### Active Emergency Triage Queue (Ordered by Priority P1 > P2 > P3 > P4)")
            queue_table_df = gr.Dataframe(value=get_queue_table(), interactive=False)
            
            with gr.Row():
                patient_id_input = gr.Textbox(label="Patient ID", placeholder="PAT-1001", scale=2)
                status_dropdown = gr.Dropdown(label="New Status", choices=["Waiting", "In Treatment", "Discharged"], value="In Treatment", scale=2)
                btn_update_status = gr.Button("Update Status", scale=1)
                btn_refresh_q = gr.Button("Refresh Queue", scale=1)
                
            status_msg_out = gr.Textbox(label="Status Update Response")

        # TAB 3: Patient History & Multi-turn Chat
        with gr.TabItem("Patient History & History Agent Chat"):
            gr.Markdown("### SQLite Patient Records Log & Multi-turn History Agent")
            
            with gr.Row():
                search_history_in = gr.Textbox(label="Search History Records", placeholder="Enter name or patient ID...", scale=3)
                btn_search_history = gr.Button("Search Records", scale=1)
                
            history_table_df = gr.Dataframe(value=get_history_table(), interactive=False)
            
            gr.Markdown("---")
            gr.Markdown("### Multi-turn Patient History Chatbot (SQLite Memory)")
            chatbot = gr.Chatbot(label="History Agent Conversation")
            chat_msg_in = gr.Textbox(label="Ask History Agent about previous patient records...", placeholder="e.g. Find records for Patient A")
            btn_send_chat = gr.Button("Send Query to History Agent", variant="primary")

        # TAB 4: Tavily Medical Search & ML Analytics
        with gr.TabItem("Tavily Search & ML Diagnostics"):
            gr.Markdown("### Tavily Clinical Search Tool (Search Agent)")
            with gr.Row():
                tavily_query_in = gr.Textbox(label="Clinical Search Topic", placeholder="e.g. Emergency triage protocols for chest pain and low SpO2", scale=3)
                btn_run_search = gr.Button("Run Tavily Search", variant="primary", scale=1)
            search_out = gr.Textbox(label="Tavily Search Tool Output", lines=5)
            
            gr.Markdown("---")
            gr.Markdown("### Random Forest Classifier Evaluation Metrics")
            metrics_str_out, metrics_table_df = get_metrics_data()
            metrics_summary_out = gr.Textbox(label="Model Evaluation Summary", value=metrics_str_out)
            metrics_df_out = gr.Dataframe(value=metrics_table_df, interactive=False)

    # Event Handlers
    preset_p1.click(load_preset_p1, outputs=[name_in, age_in, hr_in, spo2_in, bp_in, temp_in, rr_in, pain_in, symptoms_in])
    preset_p2.click(load_preset_p2, outputs=[name_in, age_in, hr_in, spo2_in, bp_in, temp_in, rr_in, pain_in, symptoms_in])
    preset_p4.click(load_preset_p4, outputs=[name_in, age_in, hr_in, spo2_in, bp_in, temp_in, rr_in, pain_in, symptoms_in])
    
    btn_triage.click(
        handle_triage,
        inputs=[name_in, age_in, hr_in, spo2_in, bp_in, temp_in, rr_in, pain_in, symptoms_in],
        outputs=[result_badge, llm_analysis_out, probs_out, trace_out, queue_table_df, summary_html]
    )
    
    btn_update_status.click(
        handle_update_status,
        inputs=[patient_id_input, status_dropdown],
        outputs=[status_msg_out, queue_table_df, summary_html]
    )
    
    btn_refresh_q.click(get_queue_table, outputs=[queue_table_df])
    
    btn_search_history.click(get_history_table, inputs=[search_history_in], outputs=[history_table_df])
    
    btn_send_chat.click(
        handle_history_chat,
        inputs=[chat_msg_in, chatbot],
        outputs=[chat_msg_in, chatbot]
    )
    
    btn_run_search.click(handle_tavily_search, inputs=[tavily_query_in], outputs=[search_out])

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.launch(server_name="0.0.0.0", server_port=port)

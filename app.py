import os
import json
import pandas as pd
import gradio as gr

import database
from ml_tool import load_ml_components
from agent_workflow import run_triage_workflow
from search_tool import tavily_search

# Initialize database tables and load machine learning model artifacts
database.init_db()
load_ml_components()


def handle_triage(name, age, heart_rate, spo2, blood_pressure, temperature, respiratory_rate, pain_level, symptoms):
    input_data = {
        'task_type': 'triage',
        'name': name or "Anonymous Patient",
        'age': int(age) if age is not None else 40,
        'heart_rate': int(heart_rate) if heart_rate is not None else 75,
        'spo2': int(spo2) if spo2 is not None else 98,
        'blood_pressure': str(blood_pressure or '120/80'),
        'temperature': float(temperature) if temperature is not None else 37.0,
        'respiratory_rate': int(respiratory_rate) if respiratory_rate is not None else 16,
        'pain_level': int(pain_level) if pain_level is not None else 2,
        'symptoms': str(symptoms or 'None reported')
    }
    
    res = run_triage_workflow(input_data)
    
    p_code = res.get('priority', 'P4')
    p_label = res.get('priority_label', 'P4 - Low')
    conf = res.get('confidence', 100.0)
    pid = res.get('patient_id', 'N/A')
    analysis = res.get('llm_analysis', 'Random Forest model evaluated vital signs against emergency triage standards.')
    trace = "\n".join([f"> {line}" for line in res.get('workflow_trace', [])])
    
    badge_html = f"""
    <div style="padding:16px; border-radius:8px; background-color:{'#fef2f2' if p_code=='P1' else '#fff7ed' if p_code=='P2' else '#fefce8' if p_code=='P3' else '#ecfdf5'}; border:1px solid {'#fca5a5' if p_code=='P1' else '#fdba74' if p_code=='P2' else '#fde68a' if p_code=='P3' else '#a7f3d0'}; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
        <div style="display:flex; align-items:center; gap:14px;">
            <div style="width:52px; height:52px; border-radius:8px; background-color:{'#dc2626' if p_code=='P1' else '#ea580c' if p_code=='P2' else '#d97706' if p_code=='P3' else '#059669'}; color:#ffffff; font-size:24px; font-weight:800; display:flex; align-items:center; justify-content:center;">
                {p_code}
            </div>
            <div>
                <h3 style="margin:0; font-size:18px; font-weight:700; color:#0f172a;">{p_label}</h3>
                <div style="font-size:13px; color:#475569; margin-top:3px;">
                    Assigned Patient ID: <strong style="color:#0f172a;">{pid}</strong> | ML Classification Confidence: <strong style="color:#0f172a;">{conf}%</strong>
                </div>
            </div>
        </div>
    </div>
    """
    
    return badge_html, analysis, trace, get_queue_table(), get_summary_html()


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
    <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:14px; margin-bottom:16px;">
        <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:14px 18px; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.5px;">Active Waiting Queue</div>
            <div style="font-size:24px; font-weight:800; color:#0f172a; margin-top:4px;">{summary.get('waiting_count', 0)}</div>
        </div>
        <div style="background:#ffffff; border:1px solid #e2e8f0; border-top:4px solid #dc2626; border-radius:8px; padding:14px 18px; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.5px;">P1 Critical</div>
            <div style="font-size:24px; font-weight:800; color:#dc2626; margin-top:4px;">{summary.get('p1_count', 0)}</div>
        </div>
        <div style="background:#ffffff; border:1px solid #e2e8f0; border-top:4px solid #ea580c; border-radius:8px; padding:14px 18px; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.5px;">P2 High</div>
            <div style="font-size:24px; font-weight:800; color:#ea580c; margin-top:4px;">{summary.get('p2_count', 0)}</div>
        </div>
        <div style="background:#ffffff; border:1px solid #e2e8f0; border-top:4px solid #0284c7; border-radius:8px; padding:14px 18px; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.5px;">Total Triaged Patients</div>
            <div style="font-size:24px; font-weight:800; color:#0284c7; margin-top:4px;">{summary.get('total_triaged', 0)}</div>
        </div>
    </div>
    """


def handle_update_status(patient_id, new_status):
    if not patient_id or not patient_id.strip():
        return "Please enter a valid Patient ID.", get_queue_table(), get_summary_html()
    success = database.update_patient_status(patient_id.strip(), new_status)
    msg = f"Patient {patient_id.strip()} status updated to '{new_status}'." if success else f"Patient ID {patient_id.strip()} not found."
    return msg, get_queue_table(), get_summary_html()


def handle_tavily_search(query):
    if not query or not query.strip():
        return "Please enter a clinical search topic or guideline query."
    return tavily_search(query.strip())


# Strict Light Theme & Clinical Color Palette CSS Overrides
custom_css = """
/* Force Light Theme CSS Variables across Gradio */
:root, html, body, .gradio-container, .dark, .dark * {
    --body-background-fill: #f8fafc !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f1f5f9 !important;
    --border-color-primary: #cbd5e1 !important;
    --block-background-fill: #ffffff !important;
    --block-border-color: #cbd5e1 !important;
    --block-title-text-color: #0f172a !important;
    --block-label-text-color: #334155 !important;
    --body-text-color: #0f172a !important;
    --body-text-color-subdued: #475569 !important;
    --input-background-fill: #ffffff !important;
    --input-placeholder-color: #94a3b8 !important;
    --table-text-color: #0f172a !important;
    --table-body-background-fill: #ffffff !important;
    --table-row-odd-background-fill: #ffffff !important;
    --table-row-even-background-fill: #f8fafc !important;
    --table-row-focus-background-fill: #e0f2fe !important;
    --table-cell-background-fill: #ffffff !important;
    --cell-background-fill: #ffffff !important;
    --table-border-color: #e2e8f0 !important;
    color-scheme: light !important;
}

body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #f8fafc !important;
    color: #0f172a !important;
}

/* Blocks, Panels & Containers */
.block, .panel, .form, div[class*="block"], div[class*="panel"], div.cell {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
}

/* Headings, Labels & Text */
h1, h2, h3, h4, h5, h6, .markdown {
    color: #0f172a !important;
}

label, span.label, label span, .block label {
    background: transparent !important;
    color: #0f172a !important;
    font-weight: 600 !important;
}

/* Text Inputs, Textareas, Numbers & Select Dropdowns */
input, textarea, select, input[type="text"], input[type="number"], .gr-input, .gr-box, fieldset {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
}

input:focus, textarea:focus {
    border-color: #0284c7 !important;
    box-shadow: 0 0 0 2px rgba(2, 132, 199, 0.2) !important;
}

/* Dataframe & Table Specific Overrides for 100% Light Contrast */
.dataframe, .dataframe *, table, table *, tbody, tbody *, tr, tr *, td, td *, th, th * {
    color: #0f172a !important;
}

table, .dataframe, div.table-wrap, div.table-container {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
}

table tbody tr, table tbody tr td, div.tbody div.tr, div.tbody div.td, div.cell-wrap, .cell-wrap, td.cell-wrap {
    background-color: #ffffff !important;
    color: #0f172a !important;
}

table tbody tr:nth-child(even), table tbody tr:nth-child(even) td {
    background-color: #f8fafc !important;
    color: #0f172a !important;
}

table tbody tr:nth-child(odd), table tbody tr:nth-child(odd) td {
    background-color: #ffffff !important;
    color: #0f172a !important;
}

table tbody tr:hover, table tbody tr:hover td {
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
}

th, thead th, div.thead div.th {
    background-color: #f1f5f9 !important;
    color: #0369a1 !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #cbd5e1 !important;
}

td {
    color: #0f172a !important;
    border-bottom: 1px solid #e2e8f0 !important;
}

td div, td span, td p, div.cell-wrap span, div.cell-wrap input {
    color: #0f172a !important;
    background-color: transparent !important;
}

/* Primary Buttons with Crisp White Text */
button.primary, button.primary *, .gr-button-primary, .gr-button-primary *, button.variant-primary, button.variant-primary * {
    background-color: #0284c7 !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
}

button.primary:hover, button.primary:hover *, .gr-button-primary:hover, .gr-button-primary:hover * {
    background-color: #0369a1 !important;
    color: #ffffff !important;
}

/* Secondary Buttons */
button.secondary, .gr-button-secondary, button.sm {
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    font-weight: 500 !important;
}

button.secondary:hover, .gr-button-secondary:hover {
    background-color: #e2e8f0 !important;
}

/* Header Box */
.header-box {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-top: 4px solid #0284c7 !important;
    border-radius: 8px !important;
    padding: 18px 24px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
}

.header-box h1 {
    margin: 0 !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #0369a1 !important;
    letter-spacing: -0.3px !important;
}

.header-box p {
    margin: 4px 0 0 0 !important;
    font-size: 13px !important;
    color: #475569 !important;
}

/* Tabs */
.tabs button.selected, button.tab-nav.selected {
    border-bottom: 3px solid #0284c7 !important;
    color: #0284c7 !important;
    font-weight: 700 !important;
    background-color: #ffffff !important;
}

.tabs button, button.tab-nav {
    color: #475569 !important;
    font-weight: 500 !important;
}

/* Hide Footers */
footer, .gradio-container footer, footer.svelte-12822xp, footer.svelte-15w2w4p, .api-docs-btn, button.api-docs-btn, .built-with {
    display: none !important;
}
"""

with gr.Blocks(title="Emergency Patient Priority Classification & Agentic Triage System") as app:
    gr.HTML("""
    <div class="header-box">
        <h1>Emergency Patient Priority Classification & Agentic Triage System</h1>
        <p>LangGraph Multi-Agent Triage Workflow | Random Forest Vital Signs Classification | SQLite Records & Medical Guidelines</p>
    </div>
    """)
    
    with gr.Tabs():
        # TAB 1: Patient Intake & Triage
        with gr.TabItem("Patient Intake & Triage"):
            with gr.Row():
                # Left Column: Form
                with gr.Column(scale=1):
                    gr.Markdown("### Patient Intake Form")
                    
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
                    
                    btn_triage = gr.Button("Run Triage Workflow", variant="primary")
                    
                # Right Column: Output
                with gr.Column(scale=1):
                    gr.Markdown("### Triage Output & Clinical Analysis")
                    result_badge = gr.HTML(value="""<div style="padding:20px; border:1px dashed #cbd5e1; border-radius:8px; text-align:center; color:#64748b; font-size:14px; background:#ffffff;">Submit the intake form to trigger priority classification and agentic clinical evaluation.</div>""")
                    llm_analysis_out = gr.Textbox(label="Groq LLM Clinical Rationale", lines=4)
                    trace_out = gr.Textbox(label="LangGraph Execution Trace Log", lines=8)
                    
        # TAB 2: Emergency Queue Dashboard
        with gr.TabItem("Emergency Queue Dashboard"):
            summary_html = gr.HTML(value=get_summary_html())
            
            gr.Markdown("### Active Emergency Queue (Sorted by Priority: P1 > P2 > P3 > P4)")
            queue_table_df = gr.Dataframe(value=get_queue_table(), interactive=False)
            
            with gr.Row():
                patient_id_input = gr.Textbox(label="Patient ID", placeholder="PAT-1001", scale=2)
                status_dropdown = gr.Dropdown(label="New Status", choices=["Waiting", "In Treatment", "Discharged"], value="In Treatment", scale=2)
                btn_update_status = gr.Button("Update Status", scale=1)
                btn_refresh_q = gr.Button("Refresh Queue", scale=1)
                
            status_msg_out = gr.Textbox(label="Status Update Response")

        # TAB 3: Patient Records & Guidelines Search
        with gr.TabItem("Patient Records & Guidelines Search"):
            gr.Markdown("### Patient Records Search (SQLite Database)")
            with gr.Row():
                search_history_in = gr.Textbox(label="Search Patient Records", placeholder="Enter Patient Name or ID (e.g. PAT-1001)...", scale=3)
                btn_search_history = gr.Button("Search Records", scale=1)
                
            history_table_df = gr.Dataframe(value=get_history_table(), interactive=False)
            
            gr.Markdown("---")
            gr.Markdown("### Emergency Medical Guidelines Search (Tavily API)")
            with gr.Row():
                tavily_query_in = gr.Textbox(label="Medical Guideline Query", placeholder="e.g. Triage management for chest pain and low SpO2...", scale=3)
                btn_run_search = gr.Button("Search Guidelines", variant="primary", scale=1)
            search_out = gr.Textbox(label="Tavily Guidelines Search Output", lines=8)

    # Event Handlers
    preset_p1.click(load_preset_p1, outputs=[name_in, age_in, hr_in, spo2_in, bp_in, temp_in, rr_in, pain_in, symptoms_in])
    preset_p2.click(load_preset_p2, outputs=[name_in, age_in, hr_in, spo2_in, bp_in, temp_in, rr_in, pain_in, symptoms_in])
    preset_p4.click(load_preset_p4, outputs=[name_in, age_in, hr_in, spo2_in, bp_in, temp_in, rr_in, pain_in, symptoms_in])
    
    btn_triage.click(
        handle_triage,
        inputs=[name_in, age_in, hr_in, spo2_in, bp_in, temp_in, rr_in, pain_in, symptoms_in],
        outputs=[result_badge, llm_analysis_out, trace_out, queue_table_df, summary_html]
    )
    
    btn_update_status.click(
        handle_update_status,
        inputs=[patient_id_input, status_dropdown],
        outputs=[status_msg_out, queue_table_df, summary_html]
    )
    
    btn_refresh_q.click(get_queue_table, outputs=[queue_table_df])
    
    btn_search_history.click(get_history_table, inputs=[search_history_in], outputs=[history_table_df])
    
    btn_run_search.click(handle_tavily_search, inputs=[tavily_query_in], outputs=[search_out])

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.launch(server_name="0.0.0.0", server_port=port, footer_links=[], css=custom_css, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"))

import os
import json
from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

import database
from ml_tool import predict_priority
from search_tool import tavily_search

load_dotenv()

class WorkflowState(TypedDict):
    input_data: Dict[str, Any]
    query_type: str
    triage_result: Optional[Dict[str, Any]]
    history_result: Optional[List[Dict[str, Any]]]
    search_result: Optional[str]
    agent_logs: List[str]
    final_output: Dict[str, Any]

def get_groq_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and api_key.strip() and api_key != "your_groq_api_key_here":
        for model in ["groq/compound-mini", "groq/compound", "llama-3.3-70b-versatile", "llama3-8b-8192"]:
            try:
                return ChatGroq(model=model, groq_api_key=api_key.strip(), temperature=0.2)
            except Exception:
                continue
    return None

def manager_node(state: WorkflowState) -> Dict[str, Any]:
    logs = list(state.get('agent_logs', []))
    logs.append("Manager Agent: Received request. Analyzing workflow intent.")
    
    input_data = state.get('input_data', {})
    task_type = input_data.get('task_type')
    
    if not task_type:
        if 'heart_rate' in input_data or 'spo2' in input_data:
            task_type = 'triage'
        elif 'history_query' in input_data:
            task_type = 'history'
        elif 'search_query' in input_data:
            task_type = 'search'
        else:
            task_type = 'triage'
            
    logs.append(f"Manager Agent: Delegating workflow to Specialist Agent node -> {task_type.upper()}.")
    return {'query_type': task_type, 'agent_logs': logs}

def triage_agent_node(state: WorkflowState) -> Dict[str, Any]:
    logs = list(state.get('agent_logs', []))
    logs.append("Triage Agent: Processing patient vitals. Invoking Random Forest ML tool predict_priority().")
    
    input_data = state.get('input_data', {})
    ml_res = predict_priority(input_data)
    
    logs.append(f"Triage Tool predict_priority() classified priority: {ml_res['priority_label']} ({ml_res['confidence']}% confidence).")
    
    db_patient = {
        'name': input_data.get('name', 'Anonymous Patient'),
        'age': int(input_data.get('age', 40)),
        'heart_rate': int(input_data.get('heart_rate', 75)),
        'spo2': int(input_data.get('spo2', 98)),
        'blood_pressure': str(input_data.get('blood_pressure', '120/80')),
        'temperature': float(input_data.get('temperature', 37.0)),
        'respiratory_rate': int(input_data.get('respiratory_rate', 16)),
        'pain_level': int(input_data.get('pain_level', 2)),
        'symptoms': str(input_data.get('symptoms', 'None reported')),
        'priority': ml_res['priority'],
        'confidence': ml_res['confidence'],
        'queue_status': 'Waiting'
    }
    
    patient_id = database.save_patient(db_patient)
    ml_res['patient_id'] = patient_id
    ml_res['queue_status'] = 'Waiting'
    ml_res['name'] = db_patient['name']
    
    logs.append(f"Triage Agent: Saved patient {patient_id} to SQLite database queue.")
    
    return {'triage_result': ml_res, 'agent_logs': logs}

def history_agent_node(state: WorkflowState) -> Dict[str, Any]:
    logs = list(state.get('agent_logs', []))
    logs.append("History Agent: Querying SQLite database for patient records.")
    
    input_data = state.get('input_data', {})
    query = input_data.get('history_query', input_data.get('name', ''))
    
    records = database.get_patient_history(query)
    logs.append(f"History Agent: Retrieved {len(records)} patient history records from SQLite.")
    
    return {'history_result': records, 'agent_logs': logs}

def search_agent_node(state: WorkflowState) -> Dict[str, Any]:
    logs = list(state.get('agent_logs', []))
    logs.append("Search Agent: Invoking Tavily Search Tool for clinical reference.")
    
    input_data = state.get('input_data', {})
    query = input_data.get('search_query', input_data.get('symptoms', 'emergency triage guidelines'))
    
    search_res = tavily_search(query)
    logs.append("Search Agent: Tavily clinical reference query complete.")
    
    return {'search_result': search_res, 'agent_logs': logs}

def synthesizer_node(state: WorkflowState) -> Dict[str, Any]:
    logs = list(state.get('agent_logs', []))
    logs.append("Manager Agent: Synthesizing final structured response.")
    
    triage_res = state.get('triage_result')
    history_res = state.get('history_result')
    search_res = state.get('search_result')
    
    llm = get_groq_llm()
    clinical_notes = ""
    
    if llm and triage_res:
        try:
            prompt = f"Patient vitals: {triage_res.get('features_used')}. Priority: {triage_res.get('priority_label')} ({triage_res.get('confidence')}% confidence). Symptoms: {triage_res.get('features_used', {}).get('symptoms')}. Provide a 2-sentence medical rationale."
            response = llm.invoke(prompt)
            clinical_notes = str(response.content).strip()
            logs.append("Manager Agent: Groq LLM clinical rationale generated.")
        except Exception:
            clinical_notes = "Random Forest ML model evaluated vital signs against emergency triage standards."
    else:
        clinical_notes = "Random Forest ML model evaluated vital signs against emergency triage standards."
        
    final_payload = {
        'patient_id': triage_res.get('patient_id') if triage_res else "N/A",
        'name': triage_res.get('name') if triage_res else "N/A",
        'priority': triage_res.get('priority') if triage_res else "P4",
        'priority_label': triage_res.get('priority_label') if triage_res else "P4 - Low",
        'confidence': triage_res.get('confidence') if triage_res else 100.0,
        'probabilities': triage_res.get('probabilities') if triage_res else {},
        'vitals': triage_res.get('features_used') if triage_res else {},
        'llm_analysis': clinical_notes,
        'queue_status': triage_res.get('queue_status') if triage_res else "Waiting",
        'history_records': history_res or [],
        'search_summary': search_res or "",
        'workflow_trace': logs
    }
    
    return {'final_output': final_payload, 'agent_logs': logs}

def build_agent_graph():
    builder = StateGraph(WorkflowState)
    
    builder.add_node("manager", manager_node)
    builder.add_node("triage", triage_agent_node)
    builder.add_node("history", history_agent_node)
    builder.add_node("search", search_agent_node)
    builder.add_node("synthesizer", synthesizer_node)
    
    builder.add_edge(START, "manager")
    
    def route_manager(state: WorkflowState):
        qtype = state.get('query_type', 'triage')
        if qtype == 'history':
            return "history"
        elif qtype == 'search':
            return "search"
        else:
            return "triage"
            
    builder.add_conditional_edges("manager", route_manager, {
        "triage": "triage",
        "history": "history",
        "search": "search"
    })
    
    builder.add_edge("triage", "synthesizer")
    builder.add_edge("history", "synthesizer")
    builder.add_edge("search", "synthesizer")
    builder.add_edge("synthesizer", END)
    
    return builder.compile()

graph_app = build_agent_graph()

def run_triage_workflow(patient_data):
    initial_state = {
        'input_data': patient_data,
        'query_type': patient_data.get('task_type', 'triage'),
        'agent_logs': []
    }
    result_state = graph_app.invoke(initial_state)
    return result_state['final_output']

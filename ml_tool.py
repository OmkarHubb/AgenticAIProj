import os
import joblib
import numpy as np
import pandas as pd
from train_model import train_and_evaluate

MODEL_PATH = "model.joblib"
SCALER_PATH = "scaler.joblib"

_model = None
_scaler = None

def load_ml_components():
    global _model, _scaler
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        train_and_evaluate()
        
    _model = joblib.load(MODEL_PATH)
    _scaler = joblib.load(SCALER_PATH)

def estimate_symptom_severity(symptoms_text, pain_level):
    text = str(symptoms_text).lower() if symptoms_text else ""
    
    critical_keywords = ['chest pain', 'cardiac', 'unresponsive', 'stroke', 'severe bleeding', 'seizure', 'respiratory arrest', 'cyanosis', 'anaphylaxis']
    severe_keywords = ['high fever', 'fracture', 'shortness of breath', 'severe abdominal pain', 'confusion', 'dislocated', 'asthma attack']
    moderate_keywords = ['moderate fever', 'vomiting', 'dizziness', 'sprain', 'deep cut', 'persistent cough', 'migraine']
    
    if any(k in text for k in critical_keywords) or pain_level >= 9:
        return 4
    elif any(k in text for k in severe_keywords) or pain_level >= 7:
        return 3
    elif any(k in text for k in moderate_keywords) or pain_level >= 4:
        return 2
    else:
        return 1

def parse_blood_pressure(bp_string):
    try:
        parts = str(bp_string).strip().split('/')
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return 120, 80

def predict_priority(patient_vitals):
    global _model, _scaler
    if _model is None or _scaler is None:
        load_ml_components()
        
    age = int(patient_vitals.get('age', 40))
    heart_rate = int(patient_vitals.get('heart_rate', 75))
    spo2 = int(patient_vitals.get('spo2', 98))
    
    bp_raw = patient_vitals.get('blood_pressure', '120/80')
    if isinstance(bp_raw, str):
        bp_sys, bp_dia = parse_blood_pressure(bp_raw)
    else:
        bp_sys = int(patient_vitals.get('bp_sys', 120))
        bp_dia = int(patient_vitals.get('bp_dia', 80))
        
    temperature = float(patient_vitals.get('temperature', 37.0))
    respiratory_rate = int(patient_vitals.get('respiratory_rate', 16))
    pain_level = int(patient_vitals.get('pain_level', 2))
    
    symptoms = patient_vitals.get('symptoms', '')
    symptom_severity = patient_vitals.get('symptom_severity')
    if symptom_severity is None:
        symptom_severity = estimate_symptom_severity(symptoms, pain_level)
    else:
        symptom_severity = int(symptom_severity)
        
    feature_cols = ['age', 'heart_rate', 'spo2', 'bp_sys', 'bp_dia', 'temperature', 'respiratory_rate', 'pain_level', 'symptom_severity']
    features_df = pd.DataFrame([[
        age, heart_rate, spo2, bp_sys, bp_dia, temperature, respiratory_rate, pain_level, symptom_severity
    ]], columns=feature_cols)
    
    scaled_features = _scaler.transform(features_df)
    probabilities = _model.predict_proba(scaled_features)[0]
    classes = _model.classes_
    
    prob_dict = {cls: round(float(prob) * 100, 2) for cls, prob in zip(classes, probabilities)}
    best_idx = np.argmax(probabilities)
    predicted_priority = classes[best_idx]
    confidence = round(float(probabilities[best_idx]) * 100, 2)
    
    label_map = {
        'P1': 'P1 - Critical',
        'P2': 'P2 - High',
        'P3': 'P3 - Moderate',
        'P4': 'P4 - Low'
    }
    
    return {
        'priority': predicted_priority,
        'confidence': confidence,
        'priority_label': label_map.get(predicted_priority, f"{predicted_priority} - Priority"),
        'probabilities': prob_dict,
        'features_used': {
            'age': age,
            'heart_rate': heart_rate,
            'spo2': spo2,
            'bp_sys': bp_sys,
            'bp_dia': bp_dia,
            'temperature': temperature,
            'respiratory_rate': respiratory_rate,
            'pain_level': pain_level,
            'symptom_severity': symptom_severity
        }
    }

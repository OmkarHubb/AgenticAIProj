import json
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def generate_synthetic_data(n_samples=1200):
    np.random.seed(42)
    records = []
    priorities = ['P1', 'P2', 'P3', 'P4']
    
    for _ in range(n_samples):
        target_p = np.random.choice(priorities, p=[0.2, 0.25, 0.3, 0.25])
        
        if target_p == 'P1': # Critical
            age = int(np.random.randint(1, 90))
            heart_rate = int(np.random.choice([np.random.randint(40, 55), np.random.randint(135, 185)]))
            spo2 = int(np.random.randint(70, 89))
            bp_sys = int(np.random.choice([np.random.randint(60, 85), np.random.randint(175, 210)]))
            bp_dia = int(np.random.randint(40, 115))
            temperature = round(float(np.random.uniform(34.5, 41.0)), 1)
            resp_rate = int(np.random.choice([np.random.randint(8, 11), np.random.randint(30, 45)]))
            pain_level = int(np.random.randint(8, 11))
            symptom_severity = 4
        elif target_p == 'P2': # High
            age = int(np.random.randint(1, 90))
            heart_rate = int(np.random.randint(110, 135))
            spo2 = int(np.random.randint(90, 94))
            bp_sys = int(np.random.randint(155, 175))
            bp_dia = int(np.random.randint(95, 110))
            temperature = round(float(np.random.uniform(38.5, 40.0)), 1)
            resp_rate = int(np.random.randint(24, 30))
            pain_level = int(np.random.randint(7, 9))
            symptom_severity = 3
        elif target_p == 'P3': # Moderate
            age = int(np.random.randint(1, 90))
            heart_rate = int(np.random.randint(90, 110))
            spo2 = int(np.random.randint(94, 97))
            bp_sys = int(np.random.randint(130, 155))
            bp_dia = int(np.random.randint(85, 95))
            temperature = round(float(np.random.uniform(37.5, 38.5)), 1)
            resp_rate = int(np.random.randint(18, 24))
            pain_level = int(np.random.randint(4, 7))
            symptom_severity = 2
        else: # P4 - Low
            age = int(np.random.randint(1, 90))
            heart_rate = int(np.random.randint(60, 90))
            spo2 = int(np.random.randint(97, 101))
            bp_sys = int(np.random.randint(100, 130))
            bp_dia = int(np.random.randint(60, 85))
            temperature = round(float(np.random.uniform(36.0, 37.4)), 1)
            resp_rate = int(np.random.randint(12, 18))
            pain_level = int(np.random.randint(1, 4))
            symptom_severity = 1
            
        records.append({
            'age': age,
            'heart_rate': heart_rate,
            'spo2': spo2,
            'bp_sys': bp_sys,
            'bp_dia': bp_dia,
            'temperature': temperature,
            'respiratory_rate': resp_rate,
            'pain_level': pain_level,
            'symptom_severity': symptom_severity,
            'priority': target_p
        })
        
    return pd.DataFrame(records)

def train_and_evaluate():
    df = generate_synthetic_data(1200)
    feature_cols = ['age', 'heart_rate', 'spo2', 'bp_sys', 'bp_dia', 'temperature', 'respiratory_rate', 'pain_level', 'symptom_severity']
    
    X = df[feature_cols]
    y = df['priority']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average='macro'))
    rec_macro = float(recall_score(y_test, y_pred, average='macro'))
    f1_macro = float(f1_score(y_test, y_pred, average='macro'))
    
    report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics = {
        'accuracy': round(acc, 4),
        'precision': round(prec_macro, 4),
        'recall': round(rec_macro, 4),
        'f1_score': round(f1_macro, 4),
        'total_samples': len(df),
        'test_samples': len(y_test),
        'details': report
    }
    
    joblib.dump(model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print("Model training complete.")
    print(f"Accuracy: {acc:.4f}, Precision: {prec_macro:.4f}, Recall: {rec_macro:.4f}, F1: {f1_macro:.4f}")
    return metrics

if __name__ == '__main__':
    train_and_evaluate()

import sqlite3
import os
import datetime

DB_PATH = "patients.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
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
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS history_chat (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    conn.commit()
    conn.close()

def generate_patient_id():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM patients")
    row = cursor.fetchone()
    count = row['count'] + 1001
    conn.close()
    return f"PAT-{count}"

def save_patient(data):
    conn = get_connection()
    cursor = conn.cursor()
    
    if 'patient_id' not in data or not data['patient_id']:
        data['patient_id'] = generate_patient_id()
        
    cursor.execute("""
    INSERT INTO patients (
        patient_id, name, age, heart_rate, spo2, blood_pressure,
        temperature, respiratory_rate, pain_level, symptoms,
        priority, confidence, queue_status, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data['patient_id'],
        data.get('name', 'Anonymous Patient'),
        data['age'],
        data['heart_rate'],
        data['spo2'],
        data['blood_pressure'],
        data['temperature'],
        data['respiratory_rate'],
        data['pain_level'],
        data.get('symptoms', 'None reported'),
        data['priority'],
        data['confidence'],
        data.get('queue_status', 'Waiting'),
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    
    conn.commit()
    patient_id = data['patient_id']
    conn.close()
    return patient_id

def get_patient_history(query=None):
    conn = get_connection()
    cursor = conn.cursor()
    
    if query and str(query).strip():
        q = str(query).strip()
        cursor.execute("SELECT * FROM patients WHERE patient_id = ? OR name LIKE ? ORDER BY timestamp DESC", 
                       (q, f"%{q}%"))
    else:
        cursor.execute("SELECT * FROM patients ORDER BY timestamp DESC")
        
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_emergency_queue():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
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
    """)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def update_patient_status(patient_id, new_status):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE patients SET queue_status = ? WHERE patient_id = ?", (new_status, patient_id))
    conn.commit()
    updated = cursor.rowcount > 0
    conn.close()
    return updated

def get_dashboard_summary():
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) as total FROM patients WHERE queue_status = 'Waiting'")
    waiting_count = cursor.fetchone()['total']
    
    cursor.execute("SELECT COUNT(*) as p1_count FROM patients WHERE priority = 'P1' AND queue_status = 'Waiting'")
    p1_count = cursor.fetchone()['p1_count']
    
    cursor.execute("SELECT COUNT(*) as p2_count FROM patients WHERE priority = 'P2' AND queue_status = 'Waiting'")
    p2_count = cursor.fetchone()['p2_count']
    
    cursor.execute("SELECT COUNT(*) as total_triaged FROM patients")
    total_triaged = cursor.fetchone()['total_triaged']
    
    conn.close()
    return {
        'waiting_count': waiting_count,
        'p1_count': p1_count,
        'p2_count': p2_count,
        'total_triaged': total_triaged
    }

def save_chat_message(session_id, role, content):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO history_chat (session_id, role, content, timestamp)
    VALUES (?, ?, ?, ?)
    """, (session_id, role, content, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def get_chat_history(session_id="default"):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT role, content, timestamp FROM history_chat WHERE session_id = ? ORDER BY id ASC", (session_id,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

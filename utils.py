import cv2
import numpy as np
import sqlite3
import pandas as pd
from datetime import datetime
import os
from deepface import DeepFace
from face_encode import load_encodings

DB_FILE = "attendance.db"

def init_db():
    """Initialize SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    conn.execute('''CREATE TABLE IF NOT EXISTS attendance (
                        id INTEGER PRIMARY KEY,
                        date TEXT,
                        time TEXT,
                        student_id TEXT,
                        student_name TEXT,
                        status TEXT
                    )''')
    conn.commit()
    conn.close()

def mark_attendance(student_id, student_name):
    """Mark attendance if not already marked today."""
    today = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attendance WHERE date=? AND student_id=?", (today, student_id))
    if cursor.fetchone():
        conn.close()
        return False, "Attendance already marked for today."
    
    now = datetime.now()
    cursor.execute("INSERT INTO attendance (date, time, student_id, student_name, status) VALUES (?, ?, ?, ?, ?)",
                   (today, now.strftime("%H:%M:%S"), student_id, student_name, "Present"))
    conn.commit()
    conn.close()
    return True, "Attendance marked successfully."

def get_today_attendance():
    """Get today's attendance as DataFrame."""
    today = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM attendance WHERE date=?", conn, params=(today,))
    conn.close()
    return df

def get_attendance_history(date=None, student_id=None, student_name=None):
    """Get filtered attendance history."""
    query = "SELECT * FROM attendance WHERE 1=1"
    params = []
    if date:
        query += " AND date=?"
        params.append(date)
    if student_id:
        query += " AND student_id=?"
        params.append(student_id)
    if student_name:
        query += " AND student_name LIKE ?"
        params.append(f"%{student_name}%")
    
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def export_to_csv():
    """Export all attendance to CSV."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM attendance", conn)
    conn.close()
    df.to_csv("attendance_export.csv", index=False)
    return "attendance_export.csv"

def recognize_face(frame, encodings_dict, tolerance=0.4):  # Tolerance is distance threshold
    """
    Recognize face in frame using DeepFace.
    Returns (student_id, student_name, None) or (None, None, error_message).
    """
    try:
        # Detect faces using DeepFace
        detections = DeepFace.extract_faces(frame, detector_backend='opencv', enforce_detection=False)
        if not detections:
            return None, None, "No face detected."
        
        # Process the first face
        face_img = detections[0]['face']
        face_img = (face_img * 255).astype(np.uint8)
        
        # Get embedding
        embedding = DeepFace.represent(face_img, model_name='Facenet', enforce_detection=False)[0]['embedding']
        embedding = np.array(embedding)
        
        # Compare with stored encodings using Euclidean distance
        for student_id, data in encodings_dict.items():
            stored_embedding = np.array(data['encoding'])
            distance = np.linalg.norm(embedding - stored_embedding)
            if distance < tolerance:  # Lower distance = better match
                return student_id, data['name'], None
        
        return None, None, "Face not recognized. Please register first."
    except Exception as e:
        return None, None, f"Error recognizing face: {str(e)}"
import sqlite3
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "bottle_detection.db"

def initialize_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS detections(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        time TEXT,
        status TEXT,
        confidence REAL,
        bottle_count INTEGER
    )
    """)

    conn.commit()
    conn.close()


def save_detection(status, confidence, bottle_count):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO detections(time, status, confidence, bottle_count)
    VALUES (?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        status,
        confidence,
        bottle_count
    ))

    conn.commit()
    conn.close()

def get_detections():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT * FROM detections
    ORDER BY id DESC
    """)

    data = cursor.fetchall()

    conn.close()

    return data

def get_statistics():

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT 
        COUNT(*),
        SUM(bottle_count),
        AVG(confidence),
        MAX(time)
    FROM detections
    """)

    result = cursor.fetchone()

    conn.close()

    return result

def get_chart_data():

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT time, confidence, bottle_count
    FROM detections
    ORDER BY id
    """)

    data = cursor.fetchall()

    conn.close()

    return data

if __name__ == "__main__":
    initialize_database()
    print("Database created successfully.")
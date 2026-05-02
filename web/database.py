import sqlite3

DATABASE_NAME = "../ai/traffic.db"

def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            vehicle_type TEXT,
            direction TEXT,
            plate TEXT,
            image_path TEXT,
            visit_reason TEXT
        )
    """)

    conn.commit()
    conn.close()


def get_all_records():
    conn = get_db_connection()
    records = conn.execute("""
        SELECT 
            id,
            timestamp,
            vehicle_type,
            direction,
            plate,
            image_path,
            visit_reason
        FROM vehicle_logs
        ORDER BY id DESC
    """).fetchall()
    conn.close()
    return records
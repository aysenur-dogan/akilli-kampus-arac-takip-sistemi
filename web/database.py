import sqlite3
DATABASE_NAME = "vehicle_records.db"

def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            vehicle_type TEXT,
            plate TEXT,
            visit_text TEXT,
            status TEXT,
            image_path TEXT
        )
    """)

    conn.commit()
    conn.close()
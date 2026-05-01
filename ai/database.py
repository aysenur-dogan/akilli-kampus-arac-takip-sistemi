import sqlite3

def init_db():
    conn = sqlite3.connect("traffic.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            vehicle_type TEXT,
            direction TEXT,
            plate TEXT,
            image_path TEXT
        )
    """)

    conn.commit()
    conn.close()


def insert_log(vehicle_type, direction, timestamp, plate, image_path):
    conn = sqlite3.connect("traffic.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO vehicle_logs (timestamp, vehicle_type, direction, plate, image_path)
        VALUES (?, ?, ?, ?, ?)
    """, (timestamp, vehicle_type, direction, plate, image_path))

    conn.commit()
    conn.close()

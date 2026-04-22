import sqlite3

DB_PATH = "traffic.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS vehicle_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        vehicle_type TEXT NOT NULL,
        direction TEXT NOT NULL,
        image_path TEXT
    )
    """)

    conn.commit()
    conn.close()


def insert_log(vehicle_type, direction, timestamp, image_path):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO vehicle_logs (timestamp, vehicle_type, direction, image_path)
    VALUES (?, ?, ?, ?)
    """, (timestamp, vehicle_type, direction, image_path))

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print("Veritabani hazir: traffic.db")
from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import os
import shutil
import csv
from flask import Response
from flask import jsonify

app = Flask(__name__)

DB_PATH = "../ai/traffic.db"


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def sync_snapshots():
    src_dir = "../ai/snapshots"
    dst_dir = "static/snapshots"

    os.makedirs(dst_dir, exist_ok=True)

    if os.path.exists(src_dir):
        for file in os.listdir(src_dir):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                src = os.path.join(src_dir, file)
                dst = os.path.join(dst_dir, file)

                if not os.path.exists(dst):
                    shutil.copy(src, dst)


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == "admin" and password == "1234":
            return redirect(url_for("dashboard"))

        return render_template("login.html", error="Kullanıcı adı veya şifre hatalı")

    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    sync_snapshots()

    conn = get_db_connection()

    records = conn.execute("""
        SELECT *
        FROM vehicle_logs
        ORDER BY id DESC
    """).fetchall()

    total_count = conn.execute("""
        SELECT COUNT(*)
        FROM vehicle_logs
    """).fetchone()[0]

    last_record = records[0] if records else None

    conn.close()

    return render_template(
        "dashboard.html",
        records=records,
        last_record=last_record,
        total_count=total_count
    )


@app.route("/kayitlar")
def kayitlar():
    sync_snapshots()

    conn = get_db_connection()

    records = conn.execute("""
        SELECT *
        FROM vehicle_logs
        ORDER BY id DESC
    """).fetchall()

    conn.close()

    return render_template("kayitlar.html", records=records)


@app.route("/yeni-kayit", methods=["GET", "POST"])
def yeni_kayit():
    if request.method == "POST":
        plate = request.form.get("plate")
        visit_text = request.form.get("visit_text")

        conn = get_db_connection()

        last_record = conn.execute("""
            SELECT id
            FROM vehicle_logs
            WHERE plate = ?
            ORDER BY id DESC
            LIMIT 1
        """, (plate,)).fetchone()

        if last_record:
            conn.execute("""
                UPDATE vehicle_logs
                SET visit_reason = ?
                WHERE id = ?
            """, (visit_text, last_record["id"]))

        conn.commit()
        conn.close()

        return redirect(url_for("dashboard"))

    return render_template("yeni_kayit.html")


@app.route("/raporlar")
def raporlar():
    conn = get_db_connection()

    total_count = conn.execute("""
        SELECT COUNT(*)
        FROM vehicle_logs
    """).fetchone()[0]

    unknown_count = conn.execute("""
        SELECT COUNT(*)
        FROM vehicle_logs
        WHERE plate = 'UNKNOWN'
    """).fetchone()[0]

    success_count = total_count - unknown_count

    conn.close()

    return render_template(
        "raporlar.html",
        total_count=total_count,
        success_count=success_count,
        unknown_count=unknown_count
    )

@app.route("/rapor-indir")
def rapor_indir():
    conn = get_db_connection()

    records = conn.execute("""
        SELECT timestamp, vehicle_type, direction, plate, visit_reason
        FROM vehicle_logs
        ORDER BY id DESC
    """).fetchall()

    conn.close()

    def generate():
        yield "Tarih-Saat,Araç Türü,Yön,Plaka,Giriş Amacı\n"

        for record in records:
            yield f"{record['timestamp']},{record['vehicle_type']},{record['direction']},{record['plate']},{record['visit_reason'] or 'Belirtilmedi'}\n"

    return Response(
        generate(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=gunluk_arac_raporu.csv"
        }
    )
@app.route("/arac/<plate>")
def arac_detay(plate):
    sync_snapshots()

    conn = get_db_connection()

    records = conn.execute("""
        SELECT *
        FROM vehicle_logs
        WHERE plate = ?
        ORDER BY id DESC
    """, (plate,)).fetchall()

    last_record = records[0] if records else None

    conn.close()

    return render_template(
        "arac_detay.html",
        plate=plate,
        records=records,
        last_record=last_record
    )
@app.route("/api/records")
def api_records():
    sync_snapshots()

    conn = get_db_connection()

    records = conn.execute("""
        SELECT *
        FROM vehicle_logs
        ORDER BY id DESC
        LIMIT 20
    """).fetchall()

    conn.close()

    data = []

    for r in records:
        image_url = ""

        if r["image_path"]:
            image_name = r["image_path"].split("/")[-1]
            image_url = url_for("static", filename="snapshots/" + image_name)

        data.append({
            "id": r["id"],
            "timestamp": r["timestamp"],
            "plate": r["plate"],
            "vehicle_type": r["vehicle_type"],
            "direction": r["direction"],
            "visit_reason": r["visit_reason"] if r["visit_reason"] else "Belirtilmedi",
            "image_url": image_url
        })

    return jsonify(data)
@app.route("/api/son-kayit-guncelle", methods=["POST"])
def son_kayit_guncelle():
    data = request.json
    visit_text = data.get("visit_text", "Belirtilmedi")

    conn = get_db_connection()

    last_record = conn.execute("""
        SELECT id FROM vehicle_logs
        ORDER BY id DESC
        LIMIT 1
    """).fetchone()

    if last_record:
        conn.execute("""
            UPDATE vehicle_logs
            SET visit_reason = ?
            WHERE id = ?
        """, (visit_text, last_record["id"]))

        conn.commit()
        conn.close()

        return {
            "message": "Son kayıt güncellendi",
            "visit_text": visit_text
        }

    conn.close()

    return {
        "message": "Güncellenecek kayıt bulunamadı"
    }
@app.route("/api/new-records")
def api_new_records():
    sync_snapshots()

    last_id = request.args.get("last_id", 0, type=int)

    conn = get_db_connection()

    records = conn.execute("""
        SELECT *
        FROM vehicle_logs
        WHERE id > ?
        ORDER BY id ASC
    """, (last_id,)).fetchall()

    conn.close()

    data = []

    for r in records:
        image_url = ""

        if r["image_path"]:
            image_name = r["image_path"].split("/")[-1]
            image_url = url_for("static", filename="snapshots/" + image_name)

        data.append({
            "id": r["id"],
            "timestamp": r["timestamp"],
            "plate": r["plate"],
            "vehicle_type": r["vehicle_type"],
            "direction": r["direction"],
            "visit_reason": r["visit_reason"] if r["visit_reason"] else "Belirtilmedi",
            "image_url": image_url
        })

    return jsonify(data)
if __name__ == "__main__":
    app.run(debug=True)
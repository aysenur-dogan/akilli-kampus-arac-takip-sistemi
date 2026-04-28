from flask import Flask, render_template, request, redirect, url_for
from database import init_db, get_db_connection

app = Flask(__name__)

init_db()


conn = get_db_connection()
existing_records = conn.execute("SELECT COUNT(*) FROM vehicle_records").fetchone()[0]

if existing_records == 0:
    conn.execute("""
        INSERT INTO vehicle_records 
        (timestamp, vehicle_type, plate, visit_text, status, image_path)
        VALUES (?, ?, ?, ?, ?, ?)
    """, ("14:15", "Araç", "61 AA 001", "Rektörlük Ziyareti", "✅ Kayıtlı", ""))

    conn.execute("""
        INSERT INTO vehicle_records 
        (timestamp, vehicle_type, plate, visit_text, status, image_path)
        VALUES (?, ?, ?, ?, ?, ?)
    """, ("14:22", "Araç", "34 ABC 123", "Misafir - Kütüphane", "✅ Kayıtlı", ""))

    conn.commit()

conn.close()



@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == "admin" and password == "1234":
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Kullanıcı adı veya şifre yanlış")

    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    conn = get_db_connection()
    records = conn.execute("SELECT * FROM vehicle_records ORDER BY id DESC").fetchall()
    conn.close()
    return render_template("dashboard.html", records=records)

@app.route("/kayitlar")
def kayitlar():
    return render_template("kayitlar.html")

@app.route("/raporlar")
def raporlar():
    return render_template("raporlar.html")




@app.route("/yeni-kayit", methods=["GET", "POST"])
def yeni_kayit():
    if request.method == "POST":
        plate = request.form.get("plate")
        visit_text = request.form.get("visit_text")

        conn = get_db_connection()
        conn.execute("""
            INSERT INTO vehicle_records
            (timestamp, vehicle_type, plate, visit_text, status, image_path)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("Şimdi", "Araç", plate, visit_text, "✅ Kayıtlı", ""))
        conn.commit()
        conn.close()

        return redirect(url_for("dashboard"))

    return render_template("yeni_kayit.html")



@app.route("/api/ekle", methods=["POST"])
def api_ekle():
    data = request.json

    plate = data.get("plate")
    visit_text = data.get("visit_text", "Bilinmiyor")

    conn = get_db_connection()
    conn.execute("""
        INSERT INTO vehicle_records
        (timestamp, vehicle_type, plate, visit_text, status, image_path)
        VALUES (?, ?, ?, ?, ?, ?)
    """, ("AI", "Araç", plate, visit_text, "🤖 AI Kayıt", ""))
    conn.commit()
    conn.close()

    return {"message": "Kayıt eklendi"}




if __name__ == "__main__":
    app.run(debug=True)
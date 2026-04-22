import cv2
import json
import numpy as np

VIDEO_PATH = "video.mp4"
OUT_JSON = "count_lines.json"

LINE_NAMES = ["Gelis Cizgi", "Gidis Cizgi"]

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_ACTIVE = (0, 255, 255)
COLOR_DONE = (0, 255, 0)

points = []
all_lines = {}
current_idx = 0


def mouse_cb(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append([x, y])


cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Video açılamadı.")

ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("İlk kare okunamadı.")

cv2.namedWindow("Line Creator", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Line Creator", mouse_cb)

while True:
    vis = frame.copy()

    # Kaydedilen çizgileri çiz
    for name, pts in all_lines.items():
        p1 = tuple(pts[0])
        p2 = tuple(pts[1])

        cv2.line(vis, p1, p2, COLOR_DONE, 3)
        cv2.circle(vis, p1, 6, COLOR_DONE, -1)
        cv2.circle(vis, p2, 6, COLOR_DONE, -1)
        cv2.putText(vis, f"{name} (saved)", (p1[0], p1[1] - 10),
                    FONT, 0.7, COLOR_DONE, 2)

    # Aktif çizgi
    if current_idx < len(LINE_NAMES):
        line_name = LINE_NAMES[current_idx]

        cv2.putText(
            vis,
            f"Drawing: {line_name} | 2 nokta sec | ENTER=finish | R=reset | S=save | Q=quit",
            (20, 30),
            FONT,
            0.6,
            COLOR_ACTIVE,
            2
        )

        for p in points:
            cv2.circle(vis, (p[0], p[1]), 6, COLOR_ACTIVE, -1)

        if len(points) == 2:
            cv2.line(vis, tuple(points[0]), tuple(points[1]), COLOR_ACTIVE, 3)

    else:
        cv2.putText(
            vis,
            "Tum cizgiler tamamlandi | S=save | Q=quit",
            (20, 30),
            FONT,
            0.7,
            COLOR_DONE,
            2
        )

    cv2.imshow("Line Creator", vis)
    key = cv2.waitKey(20) & 0xFF

    if key == ord("q"):
        break

    if key == ord("r"):
        points = []

    if key == 13 and current_idx < len(LINE_NAMES):  # Enter
        if len(points) != 2:
            print("Bir çizgi için tam 2 nokta seçmelisin.")
            continue

        all_lines[LINE_NAMES[current_idx]] = points.copy()
        points = []
        current_idx += 1

    if key == ord("s"):
        if len(all_lines) == 0:
            print("Kaydedilecek çizgi yok.")
            continue

        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(all_lines, f, ensure_ascii=False, indent=2)

        print(f"[OK] Çizgiler kaydedildi: {OUT_JSON}")

cv2.destroyAllWindows()
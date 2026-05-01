import cv2
import json
import numpy as np

VIDEO_PATH = "video.mp4"
OUTPUT_FILE = "rois.json"

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Video okunamadi.")
    exit()

points = []
rois = {}
current_name = "Gelis Yolu"

def mouse_callback(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"{current_name} nokta eklendi:", x, y)

while True:
    temp = frame.copy()

    if points:
        for p in points:
            cv2.circle(temp, tuple(p), 5, (0, 0, 255), -1)

        if len(points) > 1:
            cv2.polylines(temp, [np.array(points, np.int32)], False, (0, 255, 0), 2)

    cv2.putText(
        temp,
        f"{current_name} ciziliyor | ENTER: kaydet | R: sifirla | Q: cik",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow("ROI Secimi", temp)
    cv2.setMouseCallback("ROI Secimi", mouse_callback)

    key = cv2.waitKey(1) & 0xFF

    if key == 13:  # ENTER
        if len(points) >= 3:
            rois[current_name] = points.copy()
            print(f"{current_name} kaydedildi.")

            if current_name == "Gelis Yolu":
                current_name = "Gidis Yolu"
                points = []
            else:
                break
        else:
            print("En az 3 nokta secmelisin.")

    elif key == ord("r"):
        points = []
        print("Noktalar sifirlandi.")

    elif key == ord("q"):
        break

cv2.destroyAllWindows()

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(rois, f, indent=4, ensure_ascii=False)

print(f"[OK] ROI alanlari kaydedildi: {OUTPUT_FILE}")

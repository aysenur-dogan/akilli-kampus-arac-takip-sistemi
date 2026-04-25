from ultralytics import YOLO
import cv2
import json
import numpy as np

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("video.mp4")

vehicle_classes = ["car", "bus", "truck", "motorcycle"]

with open("rois.json", "r", encoding="utf-8") as f:
    rois = json.load(f)

gelis_poly = np.array(rois["Gelis Yolu"], np.int32)
gidis_poly = np.array(rois["Gidis Yolu"], np.int32)


def point_in_polygon(point, polygon):
    x, y = point
    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    names = results[0].names
    boxes = results[0].boxes

    gelis_count = 0
    gidis_count = 0

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = names[cls_id]

        if cls_name not in vehicle_classes:
            continue

        if conf < 0.40:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        in_gelis = point_in_polygon((cx, cy), gelis_poly)
        in_gidis = point_in_polygon((cx, cy), gidis_poly)

        if not (in_gelis or in_gidis):
            continue

        if in_gelis:
            color = (0, 255, 0)
            label = "Gelis"
            gelis_count += 1
        else:
            color = (255, 0, 0)
            label = "Gidis"
            gidis_count += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{cls_name} | {label}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.putText(frame, f"Gelis Sayisi: {gelis_count}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Gidis Sayisi: {gidis_count}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("ROI Vehicles Only", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
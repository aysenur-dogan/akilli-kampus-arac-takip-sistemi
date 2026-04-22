from ultralytics import YOLO
import cv2
import json
import numpy as np
from database import init_db, insert_log
from datetime import datetime
import os

# DB ve klasör hazırlığı
init_db()
os.makedirs("snapshots", exist_ok=True)

# Model ve video
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("video.mp4")

vehicle_classes = ["car", "bus", "truck", "motorcycle"]

# ROI ve çizgi dosyaları
with open("rois.json", "r", encoding="utf-8") as f:
    rois = json.load(f)

with open("count_lines.json", "r", encoding="utf-8") as f:
    lines = json.load(f)

gelis_poly = np.array(rois["Gelis Yolu"], np.int32)
gidis_poly = np.array(rois["Gidis Yolu"], np.int32)

gelis_line = lines["Gelis Cizgi"]
gidis_line = lines["Gidis Cizgi"]

# Sayaçlar
gelis_count = 0
gidis_count = 0

# Takip geçmişi
track_history = {}
counted_gelis_ids = set()
counted_gidis_ids = set()


def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def point_line_side(point, line):
    x, y = point
    x1, y1 = line[0]
    x2, y2 = line[1]
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, verbose=False)

    if not results or results[0].boxes is None:
        cv2.imshow("Vehicle Counting", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    boxes = results[0].boxes
    names = results[0].names

    if boxes.id is None:
        cv2.imshow("Vehicle Counting", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    for box, track_id_tensor in zip(boxes, boxes.id):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = names[cls_id]
        track_id = int(track_id_tensor)

        if cls_name not in vehicle_classes:
            continue

        if conf < 0.40:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        center = (cx, cy)

        in_gelis = point_in_polygon(center, gelis_poly)
        in_gidis = point_in_polygon(center, gidis_poly)

        if not (in_gelis or in_gidis):
            continue

        if track_id not in track_history:
            track_history[track_id] = []

        track_history[track_id].append(center)

        if len(track_history[track_id]) > 10:
            track_history[track_id].pop(0)

        label = f"{cls_name} ID:{track_id}"

        if in_gelis:
            color = (0, 255, 0)

            if len(track_history[track_id]) >= 2 and track_id not in counted_gelis_ids:
                prev_center = track_history[track_id][-2]
                curr_center = track_history[track_id][-1]

                prev_side = point_line_side(prev_center, gelis_line)
                curr_side = point_line_side(curr_center, gelis_line)

                if prev_side * curr_side < 0:
                    gelis_count += 1
                    counted_gelis_ids.add(track_id)
                    print("GELIS sayildi:", cls_name, track_id)

                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    filename_time = datetime.now().strftime("%Y%m%d_%H%M%S")

                    h, w = frame.shape[:2]
                    x1_safe = max(0, x1)
                    y1_safe = max(0, y1)
                    x2_safe = min(w, x2)
                    y2_safe = min(h, y2)

                    vehicle_crop = frame[y1_safe:y2_safe, x1_safe:x2_safe]

                    if vehicle_crop.size > 0:
                        image_path = f"snapshots/gelis_{track_id}_{filename_time}.jpg"
                        cv2.imwrite(image_path, vehicle_crop)
                    else:
                        image_path = None

                    insert_log(cls_name, "Gelis", timestamp, image_path)

        else:
            color = (255, 0, 0)

            if len(track_history[track_id]) >= 2 and track_id not in counted_gidis_ids:
                prev_center = track_history[track_id][-2]
                curr_center = track_history[track_id][-1]

                prev_side = point_line_side(prev_center, gidis_line)
                curr_side = point_line_side(curr_center, gidis_line)

                if prev_side * curr_side < 0:
                    gidis_count += 1
                    counted_gidis_ids.add(track_id)
                    print("GIDIS sayildi:", cls_name, track_id)

                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    filename_time = datetime.now().strftime("%Y%m%d_%H%M%S")

                    h, w = frame.shape[:2]
                    x1_safe = max(0, x1)
                    y1_safe = max(0, y1)
                    x2_safe = min(w, x2)
                    y2_safe = min(h, y2)

                    vehicle_crop = frame[y1_safe:y2_safe, x1_safe:x2_safe]

                    if vehicle_crop.size > 0:
                        image_path = f"snapshots/gidis_{track_id}_{filename_time}.jpg"
                        cv2.imwrite(image_path, vehicle_crop)
                    else:
                        image_path = None

                    insert_log(cls_name, "Gidis", timestamp, image_path)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, center, 4, (0, 0, 255), -1)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.putText(
        frame,
        f"Gelis: {gelis_count}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    cv2.putText(
        frame,
        f"Gidis: {gidis_count}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.imshow("Vehicle Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
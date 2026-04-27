from ultralytics import YOLO
import cv2
import json
import numpy as np
from database import init_db, insert_log
from datetime import datetime
import os
import re
import easyocr

init_db()
os.makedirs("snapshots", exist_ok=True)
os.makedirs("plate_crops", exist_ok=True)

vehicle_model = YOLO("yolov8n.pt")
plate_model = YOLO("models/plate_detector_v2.pt")
reader = easyocr.Reader(["en"])

cap = cv2.VideoCapture("video.mp4")

vehicle_classes = ["car", "bus", "truck", "motorcycle"]

with open("rois.json", "r", encoding="utf-8") as f:
    rois = json.load(f)

with open("count_lines.json", "r", encoding="utf-8") as f:
    lines = json.load(f)

gelis_poly = np.array(rois["Gelis Yolu"], np.int32)
gidis_poly = np.array(rois["Gidis Yolu"], np.int32)

gelis_line = lines["Gelis Cizgi"]
gidis_line = lines["Gidis Cizgi"]

gelis_count = 0
gidis_count = 0

track_history = {}
track_crops = {}

counted_gelis_ids = set()
counted_gidis_ids = set()


def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def point_line_side(point, line):
    x, y = point
    x1, y1 = line[0]
    x2, y2 = line[1]
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)


def clean_text(text):
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def format_plate(text):
    text = clean_text(text)
    matches = re.findall(r"\d{2}[A-Z]{1,3}\d{2,4}", text)

    if matches:
        plate = max(matches, key=len)

        if len(plate) > 8:
            plate = plate[:8]

        return plate

    return "UNKNOWN"
def fix_common_errors(plate):
    if plate == "UNKNOWN":
        return plate

    # 1. İl kodu düzeltme (ilk 2 rakam)
    if len(plate) >= 2:
        il_kod = plate[:2]

        # OCR hataları
        replacements = {
            "67": "61",
            "65": "55",
            "68": "61",
            "69": "61"
        }

        if il_kod in replacements:
            plate = replacements[il_kod] + plate[2:]

    # 2. Harf-rakam karışıklığı
    plate = plate.replace("O", "0")
    plate = plate.replace("I", "1")
    

    return plate
def fix_common_errors(plate):
    if plate == "UNKNOWN":
        return plate

    if len(plate) >= 2:
        il_kod = plate[:2]

        replacements = {
            "67": "61",
            "65": "55",
            "68": "61",
            "69": "61"
        }

        if il_kod in replacements:
            plate = replacements[il_kod] + plate[2:]

    if len(plate) > 8:
        plate = plate[:8]

    return plate

def plate_score(plate):
    if plate == "UNKNOWN":
        return -100

    score = 0

    if re.fullmatch(r"\d{2}[A-Z]{1,3}\d{2,4}", plate):
        score += 50

    if plate[:2].isdigit() and 1 <= int(plate[:2]) <= 81:
        score += 30

    if 6 <= len(plate) <= 9:
        score += 20
    else:
        score -= 20

    return score


def select_best_plate(candidates):
    if not candidates:
        return "UNKNOWN"

    return max(candidates, key=plate_score)


def preprocess_plate(plate_crop):
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    return thresh


def read_plate_from_vehicle(vehicle_crop, track_id, direction, filename_time):
    if vehicle_crop is None or vehicle_crop.size == 0:
        return "UNKNOWN"

    plate_results = plate_model(vehicle_crop, verbose=False)

    best_plate_crop = None
    best_conf = 0

    for result in plate_results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            conf = float(box.conf[0])

            if conf > best_conf:
                best_conf = conf
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                h, w = vehicle_crop.shape[:2]

                pad = 12
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)

                best_plate_crop = vehicle_crop[y1:y2, x1:x2]

    if best_plate_crop is None or best_plate_crop.size == 0:
        return "UNKNOWN"

    plate_crop_path = f"plate_crops/{direction.lower()}_{track_id}_{filename_time}.jpg"
    cv2.imwrite(plate_crop_path, best_plate_crop)

    processed = preprocess_plate(best_plate_crop)
    easy_results = reader.readtext(processed)

    plate_text = ""

    for _, text, prob in easy_results:
        if prob > 0.10:
            plate_text += clean_text(text) + " "

    return format_plate(plate_text)


def save_vehicle_and_log(frame, x1, y1, x2, y2, cls_name, direction, track_id):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    h, w = frame.shape[:2]

    x1_safe = max(0, x1)
    y1_safe = max(0, y1)
    x2_safe = min(w, x2)
    y2_safe = min(h, y2)

    vehicle_crop = frame[y1_safe:y2_safe, x1_safe:x2_safe]

    if vehicle_crop.size > 0:
        image_path = f"snapshots/{direction.lower()}_{track_id}_{filename_time}.jpg"
        cv2.imwrite(image_path, vehicle_crop)
    else:
        image_path = None

    candidate_crops = track_crops.get(track_id, []).copy()

    if vehicle_crop.size > 0:
        candidate_crops.append(vehicle_crop)

    plate_candidates = []

    for i, crop in enumerate(candidate_crops):
        plate_result = read_plate_from_vehicle(
            crop,
            track_id,
            direction,
            f"{filename_time}_{i}"
        )

        if plate_result != "UNKNOWN":
            plate_candidates.append(plate_result)

    plate = select_best_plate(plate_candidates)
    plate = fix_common_errors(plate)
    print(f"{direction.upper()} sayildi:", cls_name, track_id)
    print("PLAKA:", plate)

    insert_log(cls_name, direction, timestamp, plate, image_path)


while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = vehicle_model.track(frame, persist=True, verbose=False)

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

        h, w = frame.shape[:2]

        x1_safe = max(0, x1)
        y1_safe = max(0, y1)
        x2_safe = min(w, x2)
        y2_safe = min(h, y2)

        vehicle_crop_for_buffer = frame[y1_safe:y2_safe, x1_safe:x2_safe]

        if vehicle_crop_for_buffer.size > 0:
            if track_id not in track_crops:
                track_crops[track_id] = []

            track_crops[track_id].append(vehicle_crop_for_buffer.copy())

            if len(track_crops[track_id]) > 5:
                track_crops[track_id].pop(0)

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

                    save_vehicle_and_log(
                        frame,
                        x1,
                        y1,
                        x2,
                        y2,
                        cls_name,
                        "Gelis",
                        track_id
                    )

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

                    save_vehicle_and_log(
                        frame,
                        x1,
                        y1,
                        x2,
                        y2,
                        cls_name,
                        "Gidis",
                        track_id
                    )

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
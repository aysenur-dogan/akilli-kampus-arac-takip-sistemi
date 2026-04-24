from ultralytics import YOLO
import cv2
import easyocr
import os
import re

plate_model = YOLO("models/plate_detector.pt")
reader = easyocr.Reader(["en"])

image_path = "snapshots/gidis_120_20260424_144310.jpg"
image = cv2.imread(image_path)

if image is None:
    raise RuntimeError("Görüntü okunamadı.")

results = plate_model(image, verbose=False)


def clean_text(text):
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def format_plate(text):
    text = text.replace(" ", "")
    matches = re.findall(r"\d{2}[A-Z]{1,3}\d{2,4}", text)

    if matches:
        return matches[0]

    return text if text else "UNKNOWN"


def preprocess_plate(plate_crop):
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    return thresh


os.makedirs("plate_crops", exist_ok=True)

for result in results:
    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        plate_crop = image[y1:y2, x1:x2]

        crop_path = f"plate_crops/plate_{i}.jpg"
        cv2.imwrite(crop_path, plate_crop)

        processed_plate = preprocess_plate(plate_crop)
        cv2.imwrite(f"plate_crops/plate_processed_{i}.jpg", processed_plate)

        print("Plaka tespit güveni:", conf)

        ocr_results = reader.readtext(processed_plate)

        plate_text = ""

        for bbox, text, prob in ocr_results:
            cleaned = clean_text(text)
            print("OCR:", cleaned, "Güven:", prob)

            if prob > 0.15 and len(cleaned) >= 2:
                plate_text += cleaned + " "

        formatted_plate = format_plate(plate_text)

        print("Okunan Plaka:", formatted_plate)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            formatted_plate,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

cv2.imshow("Plate Detection Test", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
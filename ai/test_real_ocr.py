import os
import cv2
import re
import easyocr
from collections import Counter

reader = easyocr.Reader(["en"])

FOLDER = "real_plates_good"

def clean_text(text):
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text

def format_plate(text):
    text = clean_text(text)
    matches = re.findall(r"\d{2}[A-Z]{1,3}\d{2,4}", text)

    if not matches:
        return "UNKNOWN"

    plate = max(matches, key=len)

    if len(plate) > 8:
        plate = plate[:8]

    return plate

def ocr_variants(img):
    candidates = []

    variants = []

    variants.append(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    variants.append(big)

    _, thresh = cv2.threshold(big, 120, 255, cv2.THRESH_BINARY)
    variants.append(thresh)

    inv = cv2.bitwise_not(thresh)
    variants.append(inv)

    for v in variants:
        results = reader.readtext(v)
        text = ""

        for _, t, prob in results:
            if prob > 0.10:
                text += clean_text(t) + " "

        plate = format_plate(text)

        if plate != "UNKNOWN":
            candidates.append(plate)

    if not candidates:
        return "UNKNOWN"

    return Counter(candidates).most_common(1)[0][0]

correct = 0
total = 0

for file in os.listdir(FOLDER):
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    true_plate = file.split("_")[0].replace(".jpg", "").replace(".png", "").upper()

    img_path = os.path.join(FOLDER, file)
    img = cv2.imread(img_path)

    if img is None:
        continue

    pred = ocr_variants(img)

    print(f"Gerçek: {true_plate} | Tahmin: {pred}")

    if pred == true_plate:
        correct += 1

    total += 1

print(f"\nDoğruluk: {correct}/{total}")
if total > 0:
    print(f"Oran: %{(correct / total) * 100:.2f}")
import easyocr
import cv2

# OCR başlat
reader = easyocr.Reader(['en'])

# test edeceğimiz görüntü (snapshots içinden seç)
image_path = "snapshots/gidis_120_20260422_173441.jpg"

# resmi oku
image = cv2.imread(image_path)

# OCR çalıştır
results = reader.readtext(image)

# sonuçları yazdır
for (bbox, text, prob) in results:
    print(f"Bulunan: {text} | Güven: {prob}")
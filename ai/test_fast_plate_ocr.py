from fast_plate_ocr import LicensePlateRecognizer

# MODELİ DOĞRU TANIMLA
model = LicensePlateRecognizer("cct-s-v2-global-model")

# BURAYA SENİN DOSYA ADIN
image_path = "plate_crops/gidis_336_20260424_144321_0.jpg"

# DOĞRU KULLANIM
result = model(image_path)

print("Sonuç:", result)
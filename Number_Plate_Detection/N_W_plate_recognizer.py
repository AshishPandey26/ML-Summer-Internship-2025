import cv2
import easyocr
import json
import os
from ultralytics import YOLO

# ===== Step 1: Load or Create Database =====
DB_FILE = "database.json"

if os.path.exists(DB_FILE):
    with open(DB_FILE, "r") as f:
        db = json.load(f)
else:
    db = {}

# ===== Step 2: Registration Menu =====
print("\n--- Plate Registration ---")
while True:
    reg_plate = input("Enter plate number (or press Enter to skip): ").strip().upper()
    if not reg_plate:
        break
    owner_name = input("Enter owner's name: ").strip()
    db[reg_plate] = owner_name
    print(f"✅ Registered {reg_plate} to {owner_name}")

# Save updated DB
with open(DB_FILE, "w") as f:
    json.dump(db, f, indent=2)

# ===== Step 3: Load Image =====
image_path = input("\nEnter image path (e.g., car.jpg): ")
img = cv2.imread(image_path)

if img is None:
    print("❌ Image not found.")
    exit()

# ===== Step 4: Detect Plate Using YOLOv8 =====
model = YOLO("keremberke/license-plate-detection")
results = model(img)[0]

if results.boxes:
    x1, y1, x2, y2 = map(int, results.boxes[0].xyxy[0])
    cropped = img[y1:y2, x1:x2]
    cv2.imwrite("cropped_plate.jpg", cropped)
    print("✅ Plate cropped and saved as cropped_plate.jpg")

    # ===== Step 5: OCR on Cropped Plate =====
    reader = easyocr.Reader(['en'])
    ocr_result = reader.readtext(cropped, detail=0)

    if ocr_result:
        plate_text = ocr_result[0].replace(" ", "").upper()
        print(f"🪪 Detected Plate Number: {plate_text}")

        # ===== Step 6: Match in DB =====
        owner = db.get(plate_text, "❌ Owner not found in database.")
        print(f"👤 Owner: {owner}")
    else:
        print("❌ OCR failed to detect any text.")
else:
    print("❌ No license plate detected in the image.")

import json
import cv2
import easyocr
import os
# from ultralytics import YOLO


import torch
import ultralytics.nn.tasks

torch.serialization.add_safe_globals([
    torch.nn.modules.container.Sequential,
    ultralytics.nn.tasks.DetectionModel
])

from ultralytics import YOLO

MODEL_PATH = "D:\\Dev\\number_plate_checker\\Number_Plate_Detection\\models\\yolov8n.pt"
model = YOLO(MODEL_PATH)



reader = easyocr.Reader(['en'], gpu=False)

DB_FILE = "database.json"

# Load or create DB
if os.path.exists(DB_FILE):
    with open(DB_FILE, "r") as f:
        db = json.load(f)
else:
    db = {}

def detect_and_crop_plate(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Image not found.")
        return None

    results = model.predict(img)[0]

    # Check if any boxes detected (results.boxes can be tensor or list)
    if results.boxes and len(results.boxes) > 0:
        # Get first detected box coordinates
        x1, y1, x2, y2 = map(int, results.boxes[0].xyxy[0])
        crop = img[y1:y2, x1:x2]

        # Check if crop is valid
        if crop.size == 0:
            print("❌ Cropped plate image is empty.")
            return None
        return crop
    else:
        print("❌ No plate detected.")
        return None

def extract_plate_number(image):
    ocr_result = reader.readtext(image, detail=0)
    if ocr_result:
        plate = ocr_result[0].upper().replace(" ", "")
        print(f"🪪 Detected Plate Number: {plate}")
        return plate
    else:
        print("❌ OCR failed.")
        return None

def register_plate():
    image_path = input("Enter number plate image path (e.g., car.jpg): ").strip()
    cropped = detect_and_crop_plate(image_path)
    if cropped is None:
        return

    plate = extract_plate_number(cropped)
    if plate:
        owner = input("Enter owner's name: ").strip()
        db[plate] = owner
        with open(DB_FILE, "w") as f:
            json.dump(db, f, indent=2)
        print(f"✅ Registered {plate} to {owner}")

def check_owner():
    image_path = input("Enter image path to check owner: ").strip()
    cropped = detect_and_crop_plate(image_path)
    if cropped is None:
        return

    plate = extract_plate_number(cropped)
    if plate:
        owner = db.get(plate)
        if owner:
            print(f"✅ Match Found: {plate} belongs to {owner}")
        else:
            print("❌ Plate not registered.")

def main_menu():
    while True:
        print("\n--- Menu ---")
        print("1. Check Car Owner")
        print("2. Register New Car Number")
        print("3. Exit")
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            check_owner()
        elif choice == "2":
            register_plate()
        elif choice == "3":
            print("👋 Exiting...")
            break
        else:
            print("❌ Invalid choice.")

if __name__ == "__main__":
    main_menu()

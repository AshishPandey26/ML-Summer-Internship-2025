import json
import easyocr

# Load JSON DB
with open("database.json", "r") as f:
    db = json.load(f)

# Load image
print("Ready to accept image path input.")
image_path = input("Enter image path (e.g., car.jpg): ")
reader = easyocr.Reader(['en'], gpu=False)
results = reader.readtext(image_path)

# Scan results
for _, text, _ in results:
    plate = text.upper().replace(" ", "")
    print(f"Detected: {plate}")
    if plate in db:
        print(f"✅ Match Found: {plate} belongs to {db[plate]}")
        break
else:
    print("❌ Plate not registered.")

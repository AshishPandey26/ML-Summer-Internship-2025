import face_recognition
import os
import cv2
import numpy as np
from pathlib import Path

# Paths
REFERENCE_DIR = "reference_images"
GROUP_DIR = "group_photos"
OUTPUT_DIR = "output_faces"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Load reference images and compute encodings
reference_encodings = []
reference_names = []

for filename in os.listdir(REFERENCE_DIR):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    path = os.path.join(REFERENCE_DIR, filename)
    img = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(img)

    if len(encodings) > 0:
        reference_encodings.append(encodings[0])
        person_name = Path(filename).stem  # filename without extension
        reference_names.append(person_name)
        print(f"Loaded encoding for: {person_name}")
    else:
        print(f"⚠️ No face found in reference image: {filename}")

# Step 2: Process each group photo
for group_filename in os.listdir(GROUP_DIR):
    if not group_filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    group_path = os.path.join(GROUP_DIR, group_filename)
    group_img = face_recognition.load_image_file(group_path)
    face_locations = face_recognition.face_locations(group_img)
    face_encodings = face_recognition.face_encodings(group_img, face_locations)

    print(f"\n📸 Processing {group_filename} - Found {len(face_encodings)} face(s)")

    for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
        # Compare with known people
        matches = face_recognition.compare_faces(reference_encodings, face_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(reference_encodings, face_encoding)

        if any(matches):
            best_match_idx = np.argmin(face_distances)
            name = reference_names[best_match_idx]
        else:
            name = "unknown"

        # Extract face
        top, right, bottom, left = face_location
        face_image = group_img[top:bottom, left:right]
        face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)

        # Create person folder
        person_folder = os.path.join(OUTPUT_DIR, name)
        os.makedirs(person_folder, exist_ok=True)

        # Save cropped face
        save_path = os.path.join(person_folder, f"{group_filename}_face{i+1}.jpg")
        cv2.imwrite(save_path, face_image_bgr)
        print(f"✅ Saved face to {save_path}")

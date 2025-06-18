import cv2
import os
import time

video = cv2.VideoCapture(0)

from config import CASCADE_PATH
facedetect = cv2.CascadeClassifier(CASCADE_PATH)

name = input("Enter Your Name: ")

save_path = f'data/{name}/'
os.makedirs(save_path, exist_ok=True)

i = 0
last_capture_time = time.time()

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    current_time = time.time()

    for (x, y, w, h) in faces:
        # Save only every 0.6 seconds
        if current_time - last_capture_time >= 0.6:
            crop_img = frame[y:y+h, x:x+w]
            resized_img = cv2.resize(crop_img, (100, 100))
            cv2.imwrite(os.path.join(save_path, f'{i}.jpg'), resized_img)
            i += 1
            last_capture_time = current_time

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, str(i), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Collecting Faces", frame)
    if cv2.waitKey(1) == ord('q') or i >= 100:
        break

video.release()
cv2.destroyAllWindows()

import cv2
import os

video = cv2.VideoCapture(0)


from config import CASCADE_PATH
facedetect = cv2.CascadeClassifier(CASCADE_PATH)

# facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

name = input("Enter Your Name: ")
save_path = f'data/{name}/'
os.makedirs(save_path, exist_ok=True)

i = 0
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (100, 100))
        cv2.imwrite(os.path.join(save_path, f'{i}.jpg'), resized_img)
        i += 1
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, str(i), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imshow("Collecting Faces", frame)
    if cv2.waitKey(1) == ord('q') or i >= 100:
        break

video.release()
cv2.destroyAllWindows()

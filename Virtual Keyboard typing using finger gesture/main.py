import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8)


class Button():
    def __init__ (self, pos, text, size =[100,100]):
        self.pos = pos
        self.size = size
        self.text = text
    
    def draw(self, img):
        cv2.rectangle(img, (self.pos), self.size, (255,0,255), cv2.FILLED)
        cv2.putText(img, self.text, (self.pos[0]+25, self.pos[1]+25), 
                    cv2.FONT_HERSHEY_COMPLEX, 3, (255,255,255), 5)
        return img

mybutton = Button([100,100], "Q")

while True:
    success, img = cap.read()
    
    # Detect hands and draw landmarks
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        bboxInfo = hand["bbox"]

        print("Index finger tip:", lmList[8])
     
    img= mybutton.draw(img)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
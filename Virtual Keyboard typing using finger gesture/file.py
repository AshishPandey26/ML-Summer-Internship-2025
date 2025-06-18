import cv2
from cvzone.HandTrackingModule import HandDetector
from time import sleep
import cvzone
from pynput.keyboard import Controller


# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Hand detector
detector = HandDetector(detectionCon=0.8)
keyboard = Controller()

# Virtual keyboard keys
keys = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]
]

finalText = ""

# Define custom colors
button_color = (0, 255, 255)  # Cyan for the buttons
highlight_color = (255, 0, 255)  # Pink for highlighting buttons when hovered
text_color = (255, 255, 255)  # White text on buttons
background_color = (175, 0, 175)  # Purple background for the final text display
click_color = (0, 255, 0)  # Green for the click gesture visual cue

# Button class
class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text

# Create buttons
buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100 * j + 50, 100 * i + 50], key))

# Draw all buttons
def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cvzone.cornerRect(img, (x, y, w, h), 20, rt=0)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, button.text, (x + 20, y + 65),
                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    return img

# Main loop
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)  # returns hands and image

    img = drawAll(img, buttonList)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]

        for button in buttonList:
            x, y = button.pos
            w, h = button.size

            # Check if index fingertip is on button
            if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)
                cv2.putText(img, button.text, (x + 20, y + 65),
                            cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

                # Get distance between index and middle fingertips (for "click" gesture)
                l, _, _ = detector.findDistance(lmList[8][:2], lmList[12][:2], img)

                # Detect if the distance is small enough to simulate a click
                if l < 50:  # Adjust the distance threshold for click
                    # Key press only when distance between fingers is small enough
                    keyboard.press(button.text)
                    finalText += button.text
                    print(f"Key pressed: {button.text}")
                    sleep(0.3)  # Small debounce time to avoid multiple presses
                
                    # Draw a visual cue (green circle) to show click gesture
                    cv2.circle(img, (lmList[8][0], lmList[8][1]), 10, (0, 255, 0), cv2.FILLED)
        
    # Display final text
    cv2.rectangle(img, (50, 350), (700, 450), (175, 0, 175), cv2.FILLED)
    cv2.putText(img, finalText, (60, 430),
                cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

    cv2.imshow("Virtual Keyboard", img)
    cv2.waitKey(1)

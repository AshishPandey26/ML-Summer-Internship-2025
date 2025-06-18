import cv2
import numpy as np

# Load the image
img = cv2.imread("mixshapes.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 150)

shapes_inp = input(print('Enter the name of shape you want to find (circle/rectangle/triangle/square)'))

# Find contours
contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
def shape_pred(s1=None,mode=None):
    flag=0
    s1=s1.capitalize()
    if mode==True:
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(approx)
            if len(approx) == 3:
                shape = "Triangle"
                if shape==s1:
                    cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)
                    cv2.putText(img, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    flag=1
            elif len(approx) == 4:
                aspectRatio = w / float(h)
                shape = "Square" if 0.95 <= aspectRatio <= 1.05 else "Rectangle"
                if shape==s1:
                    cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)
                    cv2.putText(img, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    flag=1
            elif len(approx) == 5:
                shape = "Pentagon"
                if shape==s1:
                    cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)
                    cv2.putText(img, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    flag=1
            elif len(approx) == 6:
                shape = "Hexagon"
                if shape==s1:
                    cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)
                    cv2.putText(img, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    flag=1
            elif len(approx) >= 7:
                shape = "Circle"
                if shape==s1:
                    cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)
                    cv2.putText(img, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    flag=1
            if flag == 0:
                cv2.putText(img, "NOT MATCHED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow("Shape Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(approx)
            if len(approx) == 3:
                shape = "Triangle"
            elif len(approx) == 4:
                aspectRatio = w / float(h)
                shape = "Square" if 0.95 <= aspectRatio <= 1.05 else "Rectangle"
            elif len(approx) == 5:
                shape = "Pentagon"
            elif len(approx) == 6:
                shape = "Hexagon"
            elif len(approx) >= 7:
                shape = "Circle"
            else:
                shape = "Unknown"
            cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)
            cv2.putText(img, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow("Shape Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()   
shape_pred(shapes_inp,True)
# shape_pred("triangle",True)
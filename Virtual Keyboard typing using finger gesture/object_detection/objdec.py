import cv2
import numpy as np

# Read and preprocess the image
img = cv2.imread('shapes.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

# Detect contours
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Loop through contours
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    x, y = approx.ravel()[0], approx.ravel()[1] - 5
    shape = "Circle"

    if len(approx) == 3:
        shape = "Triangle"
    elif len(approx) == 4:
        x1, y1, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"

    # Draw and label shape
    cv2.drawContours(img, [approx], 0, (0, 0, 0), 3)
    cv2.putText(img, shape, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Show final result
cv2.imshow("Shapes Detected", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

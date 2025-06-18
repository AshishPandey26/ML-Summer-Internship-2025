import cv2
import numpy as np

# Load and preprocess image
img = cv2.imread("shapes.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 150)

# Ask for user input
shape_input = input("Enter the name of shape you want to find (circle/rectangle/triangle/square): ").strip().capitalize()

# Find contours
contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Flag to track if shape was found
shape_found = False

# Loop through contours
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
    x, y, w, h = cv2.boundingRect(approx)
    shape = "Unknown"

    if len(approx) == 3:
        shape = "Triangle"
    elif len(approx) == 4:
        aspect_ratio = w / float(h)
        shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
    elif len(approx) > 5:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        shape = "Circle" if 0.85 <= circularity <= 1.15 else "Ellipse"  # use 0.85–1.15 for slight imperfections

    # Check if user-input shape matches detected shape
    if shape == shape_input:
        cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)
        cv2.putText(img, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        shape_found = True

# Show result
if not shape_found:
    print(f"No {shape_input} found.")
else:
    print(f"{shape_input} found and highlighted.")

cv2.imshow("Shape Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

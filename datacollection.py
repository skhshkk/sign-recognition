import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initialize variables
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300
counter = 0
folder = "I:/project2/Sign-Language-detection/Data/Unknown"

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from webcam.")
        break

    hands, img = detector.findHands(img)  # Detect hands in the frame
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]  # Get the bounding box of the hand

        # Create a white image to act as a canvas
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the hand region with padding
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:  # If crop is empty, skip this frame
            continue

        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        # Resize the cropped image and center it on the canvas
        if aspectRatio > 1:
            # Height > Width
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap : wGap + wCal] = imgResize
        else:
            # Width > Height
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap : hGap + hCal, :] = imgResize

        # Show the cropped and processed images
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Display the original frame
    cv2.imshow("Image", img)

    # Save the processed image when 's' is pressed
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", imgWhite)
        print(f"Saved: {counter}")

    # Exit on pressing 'q'
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

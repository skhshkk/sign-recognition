import cv2
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Constants
offset = 20
imgSize = 300
labels = [
    "Fine",
    "Hello",
    "I Love You",
    "Like",
    "Me",
    "No",
    "Thank you",
    "Unknown",
    "Where",
    "Yes",
]

# Load Hand Detector and Classifier
detector = HandDetector(maxHands=1)
classifier = Classifier(
    r"I:/project2/Sign-Language-detection/Model/keras_model.h5",
    r"I:/project2/Sign-Language-detection/Model/labels.txt",
)

# New CSS Styling for Overall UI Background
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(145deg, #4a90e2, #2e86de);
        font-family: 'Helvetica', sans-serif;
        color: #f7f7f7;
    }
    .css-1d391kg {
        background-color: #34495e;
        padding: 20px;
        border-radius: 20px;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2);
    }
    .css-1v3fvcr {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        box-shadow: 0px 6px 15px rgba(0, 0, 0, 0.1);
        padding: 20px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1abc9c, #16a085);
        color: white;
        font-size: 16px;
        padding: 12px 35px;
        border-radius: 30px;
        border: none;
        cursor: pointer;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #16a085, #1abc9c);
        transform: scale(1.05);
        box-shadow: 0px 6px 15px rgba(0, 0, 0, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit UI
st.title("Sign Language Detection")
st.sidebar.title("Controls")
st.sidebar.write("Use the buttons below to control the webcam.")
threshold = st.sidebar.slider("Confidence Threshold", 0.5, 1.0, 0.7, 0.05)
run = st.sidebar.button("Start Webcam")
stop = st.sidebar.button("Stop Webcam")

FRAME_WINDOW = st.empty()
cap = None


def process_frame(img, threshold):
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset : y + h + offset, x - offset : x + w + offset]

        aspectRatio = h / w

        try:
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap : wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap : hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            confidence = max(prediction)

            if confidence < threshold:
                label = "Unknown"
                color = (0, 0, 255)  # Red for unknown
            else:
                label = labels[index]
                color = (255, 152, 0)  # Orange for known labels

            # Display Prediction on Webcam Output
            cv2.rectangle(
                imgOutput,
                (x - offset, y - offset - 70),
                (x - offset + 400, y - offset + 60 - 50),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                imgOutput,
                f"{label} ({confidence*100:.1f}%)",
                (x, y - 30),
                cv2.FONT_HERSHEY_COMPLEX,
                1.5,
                (255, 255, 255),  # White Text
                2,
            )
            cv2.rectangle(
                imgOutput,
                (x - offset, y - offset),
                (x + w + offset, y + h + offset),
                color,
                4,
            )
        except Exception:
            pass

    return imgOutput


# Webcam Control
if run:
    if not cap:
        cap = cv2.VideoCapture(0)
    while run and not stop:
        success, img = cap.read()
        if not success:
            st.error("Failed to access the webcam. Please check your camera.")
            break

        img = process_frame(img, threshold)
        FRAME_WINDOW.image(img, channels="BGR")

    if stop:
        if cap:
            cap.release()
        FRAME_WINDOW.empty()
        st.sidebar.write("Webcam stopped.")
else:
    if cap:
        cap.release()
        FRAME_WINDOW.empty()

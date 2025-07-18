import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
from PIL import Image

# Load model and labels
model = load_model('emotion_model.h5', compile=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("📷 Real-Time Emotion Detection via Webcam")
run_camera = st.button("📸 Capture from Webcam")

if run_camera:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (64, 64))
            face_normalized = face_resized / 255.0
            face_input = np.expand_dims(face_normalized, axis=-1)
            face_input = np.expand_dims(face_input, axis=0)

            predictions = model.predict(face_input)
            emotion = emotion_labels[np.argmax(predictions)]

            cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_rgb, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        st.image(img_rgb, caption="Webcam Capture with Emotion", channels="RGB")
    else:
        st.error("Failed to access webcam.")

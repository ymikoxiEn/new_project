import cv2
import numpy as np
from keras.models import load_model

# Load model
model = load_model('C:/Users/User/Desktop/MIT 216 - Machine Learning/emotion_model.h5', compile=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Optional: Add padding to the face box
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(gray.shape[1], x + w + padding)
        y2 = min(gray.shape[0], y + h + padding)

        # Crop the padded face area
        face = gray[y1:y2, x1:x2]

        # Preprocess
        face_resized = cv2.resize(face, (64, 64))
        face_normalized = face_resized / 255.0
        face_input = np.expand_dims(face_normalized, axis=-1)  # (64, 64, 1)
        face_input = np.expand_dims(face_input, axis=0)        # (1, 64, 64, 1)

        # Predict emotion
        predictions = model.predict(face_input)
        emotion = emotion_labels[np.argmax(predictions)]

        # Draw rectangle using original (non-padded) face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow("Real-Time Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

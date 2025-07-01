import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3
import time

# Load model and labels
model = tf.keras.models.load_model('models/asl_model.h5')
class_names = np.load('models/class_names.npy')

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Capture video
cap = cv2.VideoCapture(0)

prev_prediction = ''
last_spoken_time = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    landmarks = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

        if len(landmarks) == 63:
            prediction = model.predict(np.array([landmarks]), verbose=0)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)

            # Speak only if prediction changed or 3 seconds passed
            current_time = time.time()
            if predicted_class != prev_prediction or current_time - last_spoken_time > 3:
                engine.say(predicted_class)
                engine.runAndWait()
                prev_prediction = predicted_class
                last_spoken_time = current_time

            cv2.putText(frame, f"{predicted_class} ({confidence:.2f})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

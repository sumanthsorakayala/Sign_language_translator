import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load trained model and labels
model = tf.keras.models.load_model('models/asl_model.h5')
class_names = np.load('models/class_names.npy')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip frame horizontally for natural feel
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    landmarks = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract (x, y, z) for all 21 landmarks
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

        # Ensure 63 features (21 points * 3 values)
        if len(landmarks) == 63:
            prediction = model.predict(np.array([landmarks]))
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)

            # Display prediction
            cv2.putText(frame, f"{predicted_class} ({confidence:.2f})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

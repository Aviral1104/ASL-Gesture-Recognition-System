import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mediapipe as mp

def load_data(data_dir):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):  # Ensure it's a directory
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (128, 128))
                    images.append(img)
                    labels.append(label)
    return np.array(images), np.array(labels)

data_dir = 'C:/Users/avi11/Downloads/Hand draft 2/Sign Language for Alphabets'
X, y = load_data(data_dir)

le = LabelEncoder()
y = le.fit_transform(y)

X = X.astype('float32') / 255.0
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), batch_size=32)

model.save('asl_gesture_model.h5')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

loaded_model = tf.keras.models.load_model('asl_gesture_model.h5')

def predict_gesture(frame, model, le):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            h, w, _ = frame.shape
            x_min, x_max, y_min, y_max = w, 0, h, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, x_max = min(x_min, x), max(x_max, x)
                y_min, y_max = min(y_min, y), max(y_max, y)
            
            padding = 20
            hand_img = frame[max(0, y_min-padding):min(h, y_max+padding), 
                             max(0, x_min-padding):min(w, x_max+padding)]
            
            hand_img = cv2.resize(hand_img, (128, 128))
            hand_img = hand_img.astype('float32') / 255.0
            hand_img = np.expand_dims(hand_img, axis=0)
            
            prediction = model.predict(hand_img)
            gesture_class = le.inverse_transform([np.argmax(prediction)])[0]
            confidence = np.max(prediction)
            
            return gesture_class, confidence, frame
    
    return None, None, frame

#  real-time prediction
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gesture, conf, annotated_frame = predict_gesture(frame, loaded_model, le)

    if gesture:
        if gesture == 'unknown' or conf < 0.4:
            display_text = "Unknown Gesture"
        else:
            display_text = f"Gesture: {gesture}"
        cv2.putText(annotated_frame, f"{display_text} ({conf:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('ASL Gesture Recognition', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

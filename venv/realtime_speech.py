import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from holistic_landmark import mediapipe_detection, draw_styled_landmarks
from make_folder import actions
from extract_keypoint import extract_keypoints
import matplotlib.pyplot as plt
import io
import pygame
import random
from gtts import gTTS
import os
import time

# Load model
model = tf.keras.models.load_model(r'C:\Users\david\Documents\my private document\Skripsi Project\KSULI_Bisindo\output\bilstm_model_100(2).h5')

# Generate dynamic colors
def generate_colors(n):
    return [tuple(random.choices(range(50, 256), k=3)) for _ in range(n)]

colors = generate_colors(len(actions))

# Visualization function
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 300), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, f'{actions[num]}: {prob*100:.2f}%', 
                    (10, 85+num*40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

# TTS function
def speak_text(text):
    tts = gTTS(text=text, lang='id')
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    pygame.mixer.init()
    pygame.mixer.music.load(mp3_fp)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        continue

# Variables
sequence = []
sentence = []
predictions = []
threshold = 0.9
last_spoken = ""  # track last spoken word

# Start video capture
cap = cv2.VideoCapture(0)

with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detection
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))

            # Confirm prediction
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    predicted_word = actions[np.argmax(res)]
                    sentence = [predicted_word]

                    # Speak if new word detected
                    if predicted_word != last_spoken:
                        speak_text(predicted_word)
                        last_spoken = predicted_word

            # Visualization probabilities
            image = prob_viz(res, actions, image, colors)

        # Display Sentence
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show
        cv2.imshow('Gesture Recognition', image)

        # Break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

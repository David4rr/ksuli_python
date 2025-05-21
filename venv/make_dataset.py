import os
import cv2
import time
import numpy as np
from holistic_landmark import mediapipe_detection, draw_styled_landmarks, mp_holistic
from extract_keypoint import extract_keypoints
from make_folder import DATA_PATH, actions, no_sequences, sequence_length

# Fungsi untuk mengecek dari sequence mana harus mulai
def get_start_sequence(action):
    action_path = os.path.join(DATA_PATH, action)
    for seq in range(no_sequences):  
        seq_path = os.path.join(action_path, str(seq))
        if not os.listdir(seq_path):  # Jika folder kosong
            return seq  
    return no_sequences  # Jika semua sudah terisi

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Gunakan Mediapipe Holistic
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        start_sequence = get_start_sequence(action)

        for sequence in range(start_sequence, no_sequences):
            # Logic untuk menekan button n terlebih dahulu sebelum mengambil data
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Error: Failed to capture image.")
                    break
                cv2.putText(frame, f"Siap merekam '{action}', Video ke-{sequence+1}", (15, 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, "Tekan 'N' untuk mulai...", (15, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('n'):
                    break
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

            # Countdown 5 detik sebelum rekaman dimulai
            start_time = time.time()
            while time.time() - start_time < 5:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Error: Failed to capture image.")
                    break
                remaining_time = 5 - int(time.time() - start_time)
                cv2.putText(frame, f"Starting in {remaining_time}s", (15, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

            # Rekam video
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Error: Failed to capture image.")
                    break

                # Deteksi pose, wajah, dan tangan dengan Mediapipe
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                # Tampilkan status rekaman
                cv2.putText(image, f'Merekam "{action}" - video {sequence+1}', (15, 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f'Frame {frame_num+1}/{sequence_length}', (15, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                
                cv2.imshow('OpenCV Feed', image)

                # Ekstrak keypoints dan simpan
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Memberikan jeda antar frame
                time.sleep(0.1)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

# Tutup semua jendela setelah selesai
cap.release()
cv2.destroyAllWindows()
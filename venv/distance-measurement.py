import cv2
import mediapipe as mp
import numpy as np
import json

# Inisialisasi Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Baca data kalibrasi dari file JSON
try:
    with open("data/kalibrasi/kalibrasi.json", "r") as f:
        data = json.load(f)
    calibration_samples = data["sample"]
    KNOWN_DISTANCE = data["known_distance"]
    KNOWN_FACE_WIDTH = data["known_width"]
    print("Data kalibrasi berhasil dimuat dari file JSON")
except Exception as e:
    print(f"Error loading calibration data: {e}")
    # Default values jika file tidak ada
    calibration_samples = []
    KNOWN_DISTANCE = 80  # cm
    KNOWN_FACE_WIDTH = 14.3  # cm

focal_length = None

def calculate_focal_length(measured_distance, real_width, pixel_width):
    return (pixel_width * measured_distance) / real_width

def calculate_distance(focal_length, real_width, pixel_width):
    if pixel_width == 0:
        return 0
    return (real_width * focal_length) / pixel_width

def get_face_width_improved(face_landmarks, frame_width, frame_height):
    """Menghitung lebar wajah dengan metode yang lebih akurat"""
    if not face_landmarks:
        return 0
    
    # Titik 234 (kiri) dan 454 (kanan) adalah titik tepi wajah
    left_face = face_landmarks[234]
    right_face = face_landmarks[454]
    
    left_x = left_face.x * frame_width
    right_x = right_face.x * frame_width
    
    return abs(right_x - left_x)

# Jika sudah ada sampel kalibrasi, hitung focal length langsung
if calibration_samples and len(calibration_samples) >= 3:
    median_face_width = np.median(calibration_samples)
    focal_length = calculate_focal_length(KNOWN_DISTANCE, KNOWN_FACE_WIDTH, median_face_width)
    print(f"Focal length dari data kalibrasi: {focal_length:.2f}")
    calibrating = False
else:
    print("Memulai proses kalibrasi...")
    calibrating = True

cap = cv2.VideoCapture(0)

# Proses kalibrasi jika diperlukan
while calibrating:
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb_frame)
    
    if results.face_landmarks:
        face_width = get_face_width_improved(results.face_landmarks.landmark, 
                                           frame.shape[1], frame.shape[0])
        
        # Tampilkan preview
        cv2.rectangle(frame, (50, 50), (400, 150), (0, 0, 0), -1)
        cv2.putText(frame, f"Lebar wajah: {face_width:.1f} px", (60, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Tekan 'c' untuk kalibrasi", (60, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Sampel: {len(calibration_samples)}/5", (60, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=1))
    
    cv2.imshow('Kalibrasi Jarak Wajah', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c') and results.face_landmarks:
        face_width = get_face_width_improved(results.face_landmarks.landmark, 
                                           frame.shape[1], frame.shape[0])
        calibration_samples.append(face_width)
        print(f"Sampel {len(calibration_samples)}: {face_width:.1f} px")
        
        if len(calibration_samples) >= 5:
            median_face_width = np.median(calibration_samples)
            focal_length = calculate_focal_length(KNOWN_DISTANCE, KNOWN_FACE_WIDTH, median_face_width)
            print(f"Kalibrasi selesai! Focal length: {focal_length:.2f}")
            print(f"Median lebar wajah: {median_face_width:.1f} px")
            
            # Simpan data kalibrasi ke JSON
            data = {
                "sample": calibration_samples,
                "known_distance": KNOWN_DISTANCE,
                "known_width": KNOWN_FACE_WIDTH
            }
            with open("data/kalibrasi/kalibrasi.json", "w") as f:
                json.dump(data, f)
            
            calibrating = False
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

# Pengukuran jarak real-time
distance_history = []
SMOOTHING_FACTOR = 5

print("Mulai pengukuran jarak. Tekan 'q' untuk keluar")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb_frame)
    
    if results.face_landmarks:
        face_width_px = get_face_width_improved(results.face_landmarks.landmark, 
                                              frame.shape[1], frame.shape[0])
        
        distance = calculate_distance(focal_length, KNOWN_FACE_WIDTH, face_width_px)
        
        distance_history.append(distance)
        if len(distance_history) > SMOOTHING_FACTOR:
            distance_history.pop(0)
        
        smooth_distance = np.mean(distance_history)
        
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=1))
        
        cv2.rectangle(frame, (20, 20), (350, 120), (0, 0, 0), -1)
        cv2.putText(frame, f"Jarak: {smooth_distance:.1f} cm", (30, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Lebar wajah: {face_width_px:.1f} px", (30, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Focal length: {focal_length:.1f}", (30, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        cv2.putText(frame, "Wajah tidak terdeteksi", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Pengukuran Jarak Wajah Enhanced', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
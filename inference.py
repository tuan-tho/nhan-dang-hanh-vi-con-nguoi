import cv2
import mediapipe as mp
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading

# Cấu hình camera Yoosee
USERNAME = "admin"
PASSWORD = "123456789minhanh"
IP = "192.168.1.17"
RTSP_URL = f"rtsp://{USERNAME}:{PASSWORD}@{IP}:554/onvif1"

# Kiểm tra GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    print("✅ Using GPU")
else:
    print("⚠️ GPU not found, using CPU instead.")

# Load model đã huấn luyện
num_of_timesteps = 7
model_path = f'model/model_{num_of_timesteps}.keras'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"🚨 Model file '{model_path}' not found!")

model = load_model(model_path)
print("✅ Model loaded successfully.")

# Khởi tạo MediaPipe Pose
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils


def make_landmark_timestep(results):
    if not results.pose_landmarks:
        return None

    lm_list = []
    landmarks = results.pose_landmarks.landmark
    base_x, base_y, base_z = landmarks[0].x, landmarks[0].y, landmarks[0].z
    center_x = np.mean([lm.x for lm in landmarks])
    center_y = np.mean([lm.y for lm in landmarks])
    center_z = np.mean([lm.z for lm in landmarks])
    distances = [np.sqrt((lm.x - center_x) ** 2 + (lm.y - center_y) ** 2 + (lm.z - center_z) ** 2) for lm in
                 landmarks[1:]]
    scale_factors = [1.0 / (dist + 1e-6) for dist in distances]

    lm_list.append(0.0)
    lm_list.append(0.0)
    lm_list.append(0.0)
    lm_list.append(landmarks[0].visibility)

    for lm, scale_factor in zip(landmarks[1:], scale_factors):
        lm_list.append((lm.x - base_x) * scale_factor)
        lm_list.append((lm.y - base_y) * scale_factor)
        lm_list.append((lm.z - base_z) * scale_factor)
        lm_list.append(lm.visibility)

    return lm_list


def draw_class_on_image(label, img):
    cv2.putText(img, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return img


def draw_skeleton(results, img):
    """ Vẽ khung xương lên ảnh """
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS,
                              mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                              mpDraw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
    return img


def detect(model, lm_list):
    global label
    lm_list = np.expand_dims(np.array(lm_list), axis=0)
    results = model.predict(lm_list)
    predicted_label_index = np.argmax(results, axis=1)[0]
    classes = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
    confidence = np.max(results, axis=1)[0]
    label = classes[predicted_label_index] if confidence > 0.95 else "neutral"


# Kết nối camera Yoosee
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("⚠️ Không thể kết nối với camera! Kiểm tra lại URL RTSP.")
    exit()

label = "Unknown"
lm_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)
    lm = make_landmark_timestep(results)

    if lm:
        lm_list.append(lm)
        if len(lm_list) == num_of_timesteps:
            threading.Thread(target=detect, args=(model, lm_list,)).start()
            lm_list = []

    # Vẽ khung xương và nhãn lên ảnh
    frame = draw_skeleton(results, frame)
    frame = draw_class_on_image(label, frame)

    cv2.imshow("Human Activity Recognition", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

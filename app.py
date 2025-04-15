import streamlit as st
import cv2
import os
import pickle
import face_recognition
import numpy as np
from ultralytics import YOLO
import dlib
from datetime import datetime


# Load saved embeddings
with open("./fr_embeddings.pkl", "rb") as f:
    saved_encodings = pickle.load(f)

# Load models
yolo = YOLO("./yolov8n-face.pt")
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

# Configuration
dataset_path = "./dataset"
verification_count = 5
threshold = 0.45
attendance_log = {}

def align_face(img, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return img[y1:y2, x1:x2]

def get_embedding(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    enc = face_recognition.face_encodings(img_rgb)
    return enc[0] if enc else None

def verify_identity(identity, test_embedding):
    person_dir = os.path.join(dataset_path, identity)
    if not os.path.exists(person_dir):
        return False

    verified = 0
    checked = 0

    for img_name in os.listdir(person_dir):
        if checked >= verification_count:
            break
        img_path = os.path.join(person_dir, img_name)
        db_img = cv2.imread(img_path)
        if db_img is None:
            continue
        db_results = yolo(db_img)
        if len(db_results[0].boxes) == 0:
            continue
        db_box = db_results[0].boxes[0].xyxy[0].cpu().numpy()
        db_face = align_face(db_img, db_box)
        db_embedding = get_embedding(db_face)
        if db_embedding is None:
            continue
        checked += 1
        if np.linalg.norm(db_embedding - test_embedding) < threshold:
            verified += 1

    return verified >= (verification_count // 2)

def mark_attendance(name):
    if name not in attendance_log:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        attendance_log[name] = timestamp

st.set_page_config(layout="wide")
st.title("🎯 Real-Time Face Recognition with Attendance")

start_button = st.button("📸 Start Webcam")

FRAME_WINDOW = st.empty()
LOG_WINDOW = st.empty()

if start_button:
    cap = cv2.VideoCapture(0)
    st.info("Press 'Stop' in the browser to terminate webcam manually.")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to capture frame.")
            break

        results = yolo(frame)
        if len(results[0].boxes) > 0:
            for box_tensor in results[0].boxes:
                box = box_tensor.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                face = align_face(frame, box)
                embedding = get_embedding(face)

                if embedding is None:
                    name = "Face encoding failed"
                else:
                    name = "Unknown"
                    min_dist = float("inf")
                    for record in saved_encodings:
                        dist = np.linalg.norm(record["embedding"] - embedding)
                        if dist < min_dist:
                            min_dist = dist
                            candidate = record["name"]

                    if min_dist <= threshold and verify_identity(candidate, embedding):
                        name = candidate
                        mark_attendance(name)

                # Draw bounding box and name
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb, channels="RGB")

        # Show attendance log live
        if attendance_log:
            with LOG_WINDOW.container():
                st.subheader("📋 Attendance Log")
                for person, time in attendance_log.items():
                    st.markdown(f"**{person}** at _{time}_")

        if not st.session_state.get("run_webcam", True):
            break

    cap.release()
    FRAME_WINDOW.image(frame_rgb, caption="Final Frame", channels="RGB")
    st.success("✅ Webcam stopped.")

import os
import cv2
import pickle
import numpy as np
from flask import Flask, render_template, Response
from ultralytics import YOLO
import face_recognition
import dlib
from datetime import datetime

# Flask app
app = Flask(__name__)

# Load YOLO and dlib
yolo = YOLO("./yolov8n-face.pt")
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

# Load embeddings
with open("all_embeddings.pkl", "rb") as f:
    saved_encodings = pickle.load(f)

# Globals
DATASET_PATH = "./boosted_data"
SIMILARITY_THRESHOLD = 0.45
VERIFY_COUNT = 5

last_identity = None
cooldown = False

def align_face(img, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return img[int(y1):int(y2), int(x1):int(x2)]

def get_embedding(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    enc = face_recognition.face_encodings(rgb)
    return enc[0] if enc else None

def verify_person(identity, test_embedding):
    person_dir = os.path.join(DATASET_PATH, identity)
    if not os.path.exists(person_dir):
        return False

    verified = 0
    checked = 0

    for img_name in os.listdir(person_dir):
        if checked >= VERIFY_COUNT:
            break
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        results = yolo(img)
        if len(results[0].boxes) == 0:
            continue
        db_box = results[0].boxes[0].xyxy[0].cpu().numpy()
        db_face = align_face(img, db_box)
        db_embedding = get_embedding(db_face)
        if db_embedding is None:
            continue
        checked += 1
        dist = np.linalg.norm(db_embedding - test_embedding)
        if dist < SIMILARITY_THRESHOLD:
            verified += 1

    return verified >= (VERIFY_COUNT // 2)

def generate_frames():
    global last_identity, cooldown
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = yolo(frame)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            face = align_face(frame, (x1, y1, x2, y2))
            embedding = get_embedding(face)
            if embedding is None:
                continue

            identity = "Unknown"
            min_dist = float("inf")

            for record in saved_encodings:
                dist = np.linalg.norm(record["embedding"] - embedding)
                if dist < min_dist:
                    min_dist = dist
                    identity = record["name"]

            if min_dist < SIMILARITY_THRESHOLD and identity != last_identity:
                if verify_person(identity, embedding):
                    last_identity = identity
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    label = f"{identity} ⏰ {timestamp}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    identity = "Unknown"
            elif identity == last_identity:
                continue  # skip duplicate detection
            else:
                identity = "Unknown"

            if identity == "Unknown":
                cv2.putText(frame, identity, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

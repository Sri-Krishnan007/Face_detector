import os
import cv2
import pickle
import face_recognition
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load saved embeddings
with open("./all_embeddings.pkl", "rb") as f:
    saved_encodings = pickle.load(f)

# Load YOLO model
yolo = YOLO("../yolov8n-face.pt")

def align_face(img, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    face = img[y1:y2, x1:x2]
    return face

def get_embedding(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    enc = face_recognition.face_encodings(img_rgb)
    if not enc:
        return None
    return enc[0]

def predict_identity_with_verification(test_img_path, dataset_path="C:/Users/srikr/Desktop/COLLEGE/Project Deploy/VisonStay/boosted_data", verification_count=5, threshold=0.45):
    img = cv2.imread(test_img_path)
    results = yolo(img)
    if len(results[0].boxes) == 0:
        return "No face detected!"

    box = results[0].boxes[0].xyxy[0].cpu().numpy()
    test_face = align_face(img, box)
    test_embedding = get_embedding(test_face)
    if test_embedding is None:
        return "Failed to extract embedding from test image."

    min_dist = float("inf")
    identity = "Unknown"
    for record in saved_encodings:
        dist = np.linalg.norm(record["embedding"] - test_embedding)
        if dist < min_dist:
            min_dist = dist
            identity = record["name"]

    if min_dist > threshold:
        return "No confident match found."

    return f"Initial match: {identity} (distance={min_dist:.4f})"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/snap', methods=['POST'])
def snap():
    # Here, you would save the photo (for now, it's just a placeholder)
    image_file = request.files['image']
    image_path = "./static/uploaded_image.jpg"
    image_file.save(image_path)
    
    # Run face recognition
    result = predict_identity_with_verification(image_path)
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)

import base64
import io
import time
import re # For parsing the identity string
from datetime import datetime
import collections
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify

# Import the recognizer module - this will run its initial setup code
# (loading models, embeddings, etc.) when the Flask app starts.
try:
    import recognizer
except ImportError:
    print("\n[ERROR] Failed to import recognizer.py.")
    print("Make sure recognizer.py is in the same directory as app.py and")
    print("all its dependencies (torch, ultralytics, face_recognition, etc.) are installed.")
    exit()
except Exception as e:
    print(f"\n[ERROR] An error occurred during recognizer import: {e}")
    print("Please check recognizer.py for issues during its global setup.")
    exit()

print("[INFO] Recognizer module imported successfully.")

app = Flask(__name__)

# --- Global State (for Liveness Buffer) ---
# WARNING: This global buffer is suitable for a single user demo.
# It will not work correctly with multiple concurrent users.
frame_buffer = collections.deque(maxlen=recognizer.LIVENESS_BUFFER_SIZE)
print(f"[INFO] Initialized global frame buffer with maxlen={recognizer.LIVENESS_BUFFER_SIZE}")

# --- Helper Function to Parse Recognizer Output ---
def parse_identity_string(identity_str):
    """ Parses the 'identity' string from recognizer output. """
    name = "Unknown"
    distance = None

    if identity_str == "Spoof Detected":
        name = "Spoof Detected"
    elif "Unknown" in identity_str or "Fail" in identity_str or "Processing" in identity_str:
        name = identity_str # Keep the full message
        # Try to extract distance if present (e.g., "Unknown (0.88)")
        match = re.search(r'\(([\d.]+)\)', identity_str)
        if match:
            try:
                distance = float(match.group(1))
            except ValueError:
                pass # Ignore if conversion fails
    else: # Assumes format "Name (distance)" or just "Name" if verification failed differently
        match = re.search(r'^(.*?)\s*\(([\d.]+)\)$', identity_str)
        if match:
            name = match.group(1).strip()
            try:
                distance = float(match.group(2))
            except ValueError:
                name = identity_str # Fallback if distance part isn't float
                distance = None
        else:
            # If no distance found, assume the whole string is the name/status
            name = identity_str

    return name, distance

# --- Flask Routes ---

@app.route('/')
def index():
    """ Renders the main HTML page. """
    return render_template('index.html')

@app.route('/detectface', methods=['POST'])
def detect_face_api():
    """ API endpoint to receive a frame, process it, and return results. """
    start_req_time = time.time()
    data = request.get_json()

    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400

    try:
        # Decode base64 image string
        image_data = data['image'].split(',')[1] # Remove header e.g., "data:image/jpeg;base64,"
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            print("[WARN] Could not decode image from base64 data.")
            return jsonify({"error": "Could not decode image"}), 400

    except Exception as e:
        print(f"[ERROR] Error decoding image: {e}")
        return jsonify({"error": f"Error decoding image: {e}"}), 400

    # --- Add frame to buffer (use a copy) ---
    frame_buffer.append(frame.copy())
    # print(f"[DEBUG] Frame added to buffer. Buffer size: {len(frame_buffer)}")

    # --- Process frame using the imported recognizer function ---
    process_start_time = time.time()
    try:
        # Use the globally loaded models and settings from the recognizer module
        annotations = recognizer.process_frame_for_recognition(
            frame,
            frame_buffer, # Pass the global buffer
            recognizer.yolo,
            recognizer.saved_encodings,
            recognizer.DATASET_PATH,
            recognizer.VERIFICATION_COUNT,
            recognizer.THRESHOLD
        )
    except Exception as e:
        print(f"[ERROR] Error during process_frame_for_recognition: {e}")
        # If the recognizer fails, we should still return a response
        return jsonify({
            "name": "Error",
            "distance": "N/A",
            "time": datetime.now().strftime("%H:%M:%S.%f")[:-3], # Add milliseconds
            "liveness": "N/A",
            "message": f"Processing error: {e}"
        }), 500 # Internal Server Error

    process_end_time = time.time()
    processing_duration = process_end_time - process_start_time
    total_duration = process_end_time - start_req_time

    # --- Prepare results ---
    result = {
        "name": "N/A",
        "distance": "N/A",
        "time": datetime.now().strftime("%H:%M:%S.%f")[:-3], # Add milliseconds
        "liveness": "N/A",
        "message": "No face detected or processed."
    }

    if annotations:
        # Process the first detected face only for this API response
        first_ann = annotations[0]
        identity_str = first_ann['identity']
        is_live = first_ann.get('live', None) # Get liveness status (might be None if check not ready/disabled)

        name, distance = parse_identity_string(identity_str)

        liveness_status = "N/A"
        if recognizer.LIVENESS_ENABLED:
            if is_live is True:
                liveness_status = "Live"
            elif is_live is False:
                liveness_status = "Failed (Spoof?)"
                # Ensure name reflects spoof if detected by liveness specifically
                if name != "Spoof Detected":
                     name = f"{name} (Liveness Failed)"
            else: # is_live is None (buffer not full)
                liveness_status = "Checking..."
        else: # Liveness disabled
            liveness_status = "Disabled"


        result = {
            "name": name,
            "distance": f"{distance:.2f}" if distance is not None else "N/A",
            "time": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "liveness": liveness_status,
            "message": f"Processed '{identity_str}'. Liveness: {liveness_status}. (Proc: {processing_duration:.3f}s, Total: {total_duration:.3f}s)"
        }
        # print(f"[DEBUG] API Result: {result}")

    else:
        # No faces detected in this frame, but check background liveness if enabled and buffer full
        liveness_status = "N/A"
        background_spoof = False
        if recognizer.LIVENESS_ENABLED and len(frame_buffer) == recognizer.LIVENESS_BUFFER_SIZE:
            try:
                is_live = recognizer.detect_liveness(list(frame_buffer))
                if is_live is False:
                    liveness_status = "Failed (Spoof?)"
                    background_spoof = True
                elif is_live is True:
                    liveness_status = "Live"
                else: # Should not happen if buffer is full, but handle defensively
                     liveness_status = "Unknown"
            except Exception as e:
                print(f"[ERROR] Error during background liveness check: {e}")
                liveness_status = "Error"

        elif recognizer.LIVENESS_ENABLED:
            liveness_status = "Checking..."
        else:
            liveness_status = "Disabled"

        result["liveness"] = liveness_status
        if background_spoof:
            result["name"] = "Spoof Detected (Background)"
            result["message"] = f"Potential spoof detected in background. Liveness: {liveness_status}. (Proc: {processing_duration:.3f}s, Total: {total_duration:.3f}s)"
        else:
            result["message"] = f"No face detected. Liveness: {liveness_status}. (Proc: {processing_duration:.3f}s, Total: {total_duration:.3f}s)"
        # print(f"[DEBUG] API Result (No Face): {result}")


    return jsonify(result)

if __name__ == '__main__':
    print("[INFO] Starting Flask development server...")
    # Use host='0.0.0.0' to make it accessible on your network
    # Use debug=False for any kind of production/shared environment
    app.run(host='0.0.0.0', port=5000, debug=True)
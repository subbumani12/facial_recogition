from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import os

app = Flask(__name__)

# Use relative paths for model files (ensure these files are in the same directory as app.py)
model_path = "face_recognizer.yml"
label_map_path = "label_map.npy"

# Check if the model files exist; if not, raise an error
if not os.path.exists(model_path):
    raise FileNotFoundError(f"{model_path} does not exist. Please copy it into the project directory.")
if not os.path.exists(label_map_path):
    raise FileNotFoundError(f"{label_map_path} does not exist. Please copy it into the project directory.")

# Load the trained model and label map
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)
label_map = np.load(label_map_path, allow_pickle=True).item()
reverse_label_map = {v: k for k, v in label_map.items()}

# Initialize face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (100, 100))
            label, confidence = recognizer.predict(face_img)
            # Adjust threshold as needed
            if confidence > 70:
                name = "Unauthorized"
            else:
                name = reverse_label_map.get(label, "Unknown")
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255) if name == "Unauthorized" else (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

from flask import Flask, Response, request, redirect, url_for, render_template_string
import cv2
import torch
import time
from datetime import datetime
import os

app = Flask(__name__)

# ==============================
# LOGIN CREDENTIALS
# ==============================
USERNAME = "admin"
PASSWORD = "1234"

# ==============================
# YOLO MODEL
# ==============================
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

# ==============================
# CAMERA
# ==============================
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

ret, frame1 = cap.read()
ret, frame2 = cap.read()

save_path = "data/captured_images"
if not os.path.exists(save_path):
    os.makedirs(save_path)

last_capture_time = 0

# ==============================
# VIDEO STREAM FUNCTION
# ==============================
def generate_frames():
    global frame1, frame2, last_capture_time

    while True:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion = cv2.countNonZero(thresh) > 5000

        frame = frame1.copy()

        results = model(frame)
        detections = results.pandas().xyxy[0]

        person_detected = False

        for _, row in detections.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = row['name']
            conf = row['confidence']

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            if label == "person":
                person_detected = True

        if motion:
            cv2.putText(frame, "Motion Detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        if person_detected:
            cv2.putText(frame, "Person Detected", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)

        current_time = time.time()

        if motion and person_detected:
            cv2.putText(frame, "ALERT!", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

            if current_time - last_capture_time > 3:
                filename = f"intruder_{datetime.now().strftime('%H-%M-%S')}.jpg"
                cv2.imwrite(os.path.join(save_path, filename), frame)
                last_capture_time = current_time

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        frame1 = frame2
        ret, frame2 = cap.read()


# ==============================
# LOGIN PAGE
# ==============================
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form['username']
        pwd = request.form['password']

        if user == USERNAME and pwd == PASSWORD:
            return redirect(url_for('video'))
        else:
            return "<h3>Invalid Credentials</h3>"

    return render_template_string("""
        <h2>Login to CCTV System</h2>
        <form method="POST">
            Username: <input type="text" name="username"><br><br>
            Password: <input type="password" name="password"><br><br>
            <input type="submit" value="Login">
        </form>
    """)


# ==============================
# VIDEO PAGE
# ==============================
@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
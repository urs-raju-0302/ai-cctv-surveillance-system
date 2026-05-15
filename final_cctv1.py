import cv2
import torch
import os
import time
from datetime import datetime
import winsound

# ==============================
# SETUP
# ==============================
save_path = "data/captured_images"
log_path = "logs/activity_log.txt"

if not os.path.exists(save_path):
    os.makedirs(save_path)

if not os.path.exists("logs"):
    os.makedirs("logs")

# ==============================
# LOGGER
# ==============================
def log_event(msg):
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a") as f:
        f.write(f"{t} → {msg}\n")

# ==============================
# LOAD YOLO MODEL
# ==============================
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

# OPTIONAL: detect only person (faster)
# model.classes = [0]

# ==============================
# CAMERA
# ==============================
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

ret, frame1 = cap.read()
ret, frame2 = cap.read()

# ==============================
# VARIABLES
# ==============================
stored_frame = None
state = "Cam1"
last_capture_time = 0

print("Press 's' to switch Cam1 ↔ Cam2")
print("Press 'q' to exit")

# ==============================
# MAIN LOOP
# ==============================
while True:

    # MOTION DETECTION
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    motion = cv2.countNonZero(thresh) > 5000

    frame = frame1.copy()

    # YOLO DETECTION
    results = model(frame)
    detections = results.pandas().xyxy[0]

    person_detected = False

    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        conf = row['confidence']

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        if label == "person":
            person_detected = True

    # LABELS
    if motion:
        cv2.putText(frame, "Motion Detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    if person_detected:
        cv2.putText(frame, "Person Detected", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)

    current_time = time.time()

    # ALERT + IMAGE CAPTURE
    if motion and person_detected:
        cv2.putText(frame, "ALERT! INTRUDER", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        if current_time - last_capture_time > 3:

            filename = f"intruder_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
            cv2.imwrite(os.path.join(save_path, filename), frame)

            log_event("Intruder detected (YOLO) + image saved")

            for _ in range(3):
                winsound.Beep(1500, 300)

            last_capture_time = current_time

    # MULTI-CAMERA SIMULATION
    if person_detected:

        if stored_frame is None:
            stored_frame = frame
            log_event("Frame stored from Cam1")

        else:
            if state == "Cam2":
                cv2.putText(frame, "PATH: Cam1 → Cam2", (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)

                log_event("Movement detected Cam1 → Cam2")

                winsound.Beep(2000, 500)

    # SHOW STATE
    cv2.putText(frame, f"Current: {state}", (10, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("SMART CCTV (YOLO)", frame)

    # UPDATE FRAMES
    frame1 = frame2
    ret, frame2 = cap.read()

    # KEY CONTROLS
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        state = "Cam2" if state == "Cam1" else "Cam1"
        log_event(f"Switched to {state}")

    elif key == ord('q'):
        break

# CLEANUP
cap.release()
cv2.destroyAllWindows()
import cv2
import os
import time
from datetime import datetime
import winsound

# ==============================
# SETUP FOLDERS
# ==============================
save_path = "data/captured_images"
log_path = "logs/activity_log.txt"

if not os.path.exists(save_path):
    os.makedirs(save_path)

if not os.path.exists("logs"):
    os.makedirs("logs")

# ==============================
# LOGGER FUNCTION
# ==============================
def log_event(message):
    time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a") as file:
        file.write(f"{time_stamp} → {message}\n")

# ==============================
# FACE DETECTOR
# ==============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ==============================
# MOTION DETECTION FUNCTION
# ==============================
def detect_motion(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    max_contour = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1500 and area > max_area:
            max_area = area
            max_contour = contour

    if max_contour is not None:
        motion_detected = True
        x, y, w, h = cv2.boundingRect(max_contour)

        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(frame1, "Moving Object", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame1, motion_detected

# ==============================
# FACE DETECTION FUNCTION
# ==============================
def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    face_detected = False
    face_img = None

    for (x, y, w, h) in faces:
        face_detected = True
        face_img = gray[y:y+h, x:x+w]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        cv2.putText(frame, "Face Detected", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return frame, face_detected, face_img

# ==============================
# FACE COMPARISON
# ==============================
def compare_faces(face1, face2):
    face1 = cv2.resize(face1, (100, 100))
    face2 = cv2.resize(face2, (100, 100))

    hist1 = cv2.calcHist([face1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([face2], [0], None, [256], [0, 256])

    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity > 0.7

# ==============================
# MAIN PROGRAM
# ==============================
cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
ret, frame2 = cap.read()

stored_face = None
state = "Cam1"
last_capture_time = 0

print("Press 's' to switch Cam1 ↔ Cam2")
print("Press 'q' to exit")

while True:

    frame, motion = detect_motion(frame1, frame2)
    frame, face_detected, face_img = detect_face(frame)

    # Labels
    if motion:
        cv2.putText(frame, "Motion Detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    if face_detected:
        cv2.putText(frame, "Human Detected", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    current_time = time.time()

    # IMAGE + ALERT
    if motion and face_detected:
        cv2.putText(frame, "ALERT! INTRUDER", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        if current_time - last_capture_time > 3:

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"intruder_{timestamp}.jpg"
            filepath = os.path.join(save_path, filename)

            cv2.imwrite(filepath, frame)

            print(f"[ALERT] Image saved: {filename}")
            log_event("Intruder detected and image captured")

            for i in range(3):
                winsound.Beep(1500, 300)

            last_capture_time = current_time

    # MULTI-CAMERA SIMULATION
    if face_detected and face_img is not None:

        if stored_face is None:
            stored_face = face_img
            log_event("Face stored from Cam1")

        else:
            if compare_faces(stored_face, face_img) and state == "Cam2":
                cv2.putText(frame, "PATH: Cam1 -> Cam2", (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

                print("[TRACKING] Movement detected")
                log_event("Person moved Cam1 → Cam2")

                winsound.Beep(2000, 500)

    # Show state
    cv2.putText(frame, f"Current: {state}", (10, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Smart CCTV System", frame)

    frame1 = frame2
    ret, frame2 = cap.read()

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if state == "Cam1":
            state = "Cam2"
            log_event("Switched to Cam2")
        else:
            state = "Cam1"
            log_event("Switched to Cam1")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
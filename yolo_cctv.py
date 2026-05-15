import cv2
import torch

# Load YOLOv5 nano model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

# Only detect person (optional but faster)
#model.classes = [0]  # 0 = person

cap = cv2.VideoCapture(0)

print("Press 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Convert results to pandas dataframe
    detections = results.pandas().xyxy[0]

    for index, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = row['confidence']
        label = row['name']

        text = f"{label} {confidence:.2f}"

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("YOLO CCTV", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
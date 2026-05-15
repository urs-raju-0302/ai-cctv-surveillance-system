import cv2

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

    # Find biggest moving object
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1500 and area > max_area:
            max_area = area
            max_contour = contour

    if max_contour is not None:
        motion_detected = True
        x, y, w, h = cv2.boundingRect(max_contour)

        # Draw one clean box
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 3)

        # Add label
        cv2.putText(frame1, "Moving Object", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame1, motion_detected
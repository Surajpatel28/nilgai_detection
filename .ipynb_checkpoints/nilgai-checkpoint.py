import cv2
import time
import winsound   # works on Windows
from ultralytics import YOLO

# Load trained model
model = YOLO("best.pt")  # replace with your trained weights

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # optional: set resolution
cap.set(4, 480)

confidence_threshold = 0.80  # only alert above 80%

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 inference
    results = model.predict(frame, verbose=False)[0]

    # Draw boxes and check for Nilgai
    detected = False
    for box in results.boxes:
        conf = float(box.conf)
        cls = int(box.cls)

        if conf >= confidence_threshold and cls == 0:  # Nilgai class (id=0)
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            label = model.names[cls]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            detected = True

    # If Nilgai detected → beep
    if detected:
        print("Nilgai detected!")
        winsound.Beep(1000, 1000)  # frequency=1000Hz, duration=500ms

    # Display video
    cv2.imshow("Nilgai Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

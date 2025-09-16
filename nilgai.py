import cv2
import time
from ultralytics import YOLO

# Load trained model
model = YOLO("best.pt")  # replace with your model path

# Start webcam
cap = cv2.VideoCapture(0)

last_alert = 0
alert_interval = 5  # seconds
confidence_threshold = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 inference
    results = model(frame)[0]  # first (and only) result

    # Draw boxes and check for Nilgai
    for box in results.boxes:
        conf = box.conf.item()
        cls = int(box.cls.item())

        if conf >= confidence_threshold and cls == 0:  # Nilgai class
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Nilgai {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Alert with delay
            if time.time() - last_alert > alert_interval:
                print("Nilgai detected!")
                last_alert = time.time()

    # Display video
    cv2.imshow("Nilgai Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


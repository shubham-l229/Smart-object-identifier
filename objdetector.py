# Install dependencies first (run in terminal):
# pip install ultralytics opencv-python wikipedia

import cv2
import wikipedia
from ultralytics import YOLO

# Load YOLOv8 (Nano model is fastest)
model = YOLO("yolov8n.pt")

def get_object_info(obj_name):
    """Fetch short details about the object from Wikipedia"""
    try:
        summary = wikipedia.summary(obj_name, sentences=1)
        return summary
    except:
        return "No details available."

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO on current frame
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    # Get detections
    for r in results[0].boxes:
        cls_id = int(r.cls[0])
        conf = float(r.conf[0])
        name = model.names[cls_id]

        # Bounding box
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Fetch object info
        info = get_object_info(name)

        # Display text
        cv2.putText(annotated_frame, f"{name} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(annotated_frame, info[:60], (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show webcam feed
    cv2.imshow("Object Detection + Info", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO
import time
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..","models", "can.pt")
output_file = "multi_can_output.mp4"
min_confidence = 0.5

# Load the trained model
model = YOLO(MODEL_PATH)

# Open the webcam
cap = cv2.VideoCapture(0)
# Get the width and height of the video frames
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

while True:
    start_time = time.time()
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame,conf=min_confidence, verbose=False)
    annotated_frame = results[0].plot()

    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 40), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)

    end_time = time.time()
    fps = 1 / (end_time - start_time)
    fps_text = f"FPS: {fps:.2f}"

    class_counts = {}
    for result in results:
        for c in result.boxes.cls:
            class_name = model.names[int(c)]
            class_id = int(c)
            class_counts[class_id] = class_counts.get(class_id, 0) + 1

    cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    x_offset = 150
    for name, count in class_counts.items():
        text = f"{model.names[name]}: {count}"
        cv2.putText(annotated_frame, text, (x_offset, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        x_offset += 150

    out.write(annotated_frame)
    cv2.imshow("Can Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Output saved to {output_file}")

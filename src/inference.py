import cv2
from ultralytics import YOLO
import time
import os

#config
MODEL_PATH = os.path.join('..', 'models', 'purdeys_v1.pt') # Point to your trained .pt file
OUTPUT_FILENAME = 'purdeys_detection_demo.mp4'
CONFIDENCE_THRESHOLD = 0.5  # Ignore detections below 50% confidence

#load model
print("Loading model...")
model = YOLO(MODEL_PATH)

# Initialize Webcam
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30 # standard webcam fps

# Initialize Video Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, fps, (width, height))

print(f"--- Inference Started ---")
print(f"Recording to {OUTPUT_FILENAME}. Press 'q' to stop.")

while True:
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run inference on the frame
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

    # Visualize the results on the frame
    # plot() draws bounding boxes and labels automatically
    annotated_frame = results[0].plot()

    # Calculate FPS for display (Engineering touch)
    end_time = time.time()
    fps_calc = 1 / (end_time - start_time)
    
    # Add FPS overlay
    cv2.putText(annotated_frame, f"FPS: {fps_calc:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write frame to video file
    out.write(annotated_frame)

    # Display to screen
    cv2.imshow('Purdeys Detection - YOLOv8', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Saved recording to {OUTPUT_FILENAME}")
print(f"--- Inference Ended ---")
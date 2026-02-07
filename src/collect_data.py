import cv2
import os
import time

# Get the directory where THIS project is located (src/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to project root, then into data/raw
SAVE_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'raw')

# Create directory if it doesn't exist
os.makedirs(SAVE_PATH, exist_ok=True)

# Configuration
CAN_LABEL = "purdeys_can"
BG_LABEL = "background"

cap = cv2.VideoCapture(0)

# Verify webcam opened
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print(f"--- Data Collection Started ---")
print(f"Saving to: {os.path.abspath(SAVE_PATH)}")
print(f"-----------------------------")
print(f"[s] Save CAN frame (Positive)")
print(f"[b] Save BACKGROUND frame (Negative)")
print(f"[q] Quit")

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break

    # copy frame for display so text doesn't get saved to image
    display_frame = frame.copy()
    
    # UI Overlay
    cv2.putText(display_frame, "Press 's' for CAN, 'b' for BACKGROUND", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Data Collection', display_frame)

    key = cv2.waitKey(1) & 0xFF

    # Logic to handle saving
    if key == ord('s') or key == ord('b'):
        timestamp = int(time.time())
        
        if key == ord('s'):
            label = CAN_LABEL
            print(f"Captured CAN image")
        else:
            label = BG_LABEL
            print(f"Captured BACKGROUND image")

        filename = f"{label}_{timestamp}_{count}.jpg"
        filepath = os.path.join(SAVE_PATH, filename)
        
        cv2.imwrite(filepath, frame) # Save the clean raw frame
        count += 1
        
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
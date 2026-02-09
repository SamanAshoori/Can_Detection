import cv2
import os

#config
SAVE_PATH = os.path.join("..","data","raw")
ENERGY_DRINK= "Purdeys_Can"
BEER= "Heineken_Can"

Cap = cv2.VideoCapture(0)

count = 0
while True:
    ret, frame = Cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('e'):
        save_path = os.path.join(SAVE_PATH, ENERGY_DRINK)
        os.makedirs(save_path, exist_ok=True)
        filename = f"{ENERGY_DRINK}_{count}.jpg"
        cv2.imwrite(os.path.join(save_path, filename), frame)
        print(f"Saved {filename}")
        count += 1

    elif key == ord('b'):
        save_path = os.path.join(SAVE_PATH, BEER)
        os.makedirs(save_path, exist_ok=True)
        filename = f"{BEER}_{count}.jpg"
        cv2.imwrite(os.path.join(save_path, filename), frame)
        print(f"Saved {filename}")
        count += 1

    elif key == ord('q'):
        print("Exiting...")
        break

Cap.release()
cv2.destroyAllWindows()
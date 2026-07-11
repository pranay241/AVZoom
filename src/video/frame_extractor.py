import cv2
import os

VIDEO_PATH = "/home/banapuram-pranay-kumar/AI_Portfolio/AVZoom/data/videos/test.mp4"
OUTPUT_DIR = "/home/banapuram-pranay-kumar/AI_Portfolio/AVZoom/data/frames"

os.makedirs(OUTPUT_DIR,exist_ok=True)
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0
while True:
    ret,frame = cap.read()
    if not ret:
        break

    if frame_count % 100 == 0:
        filename = os.path.join(OUTPUT_DIR,f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(filename,frame)

    frame_count+=1

cap.release()
print(f"Extracted{frame_count} frames")

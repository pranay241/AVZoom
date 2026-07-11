import cv2
import os

VIDEO_PATH = "/home/banapuram-pranay-kumar/AI_Portfolio/AVZoom/data/videos/test.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("couldnot open video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames/fps

print(f"FPS           : {fps}")
print(f"Width         : {width}")
print(f"Height        : {height}")
print(f"Frames        : {total_frames}")
print(f"Duration(sec) : {duration:.2f}")

cap.release()

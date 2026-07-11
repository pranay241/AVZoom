import cv2
from src.video.face_detector import FaceDetector

detector = FaceDetector()

cap = cv2.VideoCapture("data/videos/test.mp4")

ret, frame = cap.read()

faces = detector.detect(frame)

print(faces)

import cv2

from src.video.face_detector import FaceDetector
from src.video.tracker import FaceTracker

detector = FaceDetector()
tracker = FaceTracker()

cap = cv2.VideoCapture("data/videos/test.mp4")

while True:
    ret,frame = cap.read()
    print(ret)
    print("Frame read")

    if not ret:
        break
    faces = detector.detect(frame)
    print(f"Faces detected:{len(faces)}")
    tracked_faces = tracker.update(faces)
    print(tracked_faces)

    for face in tracked_faces:
        x,y,w,h = face["bbox"]
        face_id = face["id"]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2,)
        cv2.putText(frame,f"ID{face_id}",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2,)

    cv2.imshow("AVZOOM tracker",frame)

    if not cap.isOpened():
        print("Error: Could not open video!")
        exit()

    if cv2.waitKey(30)&0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

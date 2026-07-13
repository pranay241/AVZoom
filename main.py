import cv2
from src.video.face_detector import FaceDetector
from src.video.tracker import FaceTracker
from src.video.face_mesh import FaceMeshDetector
from src.video.mouth import MouthAnalyzer

detector = FaceDetector()
tracker = FaceTracker()
mesh_detector = FaceMeshDetector()
mouth = MouthAnalyzer()

cap = cv2.VideoCapture("data/videos/test.mp4")
if not cap.isOpened():
    print("Error: Could not open video!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    faces = detector.detect(frame)
    tracked_faces = tracker.update(faces)
    for face in tracked_faces:
        x, y, w, h = face["bbox"]
        face_id = face["id"]
        padding = 30
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue
        mesh_results = mesh_detector.detect(face_crop)
        if mesh_results.multi_face_landmarks:
            face_landmarks = mesh_results.multi_face_landmarks[0]
            crop_h, crop_w, _ = face_crop.shape
            for landmark in face_landmarks.landmark:
                px = int(landmark.x * crop_w)
                py = int(landmark.y * crop_h)
                frame_x = x1 + px
                frame_y = y1 + py
                cv2.circle(
                    frame,
                    (frame_x, frame_y),
                    1,
                    (0, 255, 0),
                    -1
                )
            result = mouth.analyze(
                face_landmarks,
                crop_w,
                crop_h
            )
            cv2.putText(
                frame,
                f"MAR: {result['mar']:.3f}",
                (x, y - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
    cv2.imshow("AVZoom", frame)
    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

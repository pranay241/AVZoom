import cv2
from src.video.face_detector import FaceDetector
from src.video.tracker import FaceTracker
from src.video.face_mesh import FaceMeshDetector
from src.video.mouth import MouthAnalyzer
from src.video.active_speaker import ActiveSpeakerSelector
from src.video.zoom import SmoothZoom


detector = FaceDetector()
tracker = FaceTracker()
mesh_detector = FaceMeshDetector()
mouth = MouthAnalyzer()
selector = ActiveSpeakerSelector()
zoomer = SmoothZoom()
out_writer = None

cap = cv2.VideoCapture("data/videos/test.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
if not cap.isOpened():
    print("Error: Could not open video!")
    exit()

mar_log = []  # each entry: (timestamp, face_id, mar)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = frame_idx / fps
    faces = detector.detect(frame)
    tracked_faces = tracker.update(faces)

    frame_mars = {}  # face_id -> mar, for THIS frame only

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
            result = mouth.analyze(face_landmarks, crop_w, crop_h)

            mar_log.append((timestamp, face_id, result["mar"]))
            frame_mars[face_id] = result["mar"]

            cv2.putText(
                frame, f"ID {face_id} MAR: {result['mar']:.3f}",
                (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )

    if frame_mars:
        active_id = max(frame_mars, key=frame_mars.get)
        

    active_id = selector.update(frame_mars)
    if active_id is not None:
        cv2.putText(
                frame , f" Active : ID {active_id}", (10,30) , cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)


    if active_id is not None:
        active_face = next((f for f in tracked_faces if f["id"] == active_id),None)
        if active_face is not None:
            zoomed = zoomer.get_crop(frame,active_face["bbox"])
            if zoomed is not None:
                if  out_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out_writer = cv2.VideoWriter("output_zoomed.mp4",fourcc,fps,zoomer.output_size)
                    out_writer.write(zoomed)
                    cv2.imshow("Zoomed Output",zoomed)


    cv2.imshow("AVZoom", frame)
    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break
    frame_idx += 1

cap.release()
if out_writer is not None:
    out_writer.release()
cv2.destroyAllWindows()

import numpy as np
np.save("mar_log.npy", np.array(mar_log, dtype=object))
print(f"Saved {len(mar_log)} MAR entries to mar_log.npy")

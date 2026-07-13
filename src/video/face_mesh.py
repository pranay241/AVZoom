import mediapipe as mp
import cv2

class FaceMeshDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=5,refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5)
    def detect(self,frame):
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        return results


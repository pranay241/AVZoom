import cv2
import mediapipe as mp


class FaceDetector:

    def __init__(
        self,
        model_selection=1,
        min_detection_confidence=0.5
    ):

        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence
        )

    def detect(self,frame):
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        faces = []

        if results.detections is None:
            return faces

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            height , width , _ = frame.shape

            x = int(bbox.xmin * width)
            y = int(bbox.ymin * height)

            w = int(bbox.width * width)
            h = int(bbox.height * height)

            confidence = detection.score[0]

            face = { "bbox":(x,y,w,h),"confidence": confidence }
            faces.append(face)

        return faces

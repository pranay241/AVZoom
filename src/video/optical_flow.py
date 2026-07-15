import cv2
import numpy as np

class MouthFlowAnalyzer:
    def __init__(self, crop_size=64, padding_ratio=0.6):
        self.crop_size = crop_size
        self.padding_ratio = padding_ratio
        self.prev_crops = {}  # face_id -> last mouth crop (grayscale)

    def _get_mouth_crop(self, frame, face_landmarks, width, height):
        landmarks = face_landmarks.landmark
        upper = landmarks[13]
        lower = landmarks[14]
        left = landmarks[61]
        right = landmarks[291]

        xs = [upper.x, lower.x, left.x, right.x]
        ys = [upper.y, lower.y, left.y, right.y]

        cx = np.mean(xs) * width
        cy = np.mean(ys) * height
        mouth_w = (max(xs) - min(xs)) * width
        mouth_h = (max(ys) - min(ys)) * height

        half_size = max(mouth_w, mouth_h) * (1 + self.padding_ratio) / 2
        x1 = int(max(0, cx - half_size))
        y1 = int(max(0, cy - half_size))
        x2 = int(min(width, cx + half_size))
        y2 = int(min(height, cy + half_size))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (self.crop_size, self.crop_size))

    def analyze(self, face_id, frame, face_landmarks, width, height):
        current_crop = self._get_mouth_crop(frame, face_landmarks, width, height)
        if current_crop is None:
            return {"flow_magnitude": 0.0}

        prev_crop = self.prev_crops.get(face_id)
        self.prev_crops[face_id] = current_crop

        if prev_crop is None:
            return {"flow_magnitude": 0.0}

        flow = cv2.calcOpticalFlowFarneback(
            prev_crop, current_crop, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return {"flow_magnitude": float(np.mean(magnitude))}

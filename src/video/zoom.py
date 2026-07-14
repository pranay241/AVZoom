import cv2

class SmoothZoom:
    def __init__(self, output_size=(640, 480), smoothing=0.8, padding=40):
        self.output_size = output_size
        self.smoothing = smoothing
        self.padding = padding
        self.smoothed_box = None  # (x1, y1, x2, y2)

    def get_crop(self, frame, bbox):
        x, y, w, h = bbox
        frame_h, frame_w = frame.shape[:2]

        x1 = max(0, x - self.padding)
        y1 = max(0, y - self.padding)
        x2 = min(frame_w, x + w + self.padding)
        y2 = min(frame_h, y + h + self.padding)

        if self.smoothed_box is None:
            self.smoothed_box = [x1, y1, x2, y2]
        else:
            a = self.smoothing
            self.smoothed_box = [
                a * self.smoothed_box[i] + (1 - a) * val
                for i, val in enumerate([x1, y1, x2, y2])
            ]

        sx1, sy1, sx2, sy2 = [int(v) for v in self.smoothed_box]
        crop = frame[sy1:sy2, sx1:sx2]

        if crop.size == 0:
            return None

        return cv2.resize(crop, self.output_size)

from collections import deque

class ActiveSpeakerSelector:
    def __init__(self, window_size=10, switch_margin=0.02):
        self.window_size = window_size
        self.switch_margin = switch_margin
        self.mar_history = {}  # face_id -> deque of recent MAR values
        self.current_speaker = None

    def update(self, frame_mars):
        for face_id, mar in frame_mars.items():
            if face_id not in self.mar_history:
                self.mar_history[face_id] = deque(maxlen=self.window_size)
            self.mar_history[face_id].append(mar)

        avg_mars = {
            fid: sum(hist) / len(hist)
            for fid, hist in self.mar_history.items()
            if fid in frame_mars
        }

        if not avg_mars:
            return self.current_speaker

        candidate = max(avg_mars, key=avg_mars.get)

        if self.current_speaker is None:
            self.current_speaker = candidate
        elif candidate != self.current_speaker:
            current_avg = avg_mars.get(self.current_speaker, 0)
            candidate_avg = avg_mars[candidate]
            if candidate_avg > current_avg + self.switch_margin:
                self.current_speaker = candidate

        return self.current_speaker

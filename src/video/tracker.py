import math

class FaceTracker:
    def __init__(self,max_distance=100):
        self.next_id = 0
        self.tracks={}
        self.max_distance = max_distance

    def _center(self,bbox):
        x,y,w,h = bbox
        return (x+w//2,y+h//2)

    def _distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0] ,p1[1] - p2[1])

    def update(self,faces):
        
        
        used_tracks = set()
        tracked_faces = []

        if len(self.tracks)==0:
            for face in faces:
                center = self._center(face["bbox"])
                self.tracks[self.next_id] = { "center" : center}
                face["id"] =self.next_id
                tracked_faces.append(face)
                self.next_id += 1

            return tracked_faces

        for face in faces:
            center = self._center(face["bbox"])
            best_id = None
            best_distance = float("inf")
            for track_id,track in self.tracks.items():
                if track_id in used_tracks:
                    continue
                distance = self._distance(center,track["center"])

                if distance<best_distance:
                    best_distance = distance
                    best_id = track_id

            if best_distance<self.max_distance:
                face["id"] = best_id
                used_tracks.add(best_id)
                self.tracks[best_id]["center"] = center

            else:
                face["id"] = self.next_id
                self.tracks[self.next_id] = { "center" : center}

                self.next_id +=1

            tracked_faces.append(face)


        return tracked_faces





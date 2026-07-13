import math

class MouthAnalyzer:
    UPPER_LIP = 13
    LOWER_LIP =14 
    LEFT_CORNER = 61
    RIGHT_CORNER = 291

    def __init__(self):
        pass

    def _distance(self,p1,p2):
        return math.hypot(p1[0] - p2[0],p1[1] - p2[1])
    
    def analyze(self,face_landmarks,width,height):
        landmarks = face_landmarks.landmark

        upper = ( landmarks[self.UPPER_LIP].x * width , landmarks[self.UPPER_LIP].y * height)
        lower = (landmarks[self.LOWER_LIP].x * width , landmarks[self.LOWER_LIP].y * height)
        left = ( landmarks[self.LEFT_CORNER].x * width , landmarks[self.LEFT_CORNER].y * height)
        right = (landmarks[self.RIGHT_CORNER].x * width , landmarks[self.RIGHT_CORNER].y * height)

        opening = self._distance(upper,lower)
        mouth_width = self._distance(left,right)


        if mouth_width == 0:
            mar = 0

        else :
            mar = opening / mouth_width

        return {
                "opening" : opening, "width" : mouth_width , "mar" : mar }

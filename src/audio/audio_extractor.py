from moviepy import VideoFileClip
import numpy as np

class AudioExtractor:
    def __init__(self,sample_rate=16000):
        self.sample_rate = sample_rate

    def extract(self,video_path):
        clip = VideoFileClip(video_path)
        audio = clip.audio.to_soundarray(fps=self.sample_rate)
        if audio.ndim > 1:
            audio = audio.mean(axis = 1)

        clip.close()
        return audio,self.sample_rate

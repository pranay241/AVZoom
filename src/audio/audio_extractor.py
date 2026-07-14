import subprocess
import numpy as np
import soundfile as sf
import tempfile
import os

class AudioExtractor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def extract(self, video_path):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", video_path,
                    "-vn",
                    "-acodec", "pcm_s16le",
                    "-ar", str(self.sample_rate),
                    "-ac", "1",
                    tmp_path
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            audio, sr = sf.read(tmp_path)
            return audio, sr
        finally:
            os.remove(tmp_path)

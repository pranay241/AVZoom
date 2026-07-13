from src.audio.audio_extractor import AudioExtractor

extractor = AudioExtractor()
audio, sr = extractor.extract("data/videos/test.mp4")
print(f"Samples: {len(audio)}, Sample rate: {sr}, Duration: {len(audio)/sr:.2f}s")

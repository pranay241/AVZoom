from src.audio.audio_extractor import AudioExtractor

extractor = AudioExtractor()
audio, sr = extractor.extract("data/videos/test.mp4")

print("min:", audio.min())
print("max:", audio.max())
print("std:", audio.std())
print("first 10 samples:", audio[:10])
print("samples at 5s:", audio[5*sr:5*sr+10])

import numpy as np
from src.audio.audio_extractor import AudioExtractor
from src.audio.features import compute_rms, compute_zcr, compute_vad

extractor = AudioExtractor()
audio, sr = extractor.extract("data/videos/test.mp4")

rms = compute_rms(audio, sr)

print("First 10 RMS values (assumed silence):", rms[:10])
print("Noise floor (mean of first 10):", np.mean(rms[:10]))
print("Threshold (1.5x noise floor):", np.mean(rms[:10]) * 1.5)
print()
print("RMS percentiles:")
for p in [5, 10, 25, 50, 75, 90]:
    print(f"  {p}th percentile: {np.percentile(rms, p):.5f}")

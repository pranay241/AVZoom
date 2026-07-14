import numpy as np
import matplotlib.pyplot as plt

from src.audio.audio_extractor import AudioExtractor
from src.audio.features import compute_rms,compute_zcr,compute_vad


extractor = AudioExtractor()
audio, sr = extractor.extract("data/videos/test.mp4")
print(f"Samples: {len(audio)}, Sample rate: {sr}, Duration: {len(audio)/sr:.2f}s")
rms = compute_rms(audio,sr)
zcr = compute_zcr(audio,sr)
vad = compute_vad(rms,zcr)

print(f"RMS frames: {len(rms)}, ZCR frames: {len(zcr)}")
print(f"ZCR - Min: {zcr.min():.5f}, Max: {zcr.max():.5f}, Mean: {zcr.mean():.5f}")

speech_frames = np.sum(vad)
total_frames = len(vad)

print(f"Speech frames: {speech_frames}/{total_frames} ({100*speech_frames/total_frames:.1f}%)")

vad_times = np.arange(len(rms)) * 0.01

mar_data = np.load("mar_log.npy", allow_pickle=True)
timestamps = mar_data[:, 0].astype(float)
face_ids = mar_data[:, 1].astype(int)
mars = mar_data[:, 2].astype(float)

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

axes[0].plot(vad_times, rms)
axes[0].set_ylabel("Audio RMS")

for fid in np.unique(face_ids):
    mask = face_ids == fid
    axes[1].plot(timestamps[mask], mars[mask], label=f"Face {fid}", alpha=0.7)
axes[1].set_ylabel("MAR per face")
axes[1].legend()

axes[2].plot(vad_times, rms > (np.mean(rms[:10]) * 1.5), drawstyle="steps-post")
axes[2].set_ylabel("VAD (speech=1)")
axes[2].set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig("correlation_check.png")
print("saved correlation_check.png")

import numpy as np
import matplotlib.pyplot as plt
from src.audio.audio_extractor import AudioExtractor
from src.audio.features import compute_rms

extractor = AudioExtractor()
audio, sr = extractor.extract("data/videos/test.mp4")
rms = compute_rms(audio, sr)
vad_times = np.arange(len(rms)) * 0.01

mar_data = np.load("mar_log.npy", allow_pickle=True)
flow_data = np.load("flow_log.npy", allow_pickle=True)

mar_times = mar_data[:, 0].astype(float)
mar_face_ids = mar_data[:, 1].astype(int)
mars = mar_data[:, 2].astype(float)

flow_times = flow_data[:, 0].astype(float)
flow_face_ids = flow_data[:, 1].astype(int)
flows = flow_data[:, 2].astype(float)

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

axes[0].plot(vad_times, rms)
axes[0].set_ylabel("Audio RMS")

for fid in np.unique(mar_face_ids):
    mask = mar_face_ids == fid
    axes[1].plot(mar_times[mask], mars[mask], label=f"Face {fid}", alpha=0.7)
axes[1].set_ylabel("MAR per face")
axes[1].legend()

for fid in np.unique(flow_face_ids):
    mask = flow_face_ids == fid
    axes[2].plot(flow_times[mask], flows[mask], label=f"Face {fid}", alpha=0.7)
axes[2].set_ylabel("Optical Flow per face")
axes[2].legend()

axes[3].plot(vad_times, rms > (np.mean(rms[:10]) * 1.5), drawstyle="steps-post")
axes[3].set_ylabel("VAD (speech=1)")
axes[3].set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig("full_correlation_check.png")
print("saved full_correlation_check.png")

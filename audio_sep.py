import pickle
import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d

TALKNET_FPS = 25.0

with open("TalkNet-ASD/demo/pranay_test/pywork/tracks.pckl", "rb") as f:
    tracks = pickle.load(f)
with open("TalkNet-ASD/demo/pranay_test/pywork/scores.pckl", "rb") as f:
    scores = pickle.load(f)

audio, sr = sf.read("data/videos/test_audio_16k.wav")
if audio.ndim > 1:
    audio = audio.mean(axis=1)

audio_times = np.arange(len(audio)) / sr

for track_id, (track, score_arr) in enumerate(zip(tracks, scores)):
    frames = track["track"]["frame"][:len(score_arr)]
    track_times = frames / TALKNET_FPS

    mask = np.zeros(len(audio), dtype=np.float32)

    for i in range(len(score_arr) - 1):
        if score_arr[i] > 0:
            t_start = track_times[i]
            t_end = track_times[i + 1]
            idx_start = np.searchsorted(audio_times, t_start)
            idx_end = np.searchsorted(audio_times, t_end)
            mask[idx_start:idx_end] = 1.0

    smooth_window = int(0.02 * sr)  # 20ms fade
    gain = uniform_filter1d(mask, size=smooth_window)

    isolated = audio * gain
    sf.write(f"speaker_{track_id}_audio.wav", isolated, sr)
    print(f"Saved speaker_{track_id}_audio.wav — active {mask.mean()*100:.1f}% of clip")

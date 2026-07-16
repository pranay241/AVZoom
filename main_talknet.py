import pickle
import numpy as np
import cv2
import subprocess
from src.video.zoom import SmoothZoom

TALKNET_FPS = 25.0

with open("TalkNet-ASD/demo/pranay_test/pywork/tracks.pckl", "rb") as f:
    tracks = pickle.load(f)
with open("TalkNet-ASD/demo/pranay_test/pywork/scores.pckl", "rb") as f:
    scores = pickle.load(f)

track_entries = []
for track_id, (track, score_arr) in enumerate(zip(tracks, scores)):
    frames = track["track"]["frame"]
    bboxes = track["track"]["bbox"]
    n = len(score_arr)
    for i in range(n):
        t = frames[i] / TALKNET_FPS
        x1, y1, x2, y2 = bboxes[i]
        track_entries.append({
            "time": t,
            "track_id": track_id,
            "bbox": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
            "score": float(score_arr[i])
        })

track_entries.sort(key=lambda e: e["time"])
entry_times = np.array([e["time"] for e in track_entries])

def find_best_at_time(query_time, tolerance=1.0 / TALKNET_FPS):
    idx = np.searchsorted(entry_times, query_time)
    candidates = []
    for offset in (-1, 0, 1):
        j = idx + offset
        if 0 <= j < len(track_entries):
            e = track_entries[j]
            if abs(e["time"] - query_time) <= tolerance:
                candidates.append(e)
    if not candidates:
        return None
    return max(candidates, key=lambda e: e["score"])

cap = cv2.VideoCapture("data/videos/test.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30

zoomer = SmoothZoom()
out_writer = None
frame_idx = 0
frames_written = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    query_time = frame_idx / fps
    best = find_best_at_time(query_time)

    if best is not None:
        zoomed = zoomer.get_crop(frame, best["bbox"])
        if zoomed is not None:
            frames_written += 1
            if out_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_writer = cv2.VideoWriter(
                    "output_talknet.mp4", fourcc, fps, zoomer.output_size
                )
            out_writer.write(zoomed)
            cv2.imshow("TalkNet Zoomed Output", zoomed)

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break
    frame_idx += 1

cap.release()
if out_writer is not None:
    out_writer.release()
cv2.destroyAllWindows()

print(f"Total frames: {frame_idx}, frames written: {frames_written}")

if out_writer is not None:
    print("Muxing audio...")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", "output_talknet.mp4", "-i", "data/videos/test.mp4",
            "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest",
            "output_talknet_with_audio.mp4"
        ],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    print("Saved output_talknet_with_audio.mp4")

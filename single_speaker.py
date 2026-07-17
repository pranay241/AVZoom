import pickle
import numpy as np
import cv2
import subprocess

TALKNET_FPS = 25.0
TARGET_ID = 0  # change this to pick which speaker to follow

with open("TalkNet-ASD/demo/pranay_test/pywork/tracks.pckl", "rb") as f:
    tracks = pickle.load(f)

track = tracks[TARGET_ID]
frames = track["track"]["frame"]
bboxes = track["track"]["bbox"]

track_lookup = {}
for i in range(len(frames)):
    t = frames[i] / TALKNET_FPS
    x1, y1, x2, y2 = bboxes[i]
    track_lookup[t] = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

track_times = np.array(sorted(track_lookup.keys()))

def find_bbox_at_time(query_time, tolerance=1.0 / TALKNET_FPS):
    idx = np.searchsorted(track_times, query_time)
    for offset in (0, -1, 1):
        j = idx + offset
        if 0 <= j < len(track_times):
            t = track_times[j]
            if abs(t - query_time) <= tolerance:
                return track_lookup[t]
    return None

from src.video.zoom import SmoothZoom
zoomer = SmoothZoom()

cap = cv2.VideoCapture("data/videos/test.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30

out_writer = None
frame_idx = 0
last_bbox = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    query_time = frame_idx / fps
    bbox = find_bbox_at_time(query_time)
    if bbox is None:
        bbox = last_bbox
    else:
        last_bbox = bbox

    if bbox is not None:
        zoomed = zoomer.get_crop(frame, bbox)
        if zoomed is not None:
            if out_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_writer = cv2.VideoWriter(
                    f"speaker_{TARGET_ID}_cam.mp4", fourcc, fps, zoomer.output_size
                )
            out_writer.write(zoomed)

    frame_idx += 1

cap.release()
if out_writer is not None:
    out_writer.release()

if out_writer is not None:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", f"speaker_{TARGET_ID}_cam.mp4",
            "-i", f"speaker_{TARGET_ID}_audio.wav",
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            f"speaker_{TARGET_ID}_final.mp4"
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    print(f"Saved speaker_{TARGET_ID}_final.mp4")

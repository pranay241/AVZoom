import pickle
import numpy as np

with open("TalkNet-ASD/demo/pranay_test/pywork/tracks.pckl", "rb") as f:
    tracks = pickle.load(f)

with open("TalkNet-ASD/demo/pranay_test/pywork/scores.pckl", "rb") as f:
    scores = pickle.load(f)

for i, t in enumerate(tracks):
    print(f"--- Track {i} ---")
    print("proc_track keys:", t["proc_track"].keys() if isinstance(t["proc_track"], dict) else type(t["proc_track"]))
    for k, v in t["proc_track"].items():
        print(f"  {k}: type={type(v)}, len={len(v) if hasattr(v, '__len__') else 'n/a'}")
    print("track keys:", t["track"].keys() if isinstance(t["track"], dict) else type(t["track"]))

for i, s in enumerate(scores):
    print(f"--- Score array {i} ---")
    print(f"  len={len(s)}, min={s.min():.3f}, max={s.max():.3f}, mean={s.mean():.3f}")

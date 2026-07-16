import pickle

with open("TalkNet-ASD/demo/pranay_test/pywork/tracks.pckl", "rb") as f:
    tracks = pickle.load(f)

with open("TalkNet-ASD/demo/pranay_test/pywork/scores.pckl", "rb") as f:
    scores = pickle.load(f)

print("=== tracks ===")
print(type(tracks), len(tracks))
print(type(tracks[0]))
print(tracks[0] if not isinstance(tracks[0], dict) else list(tracks[0].keys()))

print("=== scores ===")
print(type(scores), len(scores))
print(type(scores[0]))
print(scores[0][:20] if hasattr(scores[0], '__len__') else scores[0])

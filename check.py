import numpy as np

data = np.load("flow_log.npy", allow_pickle=True)
face_ids = data[:, 1].astype(int)
flows = data[:, 2].astype(float)

for fid in np.unique(face_ids):
    mask = face_ids == fid
    print(f"Face {fid}: min={flows[mask].min():.4f}, max={flows[mask].max():.4f}, "
          f"mean={flows[mask].mean():.4f}, std={flows[mask].std():.4f}")

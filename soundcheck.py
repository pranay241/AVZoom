import soundfile as sf
import numpy as np

s1, _ = sf.read("separated_source1.wav")
s2, _ = sf.read("separated_source2.wav")

print("Are they exactly identical?", np.array_equal(s1, s2))
print("Max absolute difference:", np.max(np.abs(s1 - s2)))
print("Correlation:", np.corrcoef(s1, s2)[0, 1])

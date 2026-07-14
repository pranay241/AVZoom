import soundfile as sf
audio, sr = sf.read("test_audio.wav")
print("min:", audio.min(), "max:", audio.max(), "std:", audio.std())

from src.audio.audio_extractor import AudioExtractor
from src.audio.features import compute_stft, compute_mel_spectrogram

extractor = AudioExtractor()
audio, sr = extractor.extract("data/videos/test.mp4")

stft_mag = compute_stft(audio, sr)
mel_spec = compute_mel_spectrogram(audio, sr)

print(f"STFT shape: {stft_mag.shape}")  # (freq_bins, time_frames)
print(f"Mel spectrogram shape: {mel_spec.shape}")  # (n_mels, time_frames)

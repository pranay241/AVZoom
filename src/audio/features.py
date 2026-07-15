import numpy as np
import librosa

def compute_rms(audio,sample_rate,frame_length_ms=25,hop_length_ms=10):
    frame_length = int(sample_rate * frame_length_ms /1000)
    hop_length = int(sample_rate * hop_length_ms /1000)

    rms_values = []

    for start in range(0,len(audio)-frame_length,hop_length):
        window = audio[start:start+frame_length]
        rms = np.sqrt(np.mean(window**2))
        rms_values.append(rms)

    return np.array(rms_values)

def compute_zcr(audio,sample_rate,frame_length_ms = 25,hop_length_ms = 10):
    frame_length = int(sample_rate * frame_length_ms/1000)
    hop_length = int(sample_rate * hop_length_ms / 1000)
    zcr_values = []

    for start in range(0,len(audio)-frame_length,hop_length):
        window = audio[start:start + frame_length]
        signs = np.sign(window)
        signs[signs==0] = 1
        sign_changes = np.sum(np.abs(np.diff(signs))>0)
        zcr = sign_changes / (2*frame_length)
        zcr_values.append(zcr)

    return np.array(zcr_values)


def compute_vad(rms,zcr,silence_frames = 10 , energy_multiplier = 1.5):
    noise_floor = np.mean(rms[:silence_frames])
    energy_threshold = noise_floor * energy_multiplier

    is_speech = rms > energy_threshold

    return is_speech

def compute_stft(audio,sample_rate,n_fft=512,hop_length=160):
    stft_result = librosa.stft(audio,n_fft = n_fft,hop_length = hop_length)
    magnitude = np.abs(stft_result)
    return magnitude

def compute_mel_spectrogram(audio,sample_rate,n_mels = 64,n_fft = 512,hop_length=160):
    mel_spec = librosa.feature.melspectrogram( y = audio , sr = sample_rate ,n_mels = n_mels, n_fft = n_fft , hop_length = hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec,ref = np.max)
    return mel_spec_db

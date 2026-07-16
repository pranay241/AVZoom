from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio

model = separator.from_hparams(
    source="speechbrain/sepformer-wsj02mix",
    savedir="pretrained_models/sepformer-wsj02mix"
)

est_sources = model.separate_file(path="data/videos/test_audio_16k.wav")

torchaudio.save("separated_source1.wav", est_sources[:, :, 0].detach().cpu(), 8000)
torchaudio.save("separated_source2.wav", est_sources[:, :, 1].detach().cpu(), 8000)

print("Saved separated_source1.wav and separated_source2.wav")

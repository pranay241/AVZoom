# AVZoom

**Audio-visual active speaker detection and auto-framing for multi-person video.**

AVZoom detects who is speaking in a multi-person video call and automatically zooms in on them — combining classical computer vision, audio signal processing, and a pretrained deep learning active-speaker-detection model into a single end-to-end pipeline.

---

## Demo

<img width="480" height="360" alt="demo" src="https://github.com/user-attachments/assets/95b96926-601e-40f0-b26c-f840f784ba48" />

| Input | Output |
|---|---|
| <img width="1920" height="1080" alt="Input frame" src="https://github.com/user-attachments/assets/48e0e060-af49-4bda-b11c-42b48b67850b" /> | <img width="1920" height="1080" alt="Auto-zoomed output frame" src="https://github.com/user-attachments/assets/4dde681c-98e1-407c-919d-f63c92859e0f" /> |

---

## Key Features

- 🎯 **Automatic active speaker detection** — combines a pretrained deep learning model (TalkNet-ASD) with a lightweight classical baseline (lip-motion analysis), giving the system both accuracy and a dependency-free fallback.
- 🔍 **Smooth, jitter-free auto-zoom** — exponentially smoothed crop boxes track the speaker without shaky or abrupt framing changes.
- 🎙️ **Multimodal signal fusion** — combines facial landmark geometry, optical flow, and audio energy/voice-activity signals for robust speaker attribution.
- 🔊 **Per-speaker isolated audio tracks** — uses speaker-timing information to generate a clean, individually gated audio track for each person, plus a dedicated single-speaker cam mode pairing one person's video with their own isolated audio.
- 🧩 **Modular pipeline** — face detection, tracking, audio processing, and speaker selection are independent, swappable components.
- ⚡ **Runs entirely on CPU** — no GPU required for inference.

---

## System Architecture

```
                        ┌───────────────────────┐
                        │      Input Video       │
                        └───────────┬───────────┘
                                    │
                ┌───────────────────┴───────────────────┐
                ▼                                         ▼
      ┌───────────────────┐                    ┌────────────────────┐
      │   Visual Pipeline   │                    │   Audio Pipeline    │
      │  Face Detection     │                    │  Extraction (ffmpeg)│
      │  Face Tracking       │                    │  RMS / ZCR / VAD    │
      │  Facial Landmarks    │                    │  STFT / Mel Spec    │
      │  Mouth Aspect Ratio  │                    └──────────┬─────────┘
      │  Optical Flow        │                               │
      └──────────┬───────────┘                               │
                 │                                           │
                 └───────────────────┬───────────────────────┘
                                     ▼
                      ┌───────────────────────────────┐
                      │   Active Speaker Selection      │
                      │  • TalkNet-ASD (deep learning)   │
                      │  • MAR + Flow fusion (classical)│
                      └───────────────┬───────────────┘
                                     ▼
                ┌─────────────────────┴─────────────────────┐
                ▼                                             ▼
   ┌───────────────────────────┐               ┌───────────────────────────┐
   │  Smoothed Auto-Zoom & Crop  │               │  Per-Speaker Isolated Audio │
   │  (multi-speaker switching)  │               │  (timing-gated masking)     │
   └──────────────┬─────────────┘               └──────────────┬─────────────┘
                 ▼                                             ▼
   ┌───────────────────────────┐               ┌───────────────────────────┐
   │ Output: multi-speaker video │               │ Output: single-speaker cam  │
   │ with original mixed audio   │               │ with isolated audio track   │
   └───────────────────────────┘               └───────────────────────────┘
```

---

## Technical Implementation

### Visual Pipeline
- **Face Detection & Tracking** — MediaPipe face detection with a centroid-distance tracker maintaining stable identity across frames.
- **Facial Landmarks** — MediaPipe Face Mesh extracts per-face landmarks from padded, tracked crops.
- **Mouth Aspect Ratio (MAR)** — a geometric openness signal computed from lip landmarks.
- **Optical Flow** — dense motion estimation (Farneback) on a mouth-centered crop, providing a *motion*-based signal that complements MAR's *geometry*-based one. The two signals fail independently — MAR is precise but landmark-dependent, optical flow is coarser but more resilient to detection dropouts — making their combination more robust than either alone.

### Audio Pipeline
- **Extraction** — direct `ffmpeg` subprocess decoding to a 16kHz mono waveform, chosen for reliability over higher-level libraries.
- **Voice Activity Detection** — RMS energy and zero-crossing rate over sliding windows, with an adaptive noise-floor threshold.
- **Spectral Features** — STFT and mel-spectrogram extraction implemented and validated, providing the standard feature representation used by modern speech models.

### Deep Learning Active Speaker Detection
AVZoom integrates [**TalkNet-ASD**](https://github.com/TaoRuijie/TalkNet-ASD), a model pretrained on the AVA-ActiveSpeaker dataset, as its primary speaker-detection engine. Rather than training a fusion model from scratch, this project focused engineering effort on **production-grade integration**:

- Resolved multiple dependency and API-compatibility issues to run a research-era model reliably on a modern, CPU-only environment.
- Converted the model's outputs to synchronized wall-clock timestamps, correctly handling a frame-rate mismatch between the model's internal video processing and the original source video — ensuring frame-accurate speaker attribution across the entire clip.
- Built a clean adapter layer (`main_talknet.py`) that consumes the model's per-face confidence scores to drive the same smoothed auto-zoom pipeline used by the classical baseline — allowing both approaches to share downstream infrastructure.

### Audio Enhancement — Per-Speaker Isolated Tracks
Rather than blind speech separation (which is best suited to simultaneous overlapping speech), AVZoom takes a more direct approach for turn-taking conversation: since the active-speaker model already knows *when* each person speaks, that timing is used to **gate** the mixed audio per speaker (`audio_sep.py`) — producing a clean, isolated track for each person by preserving the mixed audio during their speaking segments and smoothly attenuating it elsewhere. This directly enables a **single-speaker mode** (`single_speaker.py`): a dedicated camera locked on one chosen speaker for the full clip, paired with that speaker's own isolated audio track.

---

## Results

- Successfully validated deep-learning-based active speaker detection end-to-end on real multi-person video, with the model's confidence scores directly driving visual framing decisions.
- Classical MAR + optical flow baseline cross-validated against independently computed audio voice-activity signals, confirming the visual signals track actual speech activity.
- Per-speaker audio isolation validated by ear on real conversation audio — each generated track cleanly reflects one speaker's turns, enabling a single-speaker cam mode with matched, isolated audio.
- Full pipeline runs entirely on CPU with no GPU dependency, from raw video input to a zoomed, audio-synchronized output file.

---

## Repository Structure

```
main2.py                  Classical pipeline: detection, tracking, MAR + optical
                           flow fusion, smoothed active speaker selection, zoom
main_talknet.py            Deep-learning pipeline: TalkNet-ASD-driven speaker
                           selection with timestamp-synchronized zoom
audio_sep.py                Per-speaker audio isolation via timing-gated masking
single_speaker.py            Single-speaker cam: locked zoom + isolated audio
sep_audio.py                Blind source separation experiment (SepFormer)
Talkcheck.py                 Utility: inspect TalkNet-ASD's raw track/score output

src/video/
├── video_reader.py         Video metadata inspection
├── frame_extractor.py      Frame sampling utility
├── face_detector.py        MediaPipe face detection
├── tracker.py               Centroid-distance face tracking
├── face_mesh.py             Facial landmark extraction
├── mouth.py                  Mouth Aspect Ratio computation
├── optical_flow.py           Motion-based mouth activity signal
├── active_speaker.py         Hysteresis-smoothed speaker selection
└── zoom.py                    Smoothed auto-crop and resize

src/audio/
├── audio_extractor.py      ffmpeg-based waveform extraction
└── features.py               RMS, ZCR, VAD, STFT, mel-spectrogram

data/videos/                Sample input videos
requirements.txt            Python dependencies
```

---

## Installation

```bash
git clone https://github.com/pranay241/AVZoom.git
cd AVZoom
pip install -r requirements.txt
```

**For the deep-learning pipeline**, clone [TalkNet-ASD](https://github.com/TaoRuijie/TalkNet-ASD) as an external dependency and run its inference script to generate speaker-score outputs (see Usage below).

---

## Usage

**Classical pipeline** (no external model dependency):
```bash
python main2.py
```

**Deep-learning pipeline** (TalkNet-ASD driven):
```bash
# 1. Run TalkNet-ASD inference on your video (see TalkNet-ASD docs)
# 2. Copy the resulting tracks.pckl / scores.pckl into TalkNet-ASD/demo/<videoName>/pywork/
python main_talknet.py
```

**Per-speaker isolated audio and single-speaker cam:**
```bash
python audio_sep.py       # generates a gated, isolated audio track per speaker
python single_speaker.py  # generates a locked zoom on one speaker with their isolated audio
```

All variants produce a zoomed, audio-synchronized output video.

---

## Roadmap

- [ ] Automatic overlap detection to flag simultaneous cross-talk segments
- [ ] Unified tracking layer shared between the classical and deep-learning pipelines
- [ ] Broader validation across diverse video sources and conversation styles
- [ ] Real-time / streaming inference mode

---

## References

- Tao, R. et al. — [TalkNet-ASD: Is Someone Speaking?](https://github.com/TaoRuijie/TalkNet-ASD) (AVA-ActiveSpeaker pretrained model)
- Subakan, C. et al. — SepFormer (speechbrain speech separation)
- Google MediaPipe — Face Detection & Face Mesh

## Acknowledgements

Built as part of an independent audio-visual machine learning project, integrating open-source pretrained models with a custom classical computer vision and signal processing pipeline.

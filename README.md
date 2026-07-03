# Audio-Visual Emotion Recognition (Fusion Study)

A compact multimodal AI project: recognize emotion from **audio + video** using
the RAVDESS Speech dataset, comparing unimodal baselines against two fusion
strategies (late fusion vs. cross-attention fusion).

Built to demonstrate: multimodal architecture design, PyTorch model building,
ablation-driven evaluation, and understanding of *why* fusion helps (or doesn't).

---

## Why this project (for Multimodal AI roles)

- Uses **two modalities** (audio + video) with real learned fusion, not just a
  pretrained model call.
- Compares **fusion strategies empirically** (late fusion vs. cross-attention) —
  this is the "theoretical performance analysis of model architectures" story.
- Includes **modality ablations** (audio-only vs. video-only vs. fused) to show
  you understand *where* the performance gain comes from.
- Small enough to train in a couple hours on a free Colab GPU.

---

## 2-Phase Plan

**Phase 1 — Data + Baselines**
1. Download RAVDESS Speech, actors 1–8 (subset, keeps this fast — see below)
2. Run `src/dataset.py` to extract audio mel-spectrograms + sampled video frames
3. Train audio-only baseline (`src/train.py --modality audio`)
4. Train video-only baseline (`src/train.py --modality video`)

**Phase 2 — Fusion + Writeup**
1. Train late-fusion model (`src/train.py --modality fusion --fusion late`)
2. Train cross-attention fusion model (`src/train.py --modality fusion --fusion attention`)
3. Run `src/evaluate.py` to generate the ablation table + confusion matrices
4. Write up results in this README, push to GitHub

---

## Dataset

**RAVDESS Speech (Audio-Visual)** — Zenodo, CC BY-NC-SA 4.0.
Each actor's zip contains full audio-visual `.mp4` clips (modality code `01`
in the filename = full AV).

Download (pick actors 1–8 to stay fast — ~8 x 500MB):
```
https://zenodo.org/record/1188976/files/Video_Speech_Actor_01.zip
https://zenodo.org/record/1188976/files/Video_Speech_Actor_02.zip
... through Actor_08.zip
```
Extract all into `data/raw/`, so you end up with:
```
data/raw/Actor_01/*.mp4
data/raw/Actor_02/*.mp4
...
```

**Filename convention** (e.g. `01-01-06-01-02-01-12.mp4`):
`Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor`

Emotion codes: `01`=neutral `02`=calm `03`=happy `04`=sad `05`=angry
`06`=fearful `07`=disgust `08`=surprised

`dataset.py` parses this automatically and keeps only modality `01` (full AV) files.

> Only have a few hours total? Drop to actors 1–4 and/or 4 emotion classes
> (neutral/happy/sad/angry) in `dataset.py`'s `EMOTIONS` filter — still a
> legitimate fusion comparison, just faster to train.

**Troubleshooting:** RAVDESS's Zenodo page has separate zips for audio-only
(`Audio_Speech_Actors_01-24.zip`), video-only, and full audio-visual
(`Video_Speech_Actor_XX.zip`) — grab the full audio-visual ones, not audio-only.
`dataset.py` searches `data/raw` recursively for `.mp4` files, so it doesn't
matter if your zip extracts flat (`Actor_01/*.mp4`) or nested
(`Video_Speech_Actor_01/Actor_01/*.mp4`).

---

## Setup (Colab)

```bash
!pip install torch torchvision torchaudio librosa moviepy opencv-python-headless scikit-learn pandas -q
```

Then:
```bash
python src/dataset.py --raw_dir data/raw --out_dir data/processed
python src/train.py --data_dir data/processed --modality audio
python src/train.py --data_dir data/processed --modality video
python src/train.py --data_dir data/processed --modality fusion --fusion late
python src/train.py --data_dir data/processed --modality fusion --fusion attention
python src/evaluate.py --data_dir data/processed --checkpoint_dir checkpoints
```

---

## Architecture

- **Audio encoder:** log-mel spectrogram → small CNN (4 conv blocks) → 128-d embedding
- **Video encoder:** 8 sampled frames → pretrained ResNet18 (frozen backbone,
  fine-tuned last block) → temporal mean pool → 128-d embedding
- **Late fusion:** concatenate audio + video embeddings → MLP classifier
- **Cross-attention fusion:** audio embedding attends over video frame
  embeddings (and vice versa) before concatenation → MLP classifier

---

## Results

Trained on RAVDESS Speech, actors 1–8 (480 clips; 360 train / 60 val / 60 test,
speaker-independent split — actor 7 held out for val, actor 8 for test).

| Model              | Accuracy | Macro F1 |
|---------------------|---------:|---------:|
| Audio-only          |    0.367 |    0.250 |
| **Video-only**       | **0.667** | **0.601** |
| Fusion (late)        |    0.633 |    0.570 |
| Fusion (attention)   |    0.533 |    0.415 |

**Takeaway:** Video-only outperformed both fusion strategies. Audio-only was
the weakest modality by a wide margin, and both fusion variants underperformed
video-only — suggesting that with a small training set (~360 clips), the
noisier audio signal degrades the fusion model's ability to learn rather than
adding complementary information. Late fusion held up better than attention
fusion, likely because the cross-attention layer has more parameters to learn
with no extra data to support it — a classic small-data overfitting risk.

This points to two natural next steps: (1) more audio-visual training data,
or (2) a gated fusion mechanism that learns to down-weight the audio modality
per-sample rather than always combining both equally.

See `checkpoints/confusion_matrix.png` for the best model's per-class errors.

---

## Repo structure

```
av-emotion-fusion/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/            # downloaded RAVDESS actor folders
│   └── processed/      # extracted features (created by dataset.py)
├── src/
│   ├── dataset.py       # download-agnostic preprocessing + PyTorch Dataset
│   ├── models.py        # AudioEncoder, VideoEncoder, fusion heads
│   ├── train.py         # training loop, CLI
│   └── evaluate.py       # ablation table + confusion matrix
└── checkpoints/         # saved model weights
```

## Extensions (if you have more time later)
- Add text modality via ASR transcript (Whisper) + DistilBERT for a full
  tri-modal system, matching the JD's audio+text+vision wording exactly.
- Try MELD or CREMA-D for a larger, more naturalistic dataset.
- Add speaker-independent train/test split (leave-N-actors-out) for a more
  rigorous evaluation than random split.
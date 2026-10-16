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
Alternatively, run `python scripts/download_ravdess.py` to fetch and extract all
eight archives automatically (about 4.3 GB of downloads).
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
!pip install torch torchvision torchaudio opencv-python-headless imageio-ffmpeg scikit-learn pandas fastapi uvicorn python-multipart -q
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

## Web demo

After installing the dependencies and training at least one checkpoint, start
the local web app with:

```bash
python run_web.py
```

Open `http://127.0.0.1:8000`, upload an MP4 clip or choose **Live camera** and
allow browser access to the camera and microphone. Live mode sends successive
three-second audio/video windows to the selected checkpoint and updates the
prediction panel after each inference. Camera capture requires a secure browser
context; localhost qualifies. Uploaded MP4 and WebM clips are accepted.
The server loads matching weights from `checkpoints/`, including the audio,
video, tuned fusion, and face-focused checkpoints, and uses the same preprocessing
as training. Face-focused audio/video fusion is the default model; the original
full-frame video and fusion checkpoints remain available as baselines.
The face-landmark model reports how many sampled frames contained a detected
face. Predictions run locally on CPU or CUDA when available. If port 8000 is
already in use, run `python run_web.py 8001` and open port 8001.

### Face geometry experiment

After extracting the RAVDESS audio/video features, extract tracked face meshes:

```bash
python scripts/extract_face_landmarks.py
python scripts/run_gpu.py train --modality fusion --fusion attention --use_landmarks --init_checkpoint checkpoints/fusion_attention.pt --lr 0.0001 --backbone_lr 0.000003 --weight_decay 0.01
python scripts/run_gpu.py evaluate
```

Each clip stores eight normalized 478-point meshes plus 21 geometry features,
including eye openness and gaze, brow spacing and tilt, mouth opening, lip
corner lift, signed forehead-to-eye offset, and face outline. Live camera frames use
the same MediaPipe Face Landmarker pipeline. The model asset is
`assets/face_landmarker.task`.

An additional `fusion_attention_tuned` checkpoint uses label smoothing and
validation macro-AUROC checkpoint selection. On held-out actor 8 it scored
0.600 accuracy and 0.920 macro one-vs-rest AUROC, compared with 0.533 and
0.892 for the original attention fusion checkpoint. The video-only model had
0.583 accuracy and 0.932 AUROC, so video remains a selectable baseline.

Face-focused models use MediaPipe face mesh bounding boxes to crop each sampled
frame before training and inference, reducing room/background influence. Generate
training crops with `python scripts/extract_face_crops.py`; then train with
`--face_frame_dir data/face_frames`. On held-out actor 8, face-focused audio/video
fusion scored 0.733 accuracy, 0.723 macro-F1, and 0.958 macro one-vs-rest AUROC.

### External cross-corpus check

Run `python scripts/evaluate_cremad_external.py` to evaluate a fixed, balanced
50-clip CREMA-D sample from 50 distinct actors. The script reports scores against
both CREMA-D crowd votes and intended filename labels; see
[`reports/crema_d_external/results.md`](reports/crema_d_external/results.md) for
the comparison, limitations, sample manifest, and model predictions. The external
scores are substantially lower than the RAVDESS held-out scores, indicating
significant cross-corpus domain shift.

---

## Architecture

- **Audio encoder:** log-mel spectrogram → small CNN (4 conv blocks) → 128-d embedding
- **Video encoder:** 8 sampled frames → pretrained ResNet18 (frozen backbone,
  fine-tuned last block) → temporal mean pool → 128-d embedding
- **Late fusion:** concatenate audio + video embeddings → MLP classifier
- **Cross-attention fusion:** audio embedding attends over video frame
  embeddings before concatenation → MLP classifier
- **Face geometry:** MediaPipe tracks 478 facial points per frame and derives
  eye, brow, lip, forehead, and face-outline measurements. A learned landmark
  encoder joins the face sequence with video embeddings before audio-video
  cross-attention.

---

## Results

Trained on 480 RAVDESS full audio-video clips from actors 1-8. Actors 1-6
(360 clips) are training data, actor 7 (60 clips) is validation, and actor 8
(60 clips) is held out for the final test. Seeds were fixed at 42. Hyperparameter
selection used validation macro-F1; the attention fusion run was selected with
validation macro-F1 0.601 (head learning rate 1e-4, ResNet block learning rate
3e-6, weight decay 0.01).

| Model | Test accuracy | Test macro-F1 |
|:--|--:|--:|
| Audio-only | 0.450 | 0.324 |
| Video-only | **0.583** | **0.523** |
| Late fusion | 0.517 | 0.451 |
| Attention fusion | 0.533 | 0.484 |
| Attention + face landmarks | 0.600 | 0.530 |

The face-landmark candidate improves aggregate test macro-F1 over the original
attention fusion, but it does not improve every expression: on the 60-clip test
actor it recognized 2/8 surprised clips and predicted disgust 11 times, versus
7/8 surprised and 5 disgust predictions for video-only. It improves fearful
recall (5/8 versus 3/8), while neither model recognizes the four neutral clips.
Face-focused audio/video fusion is the live default, while video-only and the
face-landmark fusion model remain selectable comparisons. These RAVDESS scores
are preliminary because the test set is only one actor. Full metrics are written
to `checkpoints/ablation_results.csv`; the confusion matrix is saved at
`checkpoints/confusion_matrix.png`.

## Repo structure

```
av-emotion-fusion/
  README.md
  requirements.txt
  run_web.py
  web_app.py
  scripts/
    download_ravdess.py
    extract_face_landmarks.py
    run_gpu.py
  web/                  # monochrome HTML, CSS, and JavaScript interface
  src/
    dataset.py          # preprocessing and PyTorch Dataset
    face_geometry.py    # normalized face-mesh geometry features
    face_landmarks.py   # MediaPipe tracking shared by training and inference
    models.py           # audio/video encoders and fusion heads
    train.py            # training loop and CLI
    evaluate.py         # held-out ablation metrics and confusion matrix
  data/                 # downloaded clips and generated features (ignored)
  checkpoints/          # trained model weights and evaluation outputs (ignored)
```
## Extensions (if you have more time later)
- Add text modality via ASR transcript (Whisper) + DistilBERT for a full
  tri-modal system, matching the JD's audio+text+vision wording exactly.
- Try MELD or CREMA-D for a larger, more naturalistic dataset.
- Add speaker-independent train/test split (leave-N-actors-out) for a more
  rigorous evaluation than random split.

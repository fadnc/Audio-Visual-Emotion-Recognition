# Audio-Visual Emotion Recognition

A PyTorch audio-visual emotion classifier trained on RAVDESS. The default model combines a log-mel CNN with a ResNet-18 frame encoder through audio-conditioned attention over face-cropped frames. Optional branches add MediaPipe face landmarks, WavLM speech embeddings, modality dropout, and auxiliary per-modality heads. A FastAPI web app accepts MP4/WebM uploads or live camera recordings.

> **Retraining required.** Input decoding, the shared audio/video window, and mel
> normalization changed. Checkpoints written before this refactor have no JSON
> sidecar, so `affect-eval`, `affect-cv`, the web app, and the CREMA-D script skip
> them. Retrain with the sequence below; new checkpoints always write
> `<name>.json` next to `<name>.pt`.

## Highlights

- **Aligned inputs:** video frames and audio are decoded from the same 3.0 s window with sequential ffmpeg decoding, so WebM uploads and clips that report no frame count work.
- **Global mel normalization:** `stats.json` stores the training-split mean/std; per-clip normalization remains available as `--mel_norm clip`.
- **Speaker-independent evaluation:** RAVDESS actors 7 and 8 are validation/test by default, and `affect-cv` runs six leave-actors-out folds over actors 1–24.
- **Robustness options:** per-clip video/audio augmentation, SpecAugment, modality dropout, and auxiliary audio/video heads are all off by default.
- **Reproducible checkpoints:** sidecars record modality, fusion, face-crop/landmark settings, audio encoder, mel normalization, class order, and git/torch versions.
- **Statistical evaluation:** stratified bootstrap CIs, exact McNemar tests, ECE/MCE, reliability plots, and temperature scaling.
- **Verified packaging:** `affect-*` console commands, pytest suite, and a CPU GitHub Actions workflow.

## Install

Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash; use .venv/bin/activate on Linux/macOS
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[wavlm]"            # Hugging Face transformers for the WavLM encoder
pip install -e ".[dev,wavlm]"        # both
```

For GPU training, install the matching CUDA build of PyTorch first (see pytorch.org), then run the editable install.

## Data

Download and extract RAVDESS speech video archives:

```bash
python scripts/download_ravdess.py --actors 1-8    # ~4.3 GB, default protocol
python scripts/download_ravdess.py --actors 1-24   # ~13 GB, required for 24-actor CV
python scripts/download_ravdess.py --actors 1,3,5  # explicit actor list
```

Clips land under `data/raw/Actor_XX/`. CREMA-D is downloaded automatically by the external evaluation script; no training data is shared between corpora.

## Preprocess

```bash
affect-preprocess --raw_dir data/raw --out_dir data/processed
```

This writes one compressed `.npz` per clip (`frames`, raw `mel`, int16 `waveform`), a manifest with `source_path`, and global mel statistics in `data/processed/stats.json`. The default speaker split is validation actor 7 and test actor 8; override with `--val_actors 7` / `--test_actors 8`. Cross-validation ignores the manifest's `split` column and reassigns actors itself.

Generate the optional MediaPipe inputs from the processed manifest:

```bash
python scripts/extract_face_landmarks.py --data_dir data/processed --out_dir data/landmarks
python scripts/extract_face_crops.py --data_dir data/processed --out_dir data/face_frames
```

Both scripts decode each clip once and use `source_path`, not raw/processed file indices. The landmark script caches completed clips; face crops are overwritten.

## Train

The CLI saves the best-validation checkpoint plus its sidecar under `--checkpoint_dir`. All flags default to the documented behavior; augmentation and robustness features are opt-in.

```bash
# Audio and video baselines
affect-train --modality audio --epochs 20 --patience 5 --selection_metric macro_f1
affect-train --modality video --epochs 20 --patience 5 --selection_metric macro_f1

# Fusion
affect-train --modality fusion --fusion late
affect-train --modality fusion --fusion attention

# Face-focused fusion (recommended main model)
affect-train --modality fusion --fusion attention \
  --face_frame_dir data/face_frames --epochs 25 --patience 6 \
  --lr 1e-4 --backbone_lr 3e-6 --weight_decay 0.01 \
  --label_smoothing 0.05 --selection_metric auroc

# Geometry branch
affect-train --modality fusion --fusion attention \
  --use_landmarks --landmark_dir data/landmarks

# WavLM audio encoder (requires the wavlm extra)
affect-train --modality audio --audio_encoder wavlm \
  --wavlm_model microsoft/wavlm-base --wavlm_freeze_layers 8
```

Robustness options compose with any fusion recipe:

```bash
affect-train --modality fusion --fusion attention \
  --face_frame_dir data/face_frames \
  --aug_flip_p 0.5 --aug_color_jitter 0.2 --aug_gain_db 6 \
  --specaugment --aug_noise_std 0.01 \
  --modality_dropout 0.15 --aux_heads --aux_loss_weight 0.3
```

Augmentation applies to the training split only. Modality dropout and auxiliary heads require `--modality fusion`; landmarks and WavLM are mutually exclusive with video-only training. Checkpoint tags gain suffixes such as `_face`, `_landmarks`, `_wavlm`, `_mdrop`, and `_aux`.

WavLM freezes the feature extractor and the lowest eight transformer layers by default (`--wavlm_freeze_layers`); expect roughly 2 GB of GPU memory at batch size 8, and very slow CPU training.

## Evaluate

```bash
# Ablation table and confusion matrix for every discovered checkpoint
affect-eval --split test

# One checkpoint with confidence intervals and calibration diagnostics
affect-eval --checkpoint checkpoints/fusion_attention_face.pt \
  --bootstrap 1000 --calibration --temperature

# Paired comparison on the same clips
affect-eval --checkpoint checkpoints/fusion_attention_face.pt \
  --compare_checkpoint checkpoints/video_face.pt --bootstrap 1000

# Fast smoke evaluation
affect-eval --max_clips 100
```

Outputs are written under `--checkpoint_dir` (default `checkpoints/`): `ablation_results.csv`, `evaluation_details.json` (bootstrap, calibration, temperature, McNemar), `reliability_diagram.png`, and `confusion_matrix.png`. Statistical options are off by default. `--temperature` fits on the validation split (`--calibration_split val`).

## Cross-validation

The CV runner partitions actors 1–24 into six contiguous, speaker-disjoint folds. For each fold and seed it samples validation actors only from the non-test pool, writes a fold manifest, trains with the normal `affect-train` pipeline, and evaluates the held-out test actors.

```bash
affect-cv --modality fusion --fusion attention \
  --face_frame_dir data/face_frames \
  --n_folds 6 --seeds 0 1 2 --val_actors 2 \
  --epochs 25 --patience 6 --output_dir reports/cv
```

The command above runs 18 jobs. Outputs:

- `reports/cv/fold_XX/seed_Y/manifest.csv` — exact fold membership
- `reports/cv/fold_XX/seed_Y/checkpoints/<tag>.pt` and `.json` — fold checkpoint
- `reports/cv/fold_XX/seed_Y/metrics.json` — test metrics plus training history
- `reports/cv/cv_results.csv` — one row per fold/seed
- `reports/cv/cv_summary.json` / `cv_summary.md` — per-fold and overall mean ± population std

Use `--skip_existing` to reuse completed `metrics.json` files after an interruption. `--actor_range MIN MAX`, `--manifest`, `--data_dir`, `--audio_encoder wavlm`, and all training/robustness flags are inherited. `--checkpoint_dir` is overridden per fold. CV requires enough actors for the requested folds and validation size; download and preprocess actors 1–24 first.

Global mel statistics are computed once during preprocessing and shared by all folds. Test actors therefore contribute a small amount to the mel mean/std if they were part of the preprocessing train split; per-fold statistics are not recomputed.

## Web app

```bash
affect-web --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The app discovers checkpoints recursively from `checkpoints/`, lists only files with a valid sidecar, and rebuilds each model from its sidecar. The face-focused fusion checkpoint is preferred by default when present. Upload an MP4/WebM clip (200 MB limit) or record from the camera; inputs, face cropping, landmarks, WavLM waveforms, and mel statistics follow the selected checkpoint.

If the list is empty, the app reports that no trained checkpoints were found: retrain so sidecars exist, then restart the server.

## External evaluation: CREMA-D

```bash
python scripts/evaluate_cremad_external.py \
  --checkpoint checkpoints/fusion_attention_face.pt \
  --bootstrap 1000 --calibration
```

The script prepares a balanced 50-clip CREMA-D sample (one clip per actor, seed `20261016`), downloads the videos, and reports metrics against both the crowd-sourced multimodal vote and the intended filename label. The model has eight outputs while CREMA-D has six shared classes; predictions to calm or surprise count as incorrect. `--prepare_only` stops after data preparation, `--max_clips N` limits the run, `--full_frame` forces center-cropped frames, and `--compare_checkpoint` adds an exact McNemar test on common clips. The face-crop and full-frame comparators should be separate checkpoints.

## Results

### Post-migration baseline (current code, actors 1–8)

The face-focused fusion checkpoint was retrained with the current pipeline (`global` mel normalization, shared 3-second window) on the default actors 1–8 split, using the recipe in [Reproducing the full retraining sequence](#reproducing-the-full-retraining-sequence). Training early-stopped at epoch 16 with the best validation AUROC at epoch 10 (0.927).

| Evaluation | Model | Accuracy | Macro-F1 | Macro OvR AUROC |
|:--|:--|--:|--:|--:|
| RAVDESS test | Face-focused audio/video fusion | 0.633 | 0.563 | 0.973 |

The test set is 60 clips from a single held-out actor, so this single-split estimate is noisy and not directly comparable to the legacy numbers below; run `affect-cv` for speaker-independent estimates. Only the fusion checkpoint was retrained; the baselines and the CREMA-D external check still refer to the legacy pipeline.

### Legacy results

The table below is retained for context from the **legacy pre-migration pipeline** (per-clip mel normalization and the old decoding path). Those `.pt` files have no sidecars and cannot be evaluated by the current code.

| Evaluation | Model | Accuracy | Macro-F1 | Macro OvR AUROC |
|:--|:--|--:|--:|--:|
| RAVDESS test | Face-focused audio/video fusion (legacy) | 0.733 | 0.723 | 0.958 |
| RAVDESS test | Full-frame video baseline (legacy) | 0.583 | 0.523 | 0.932 |
| CREMA-D vote labels | Face-focused fusion (legacy) | 0.200 | 0.199 | 0.645 |
| CREMA-D vote labels | Full-frame fusion (legacy) | 0.180 | 0.125 | 0.652 |
| CREMA-D intended labels | Face-focused fusion (legacy) | 0.340 | 0.314 | 0.736 |
| CREMA-D intended labels | Full-frame fusion (legacy) | 0.220 | 0.144 | 0.675 |

The legacy external report is in [`reports/crema_d_external/results.md`](reports/crema_d_external/results.md). It used 50 clips from 50 actors and shows substantial cross-corpus domain shift; treat both RAVDESS and CREMA-D numbers as small-sample estimates.

## Reproducing the full retraining sequence

```bash
# 1. Data and features
python scripts/download_ravdess.py --actors 1-24
affect-preprocess --raw_dir data/raw --out_dir data/processed
python scripts/extract_face_landmarks.py --data_dir data/processed --out_dir data/landmarks
python scripts/extract_face_crops.py --data_dir data/processed --out_dir data/face_frames

# 2. Main checkpoints
affect-train --modality fusion --fusion attention --face_frame_dir data/face_frames \
  --epochs 25 --patience 6 --lr 1e-4 --backbone_lr 3e-6 \
  --weight_decay 0.01 --label_smoothing 0.05 --selection_metric auroc
affect-train --modality video --face_frame_dir data/face_frames
affect-train --modality fusion --fusion attention

# 3. Statistical evaluation on the actor-8 test split
affect-eval --split test --bootstrap 1000 --calibration --temperature

# 4. Leave-actors-out CV
affect-cv --modality fusion --fusion attention --face_frame_dir data/face_frames \
  --epochs 25 --patience 6 --n_folds 6 --seeds 0 1 2 --val_actors 2

# 5. External sanity check
python scripts/evaluate_cremad_external.py --bootstrap 1000 --calibration
```

The same sequence is recorded in [`CHANGELOG.md`](CHANGELOG.md).

## Project layout

```
affect_av/
  augment.py        per-sample video/audio augmentation
  checkpoints.py    JSON sidecars, discovery, config-driven loading
  cross_validate.py leave-actors-out folds and aggregate reports
  dataset.py        preprocessing, ffmpeg decoding, dataset
  evaluate.py       ablation table, statistics, plots
  face_geometry.py  landmark feature selection
  face_landmarks.py MediaPipe landmarker and face crops
  metrics.py        bootstrap, McNemar, ECE, temperature scaling
  models.py         CNN/ResNet/WavLM encoders and fusion models
  train.py          reusable training function and CLI
  web_app.py        FastAPI service
scripts/            download, face-asset, CREMA-D, CV helpers
tests/              pytest suite (58 tests on CPU)
web/                local web UI
assets/             MediaPipe face landmarker
reports/            committed evaluation reports
```

## Tests and CI

```bash
pytest
# or
python -m pytest -q
```

The suite covers decoding and normalization, augmentation, checkpoint sidecars, model variants (including a fake-backbone WavLM path), statistics, fold construction, manifest overrides, and module imports. `.github/workflows/ci.yml` runs it on CPU-only Ubuntu with Python 3.10.

## Limitations

- **One-actor RAVDESS test split.** The default test set has 60 clips from a single actor. Prefer `affect-cv` for speaker-independent estimates; the CV means and stds are more stable than a single split.
- **Shared normalization statistics.** Mel mean/std are computed once during preprocessing and reused by every CV fold.
- **Domain shift.** CREMA-D performance is far below RAVDESS performance, and the external sample is only 50 clips.
- **Optional statistics.** Bootstrap, calibration, temperature, and McNemar outputs require explicit flags; default evaluation is unchanged.
- **Feature defaults are conservative.** Augmentation, modality dropout, auxiliary heads, landmarks, and WavLM are opt-in.
- **Legacy checkpoints are unusable.** Files under `checkpoints/` without a `.json` sidecar are skipped by current tools.

## Data and licenses

RAVDESS is available from [Zenodo record 1188976](https://zenodo.org/record/1188976) under its dataset terms. CREMA-D is available from the [official repository](https://github.com/CheyneyComputerScience/CREMA-D); retain its ODbL/DBCL terms and attribution when reusing the data. The MediaPipe face landmarker asset is in `assets/face_landmarker.task`.

Generated data (`data/`) and checkpoints (`checkpoints/`) are ignored by Git. The landmarker asset and evaluation reports are committed.

## Troubleshooting

- **"No checkpoints with sidecars found":** retrain with `affect-train`; a valid checkpoint needs both `<name>.pt` and `<name>.json`.
- **"Mel stats not found":** run `affect-preprocess`, pass `--mel_stats`, or train/evaluate with `--mel_norm clip`.
- **MediaPipe asset missing:** keep `assets/face_landmarker.task` in place; face crops, landmarks, and face-checkpoint evaluation need it.
- **Decode failures:** ensure `imageio-ffmpeg` installed its bundled binary; MP4 and WebM are supported through sequential PPM decoding.
- **Slow CPU training:** the ResNet-18 video branch and WavLM are intended for GPU use. Audio-only CNN runs are the fastest CPU smoke test.

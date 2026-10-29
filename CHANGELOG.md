# Changelog

All notable changes to this project are documented here. The project followed a
phased refactor from a script-based prototype to an installed package with
config-driven checkpoints, statistical evaluation, and speaker-independent
cross-validation.

## [0.1.1] - 2026-10-29

### Added

- Post-migration RAVDESS baseline for the face-focused audio/video fusion
  checkpoint on the default actors 1–8 split: test accuracy 0.633, macro-F1
  0.563, macro OvR AUROC 0.973. Training early-stopped at epoch 16 with the
  best validation AUROC (0.927) at epoch 10. The checkpoint and its sidecar
  were generated locally and are not committed (`checkpoints/` is ignored).

### Changed

- README results now separate the post-migration baseline from the legacy
  pre-migration table, and note that only the fusion checkpoint has been
  retrained.
- `reports/crema_d_external/results.md` is marked as predating the pipeline
  migration.
- Package version bumped to 0.1.1.

### Verification

- Full chain re-run end to end on CPU: `affect-preprocess` (480 clips) →
  `extract_face_crops.py` (480 clips) → `affect-train` → `affect-eval`.
- Web smoke test: `/api/models` lists `fusion_attention_face` and
  `/api/predict` returns the 8-class distribution for a RAVDESS clip.

## [0.1.0] - 2026-10-29

### Added

- Packaged `affect_av` module with `pyproject.toml`, editable install, and
  console commands: `affect-preprocess`, `affect-train`, `affect-eval`,
  `affect-cv`, and `affect-web`.
- Shared 3.0 s audio/video decoding window using sequential ffmpeg PPM frames
  and an `fps` filter, so no frame seeking or frame-count metadata is needed.
- Global training-split log-mel normalization (`stats.json`) as the default;
  legacy per-clip normalization remains available as `--mel_norm clip`.
- Manifest `source_path` as the single source of truth for raw-to-processed
  alignment.
- Per-sample training augmentation: random resized crop, consistent horizontal
  flip across frames, color jitter, rotation, audio gain, Gaussian noise, and
  SpecAugment. All disabled by default.
- Modality dropout with learned missing tokens and optional audio/video
  auxiliary heads for fusion models.
- Optional WavLM waveform encoder (`--audio_encoder wavlm`) with frozen feature
  extractor and lowest transformer layers, weighted hidden-state pooling, and
  an optional `transformers` dependency.
- Checkpoint JSON sidecars describing modality, fusion, audio encoder, face
  cropping, landmark settings, mel normalization/statistics, class order,
  WavLM configuration, git commit, and library versions. Web, evaluation, CV,
  and CREMA-D tools rebuild models from the sidecar.
- Statistical evaluation: stratified bootstrap confidence intervals, exact
  McNemar tests, ECE/MCE, reliability diagrams, and LBFGS temperature scaling.
- Leave-actors-out cross-validation over RAVDESS actors 1–24 (`affect-cv`):
  six contiguous speaker-disjoint folds, seeded validation actors, per-fold
  manifests/checkpoints, CSV results, JSON summary, and a markdown table.
- `scripts/cross_validate.py` wrapper and `scripts/download_ravdess.py
  --actors 1-24` for CV data.
- Face-crop and landmark extraction scripts driven by the processed manifest.
- CPU pytest suite (58 tests) and a GitHub Actions CPU workflow.

### Changed

- Training is callable as `train(args) -> dict`, returning the checkpoint path,
  best epoch/score, epochs run, and per-epoch history.
- `AVEmotionDataset` accepts a manifest override for fold-specific splits and
  can return int16 waveforms scaled to float for WavLM.
- The FastAPI app creates the MediaPipe landmarker once at startup, serializes
  it with a lock, allocates monotonic timestamps, and decodes each upload once.
- Web model discovery is sidecar-based; nested checkpoints sort after preferred
  root-level models, and the face-focused fusion checkpoint is the default when
  present.
- `requirements.txt` now mirrors `pyproject.toml` core dependencies.

### Fixed

- Audio and video were sampled from different windows; both now use the first
  3.0 s of the clip.
- Mel normalization depended on each clip; training/evaluation now share global
  training-split statistics by default.
- OpenCV frame seeking and WebM frame-count handling were unreliable; decoding
  is sequential through ffmpeg.
- Horizontal flip was applied per batch; augmentation is now per sample and
  consistent across all frames of a clip.
- A single-actor RAVDESS test set produced noisy conclusions; CV now reports
  per-fold and aggregate speaker-disjoint metrics.
- Raw and processed files were aligned by index; `source_path` is used instead.
- Scripts relied on `sys.path` hacks; package entry points and absolute imports
  replace them.
- Model configuration was hard-coded in the web app and evaluation scripts;
  sidecars are now the source of truth.
- The web app created a landmarker per request and decoded uploads multiple
  times; it now uses one shared landmarker and one decode.

### Removed

- `run_web.py`, `scripts/run_gpu.py`, and the flat `src/` layout.
- Hard-coded model configuration tables.
- Legacy checkpoint support without sidecars.

### Migration and retraining

Checkpoints produced before this release lack sidecars and were trained with
different decoding/normalization. Current tools skip them. Full retraining
sequence:

```bash
# 1. Data and features (use 1-24 for cross-validation)
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

# 3. Statistical evaluation on the default actor-8 test split
affect-eval --split test --bootstrap 1000 --calibration --temperature

# 4. Leave-actors-out CV (18 runs with the defaults)
affect-cv --modality fusion --fusion attention --face_frame_dir data/face_frames \
  --epochs 25 --patience 6 --n_folds 6 --seeds 0 1 2 --val_actors 2

# 5. External CREMA-D sanity check
python scripts/evaluate_cremad_external.py --bootstrap 1000 --calibration
```

Each training run writes `<tag>.pt` and `<tag>.json`. Keep them together. If
the processed data directory moves, keep `stats.json` with the clip `.npz`
files because sidecars reference its path.

### Verification

- `python -m pytest -q`: 58 passed.
- `python -m compileall -q affect_av scripts tests`: clean.
- Two-fold synthetic CV smoke run verified fold manifests, per-fold checkpoint
  sidecars, test evaluation, `cv_results.csv`, and `cv_summary.md`.
- `affect-cv --skip_existing` verified resume behavior.

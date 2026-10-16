# Audio-Visual Emotion Recognition

An experimental PyTorch model that predicts emotion from speech and facial expression. The face-focused model crops each frame around the detected face before visual analysis. A minimal local web app supports video uploads and live camera capture.

## Run locally

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Download and extract RAVDESS actors 1–8 into `data/raw/` (the downloader fetches about 4.3 GB):

```powershell
python scripts/download_ravdess.py
```

Preprocess, generate face crops, train the face-focused fusion model, and evaluate:

```powershell
python scripts/run_gpu.py preprocess
python scripts/extract_face_crops.py
python scripts/run_gpu.py train --modality fusion --fusion attention --face_frame_dir data/face_frames --epochs 25 --patience 6 --lr 0.0001 --backbone_lr 0.000003 --weight_decay 0.01 --label_smoothing 0.05 --selection_metric auroc
python scripts/run_gpu.py evaluate --split test
```

Start the web app after training:

```powershell
python run_web.py
```

Open `http://127.0.0.1:8000`. Use **Replace clip** or **Remove** to change an upload. The face-focused fusion checkpoint is the default when available. The app also offers baseline models for comparison.

## Model

- **Audio:** log-mel spectrogram encoded by a CNN.
- **Video:** eight sampled frames encoded by a pretrained ResNet-18.
- **Fusion:** audio-conditioned attention over video frame embeddings.
- **Face crop:** MediaPipe landmarks define a padded face crop shared by training and inference.
- **Optional geometry branch:** eye, brow, mouth, lip, forehead, and face-shape features.

## Results

RAVDESS uses 480 clips: 360 train, 60 validation, and 60 test clips, split by actor. The test set contains one held-out actor.

| Evaluation | Model | Accuracy | Macro-F1 | Macro OvR AUROC |
|:--|:--|--:|--:|--:|
| RAVDESS test | Face-focused audio/video fusion | **0.733** | **0.723** | **0.958** |
| RAVDESS test | Full-frame video baseline | 0.583 | 0.523 | 0.932 |
| CREMA-D, crowd-vote labels | Face-focused fusion | 0.200 | 0.199 | 0.645 |
| CREMA-D, crowd-vote labels | Full-frame fusion baseline | 0.180 | 0.125 | 0.652 |
| CREMA-D, intended labels | Face-focused fusion | **0.340** | **0.314** | **0.736** |
| CREMA-D, intended labels | Full-frame fusion baseline | 0.220 | 0.144 | 0.675 |

The external check uses 50 CREMA-D clips from 50 distinct actors. CREMA-D crowd-voted labels can differ from the intended emotion in each filename, so both are reported. The model has eight output classes while CREMA-D covers six; calm/surprise predictions count as incorrect. External performance is much lower than RAVDESS performance, indicating substantial cross-corpus domain shift. These results are preliminary because the external sample is small.

See [the external evaluation report](reports/crema_d_external/results.md) for the protocol, per-class metrics, predictions, and sample manifest. Run `python scripts/evaluate_cremad_external.py` to reproduce the CREMA-D evaluation.

## Data and model asset

RAVDESS is available from [Zenodo](https://zenodo.org/record/1188976) under its dataset terms. CREMA-D is available from the [official repository](https://github.com/CheyneyComputerScience/CREMA-D); retain its ODbL/DBCL terms and attribution when reusing the data. The MediaPipe face landmarker asset used by the app is in `assets/face_landmarker.task`.

Generated data and checkpoints are ignored by Git. The face-landmark model asset and evaluation reports are included in the repository.

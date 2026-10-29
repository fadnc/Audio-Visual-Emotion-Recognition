# External evaluation: CREMA-D

> **Legacy pipeline.** These results predate the 0.1.0 migration (shared-window
> decoding, global training-split mel normalization). The evaluated checkpoint
> had no sidecar and cannot be re-run with the current tools; retrain before
> comparing new numbers against this report.

## Protocol

Evaluated the trained model on 50 CREMA-D audiovisual clips from 50 distinct
actors. The sample was selected with seed `20261016`, balanced across the six
single-label CREMA-D audiovisual majority-vote categories. The model was not
trained or tuned on CREMA-D. All 50 clips decoded successfully.

The face-focused model uses MediaPipe face crops and the original audio stream.
The comparator is the original attention-fusion checkpoint using its full-frame
center-crop visual preprocessing. Metrics are reported against both the
crowd-sourced audiovisual majority vote and the intended emotion in the
filename. These targets differ on this sample, so the two columns answer
different questions. The model has eight outputs; CREMA-D has no calm or
surprise classes, and predictions to those outputs count as incorrect. AUROC is
macro one-vs-rest over the six shared classes using their unnormalized model
probabilities.

## Results

| Model | Target | Accuracy | Balanced accuracy | Macro-F1 | Macro OvR AUROC | Top-2 accuracy |
|---|---|---:|---:|---:|---:|---:|
| Face-focused fusion | CREMA-D multimodal vote | 0.200 | 0.201 | 0.199 | 0.645 | 0.380 |
| Face-focused fusion | Intended filename label | 0.340 | 0.307 | 0.314 | 0.736 | 0.540 |
| Original full-frame fusion | CREMA-D multimodal vote | 0.180 | 0.188 | 0.125 | 0.652 | 0.260 |
| Original full-frame fusion | Intended filename label | 0.220 | 0.194 | 0.144 | 0.675 | 0.320 |

Against intended labels, face-focused fusion improves accuracy by 12 percentage
points and macro AUROC by 6 points over the original full-frame fusion model.
Against crowd votes, accuracy is 2 points higher but macro AUROC is essentially
unchanged (0.645 vs. 0.652). Overall external performance remains low. The
50-clip sample is small, so these are preliminary estimates rather than a
stable benchmark.

For intended labels, the face-focused model's per-class recall was: angry 0.167,
disgust 0.300, fearful 0.333, happy 0.444, neutral 0.000, and sad 0.600. It
missed all six intended-neutral clips. Five predictions were to unsupported
calm/surprise classes.

## Artifacts

- `metrics.json`: face-focused model summary and per-class reports
- `predictions.csv`: filenames, both reference labels, predictions, confidence
- `confusion_matrix.csv`: 8-output confusion matrix against crowd votes
- `full_frame_baseline/metrics.json`: baseline summary and per-class reports
- `full_frame_baseline/predictions.csv`: baseline per-clip results
- `sample_manifest.csv`: exact sample and labels

Re-run with `python scripts/evaluate_cremad_external.py`. Add `--prepare_only`
to download/prepare the sample without evaluating. The comparison command is:

```powershell
python scripts/evaluate_cremad_external.py --checkpoint checkpoints/fusion_attention.pt --full_frame --report_dir reports/crema_d_external/full_frame_baseline
```

CREMA-D provides 7,442 audiovisual clips from 91 actors and crowd-sourced
ratings ([dataset paper](https://pubmed.ncbi.nlm.nih.gov/25653738/)). Its
[official repository](https://github.com/CheyneyComputerScience/CREMA-D)
specifies ODbL 1.0 for the database and DBCL 1.0 for individual contents. Keep
those terms with any redistribution or reuse.

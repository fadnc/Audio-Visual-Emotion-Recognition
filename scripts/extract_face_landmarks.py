"""Extract one normalized 478-point face mesh per sampled video frame.

Everything is driven by the processed manifest's ``source_path`` column; clips
are decoded once through the sequential ffmpeg decoder (no index alignment).
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from affect_av.dataset import N_FRAMES
from affect_av.face_geometry import LANDMARK_FEATURE_DIM
from affect_av.face_landmarks import create_landmarker, extract_face_landmarks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--out_dir", default="data/landmarks")
    parser.add_argument("--n_frames", type=int, default=N_FRAMES)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    processed = pd.read_csv(Path(args.data_dir) / "manifest.csv")
    if "source_path" not in processed.columns:
        raise SystemExit(
            "manifest.csv has no source_path column. Re-run affect-preprocess "
            "with the current version before extracting face landmarks."
        )
    landmarker = create_landmarker()
    clips_with_face = 0
    frame_hits = 0
    frame_total = 0
    try:
        for index, row in enumerate(tqdm(processed.itertuples(index=False),
                                         total=len(processed), desc="face meshes")):
            output_path = os.path.join(args.out_dir, f"{row.clip_id}.npz")
            if os.path.exists(output_path):
                with np.load(output_path) as cached:
                    if (cached["features"].shape == (args.n_frames, LANDMARK_FEATURE_DIM)
                            and "valid_frames" in cached.files):
                        frame_hits += int(cached["valid_frames"].sum())
                        frame_total += len(cached["valid_frames"])
                        clips_with_face += int(cached["valid_frames"].sum() > 0)
                        continue
            features, valid_frames = extract_face_landmarks(
                row.source_path,
                n_frames=args.n_frames,
                landmarker=landmarker,
                timestamp_offset_ms=index * 10_000,
            )
            # Face detection samples are evenly spaced, so retain a per-clip
            # coverage count for quality checks without storing raw images.
            np.savez_compressed(output_path, features=features, valid_frames=valid_frames)
            frame_hits += int(valid_frames.sum())
            frame_total += args.n_frames
            clips_with_face += int(valid_frames.sum() > 0)
    finally:
        landmarker.close()

    print(f"Face detected in {clips_with_face}/{len(processed)} clips")
    print(f"Detected sampled frames: {frame_hits}/{frame_total}")


if __name__ == "__main__":
    main()

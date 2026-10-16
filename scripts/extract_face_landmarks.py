"""Extract one normalized 478-point face mesh per sampled video frame."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import build_manifest  # noqa: E402
from face_geometry import LANDMARK_FEATURE_DIM  # noqa: E402
from face_landmarks import create_landmarker, extract_face_landmarks  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--out_dir", default="data/landmarks")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    processed = pd.read_csv(os.path.join(args.data_dir, "manifest.csv"))
    expected = {row.clip_id: row for row in processed.itertuples(index=False)}
    raw = build_manifest(args.raw_dir)
    landmarker = create_landmarker()
    clips_with_face = 0
    frame_hits = 0
    frame_total = 0
    try:
        for index, row in tqdm(raw.iterrows(), total=len(raw), desc="face meshes"):
            clip_id = f"clip_{index:05d}"
            if clip_id not in expected:
                continue
            meta = expected[clip_id]
            if int(meta.actor) != int(row.actor) or meta.emotion != row.emotion:
                raise ValueError(f"Raw/processed clip order mismatch at {clip_id}")
            output_path = os.path.join(args.out_dir, f"{clip_id}.npz")
            if os.path.exists(output_path):
                with np.load(output_path) as cached:
                    if (cached["features"].shape == (8, LANDMARK_FEATURE_DIM)
                            and "valid_frames" in cached.files):
                        frame_hits += int(cached["valid_frames"].sum())
                        frame_total += len(cached["valid_frames"])
                        clips_with_face += int(cached["valid_frames"].sum() > 0)
                        continue
            features, valid_frames = extract_face_landmarks(
                row.path, landmarker=landmarker, timestamp_offset_ms=int(index) * 10_000
            )
            # Face detection samples are evenly spaced, so retain a per-clip
            # coverage count for quality checks without storing raw images.
            np.savez_compressed(output_path, features=features, valid_frames=valid_frames)
            frame_hits += int(valid_frames.sum())
            frame_total += 8
            clips_with_face += int(valid_frames.sum() > 0)
    finally:
        landmarker.close()

    print(f"Face detected in {clips_with_face}/{len(processed)} clips")
    print(f"Detected sampled frames: {frame_hits}/{frame_total}")


if __name__ == "__main__":
    main()

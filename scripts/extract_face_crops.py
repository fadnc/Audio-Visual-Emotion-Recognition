"""Create face-only frame arrays for every processed clip."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import build_manifest, assign_splits
from src.face_landmarks import create_landmarker, extract_face_cropped_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--out_dir", default="data/face_frames")
    args = parser.parse_args()
    manifest_path = Path(args.data_dir) / "manifest.csv"
    processed = pd.read_csv(manifest_path)
    raw = assign_splits(build_manifest(args.raw_dir))
    raw = raw.reset_index(drop=True)
    if len(raw) < len(processed):
        raise RuntimeError("Raw manifest is shorter than the processed manifest.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    landmarker = create_landmarker()
    try:
        for row in tqdm(processed.itertuples(index=False), total=len(processed),
                        desc="face crops"):
            clip_index = int(str(row.clip_id).split("_")[1])
            source = raw.iloc[clip_index]
            if int(source.actor) != int(row.actor) or source.emotion != row.emotion:
                raise RuntimeError(f"Raw/processed manifest mismatch for {row.clip_id}")
            frames = extract_face_cropped_frames(
                source.path, landmarker=landmarker,
                timestamp_offset_ms=clip_index * 10000,
            )
            np.save(out_dir / f"{row.clip_id}.npy", frames)
    finally:
        landmarker.close()
    print(f"Saved face-only frames for {len(processed)} clips to {out_dir}")


if __name__ == "__main__":
    main()

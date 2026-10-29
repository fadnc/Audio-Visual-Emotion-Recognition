"""Create uint8 face-only frame arrays for every processed clip.

Clips are decoded once through ``affect_av.dataset.decode_clip``. The processed
manifest's ``source_path`` column is the single source of truth, so there is no
raw/processed index alignment.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from affect_av.dataset import N_FRAMES
from affect_av.face_landmarks import create_landmarker, extract_face_cropped_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--out_dir", default="data/face_frames")
    parser.add_argument("--n_frames", type=int, default=N_FRAMES)
    args = parser.parse_args()

    processed = pd.read_csv(Path(args.data_dir) / "manifest.csv")
    if "source_path" not in processed.columns:
        raise SystemExit(
            "manifest.csv has no source_path column. Re-run affect-preprocess "
            "with the current version before extracting face crops."
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    landmarker = create_landmarker()
    try:
        for index, row in enumerate(tqdm(processed.itertuples(index=False),
                                         total=len(processed), desc="face crops")):
            frames = extract_face_cropped_frames(
                row.source_path,
                n_frames=args.n_frames,
                landmarker=landmarker,
                timestamp_offset_ms=index * 10_000,
            )
            np.save(out_dir / f"{row.clip_id}.npy", frames)
    finally:
        landmarker.close()
    print(f"Saved face-only frames for {len(processed)} clips to {out_dir}")


if __name__ == "__main__":
    main()

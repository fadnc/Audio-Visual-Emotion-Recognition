"""Download and extract RAVDESS speech video actor archives.

``--actors`` accepts ranges and lists (``1-8``, ``1,3,5``, ``1-24``). Use
``--actors 1-24`` for leave-actors-out cross-validation.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen
from zipfile import ZipFile
import argparse
import sys


ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = ROOT / "data" / "downloads"
RAW_DIR = ROOT / "data" / "raw"
BASE_URL = "https://zenodo.org/records/1188976/files/Video_Speech_Actor_{actor:02d}.zip?download=1"


def download(actor: int) -> Path:
    archive = DOWNLOAD_DIR / f"Video_Speech_Actor_{actor:02d}.zip"
    if archive.exists() and archive.stat().st_size > 10_000_000:
        print(f"Using existing {archive.name}")
        return archive

    partial = archive.with_suffix(".zip.part")
    url = BASE_URL.format(actor=actor)
    print(f"Downloading actor {actor:02d}: {url}", flush=True)
    try:
        with urlopen(url, timeout=90) as response, partial.open("wb") as output:
            total = int(response.headers.get("Content-Length", 0))
            copied = 0
            last_reported = 0
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
                copied += len(chunk)
                if copied - last_reported >= 64 * 1024 * 1024:
                    last_reported = copied
                    detail = f"{copied / total:.0%}" if total else f"{copied / 1024**2:.0f} MiB"
                    print(f"Actor {actor:02d}: {detail}", flush=True)
        if total and copied != total:
            raise IOError(f"Incomplete download: received {copied} of {total} bytes")
        partial.replace(archive)
        print(f"\nSaved {archive.name} ({archive.stat().st_size / 1024**2:.1f} MiB)")
        return archive
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def extract(archive: Path) -> None:
    with ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (RAW_DIR / member.filename).resolve()
            if not target.is_relative_to(RAW_DIR.resolve()):
                raise ValueError(f"Unsafe path in archive: {member.filename}")
        zipped.extractall(RAW_DIR)
    print(f"Extracted {archive.name} into {RAW_DIR}")


def parse_actors(spec: str) -> list[int]:
    """Parse ``"1-8"``, ``"1,3,5"``, or ``"1-4,10"`` into sorted unique actors."""
    actors: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low_text, high_text = part.split("-", 1)
            low, high = int(low_text), int(high_text)
            if low > high:
                raise ValueError(f"Invalid actor range: {part}")
            actors.update(range(low, high + 1))
        else:
            actors.add(int(part))
    if not actors:
        raise ValueError("No actors selected.")
    if min(actors) < 1 or max(actors) > 24:
        raise ValueError("RAVDESS actors must be in 1..24.")
    return sorted(actors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", default="1-8",
                        help="Actor selection such as '1-8', '1,3,5', or '1-24'.")
    args = parser.parse_args()
    try:
        actors = parse_actors(args.actors)
    except ValueError as exc:
        parser.error(str(exc))

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archives = {}
    with ThreadPoolExecutor(max_workers=min(8, len(actors))) as pool:
        futures = {pool.submit(download, actor): actor for actor in actors}
        for future in as_completed(futures):
            actor = futures[future]
            archives[actor] = future.result()
    for actor in actors:
        extract(archives[actor])
    count = sum(1 for _ in RAW_DIR.rglob("*.mp4"))
    print(f"Ready: {count} MP4 clips under {RAW_DIR}")
    if count == 0:
        sys.exit("No MP4 clips extracted; check the downloaded archive format.")


if __name__ == "__main__":
    main()

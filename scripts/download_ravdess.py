"""Download and extract the eight RAVDESS speech video actor archives."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen
from zipfile import ZipFile
import shutil
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


def main() -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archives = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(download, actor): actor for actor in range(1, 9)}
        for future in as_completed(futures):
            actor = futures[future]
            archives[actor] = future.result()
    for actor in range(1, 9):
        archive = archives[actor]
        extract(archive)
    count = sum(1 for _ in RAW_DIR.rglob("*.mp4"))
    print(f"Ready: {count} MP4 clips under {RAW_DIR}")
    if count == 0:
        sys.exit("No MP4 clips extracted; check the downloaded archive format.")


if __name__ == "__main__":
    main()

"""Run project commands with the system CUDA PyTorch and local app dependencies.

Usage: python scripts/run_gpu.py {preprocess|train|evaluate|web} [arguments]
"""
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# On this workstation the system interpreter has CUDA-enabled PyTorch while
# the project venv has the media/web libraries and a CPU-only PyTorch build.
import torch
import torchvision

venv_site = ROOT / ".venv" / "Lib" / "site-packages"
if venv_site.is_dir():
    sys.path.append(str(venv_site))


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"preprocess", "train", "evaluate", "web"}:
        raise SystemExit(__doc__)
    command, *arguments = sys.argv[1:]
    print(f"PyTorch {torch.__version__} · torchvision {torchvision.__version__} · "
          f"device {'CUDA' if torch.cuda.is_available() else 'CPU'}", flush=True)
    if command == "web":
        import uvicorn

        port = int(arguments[0]) if arguments else 8000
        uvicorn.run("web_app:app", host="127.0.0.1", port=port, reload=False)
        return

    sys.argv = [command, *arguments]
    if command == "preprocess":
        from src.dataset import main as run
    elif command == "train":
        from src.train import main as run
    else:
        from src.evaluate import main as run
    run()


if __name__ == "__main__":
    main()

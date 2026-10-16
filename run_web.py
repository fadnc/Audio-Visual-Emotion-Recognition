"""Start the browser app at http://127.0.0.1:8000."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
venv_site = ROOT / ".venv" / "Lib" / "site-packages"
if venv_site.is_dir():
    sys.path.append(str(venv_site))

import uvicorn


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    uvicorn.run("web_app:app", host="127.0.0.1", port=port, reload=False)

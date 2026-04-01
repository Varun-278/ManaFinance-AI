import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Avoid Windows console encoding crashes from emoji prints in legacy module.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from backend.services import financial_assistant_legacy as legacy


if __name__ == "__main__":
    legacy.demo.launch()

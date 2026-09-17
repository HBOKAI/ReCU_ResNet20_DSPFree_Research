"""Make repository-local packages importable for direct ``python tools/x.py`` runs.

Python imports ``sitecustomize`` during normal startup when it is available on
``sys.path``.  When a script under ``tools/`` is executed directly, that
directory is the script path, so this hook adds the repository root before the
script imports ``recu_hw``.  Existing tools that already insert ROOT manually
remain unchanged; the insertion is idempotent.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

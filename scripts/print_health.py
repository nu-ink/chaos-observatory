import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so package imports work like pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chaos_observatory.health.run_healthcheck import project_root, run_all

rows = run_all(project_root())
print(json.dumps(rows, indent=2, sort_keys=True))

import os
import sys
from pathlib import Path

# add src/ to path so tests can `import rag_pipeline...`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# isolate test state from any project .env
os.environ.setdefault("RAG_DATA_DIR", str(ROOT / "data"))

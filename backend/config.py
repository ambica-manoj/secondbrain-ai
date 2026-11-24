"""Local configuration. Relative data paths are resolved from the repository root."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / '.env', override=False)
DATA_DIR = Path(os.getenv('SECOND_BRAIN_DATA_DIR', str(ROOT / 'data'))).expanduser()
if not DATA_DIR.is_absolute():
    DATA_DIR = ROOT / DATA_DIR
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

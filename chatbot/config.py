"""Central configuration. Secrets and machine-specific paths come from `.env`."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
VECTOR_DB_PATH = ROOT / "vectorbase" / "db_faiss"
DEFAULT_PDFS = [DATA_DIR / "Vietnam-Travel-đã-gộp.pdf"]

# Must match the model used to build the index.
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TEMPERATURE = 0.0

CHUNK_SIZE, CHUNK_OVERLAP = 1000, 200
TOP_K, SCORE_THRESHOLD = 4, 0.5
MAX_HISTORY = 6  # previous messages sent to the LLM

LANGUAGE = "vi-VN"
PIPER_MODEL = os.getenv("PIPER_MODEL")    # path to piper .onnx voice
PIPER_CONFIG = os.getenv("PIPER_CONFIG")  # path to piper .json config

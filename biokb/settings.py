from torch.cuda import is_available
from pathlib import Path

root = Path(__file__).parent.parent
MODEL_CACHE_DIR=str(root/"models")
DB_DIR = root/"db"
DATA_DIR = root/"air/abstracts"

DEVICE_NAME= "cuda" if is_available() else "cpu"
MAX_LENGTH=15000
MAX_TIME=30 #seconds

MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"

EMBEDDING_MODEL_NAME="sentence-transformers/multi-qa-mpnet-base-cos-v1"
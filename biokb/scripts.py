from biokb.air import get_file_names
from pathlib import Path

from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from typing import Tuple
# Read all abstracts
from biokb.settings import MODEL_CACHE_DIR





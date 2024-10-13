from langchain_community.vectorstores.faiss import FAISS
from pathlib import Path
from langchain_core.documents import Document
import logging

path = Path(__file__).parent.parent


def create_documents_from_text_files(file_names):
    documents = []
    for file_name in file_names:
        with open(path/file_name, "r") as f:
            text = f.read()
            documents.append(Document(page_content=text, metadata={"file_name": file_name}))
    return documents


def get_file_names(path: Path, format: str = "txt") -> list[Path]:
    """
    Read all files in a directory and return a list of file paths.
    """
    file_gen = path.rglob(f"*.{format}")
    return [file for file in file_gen if file.is_file()]


def get_logger(
    name: str = "air-rag",
    debug: bool = False
) -> logging.Logger:
    logger = logging.getLogger(name)
    
    # Set the logging level based on the debug flag
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    # Check if the logger already has handlers to avoid duplicate logs
    if not logger.handlers:
        # Create a console handler
        handler = logging.StreamHandler()
        
        # Create a formatter and set it for the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add the handler to the logger
        logger.addHandler(handler)
    
    return logger

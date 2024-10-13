from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.embeddings import Embeddings
from pathlib import Path

from biokb.llm import AIRLLM
from biokb.utils import get_logger
from biokb.prompts import qa_prompt
from biokb.settings import MODEL_CACHE_DIR, DEVICE_NAME, DB_DIR


class AIREmbedding(Embeddings):
    def __init__(
        self, 
        model_name:str,
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, cache_folder=MODEL_CACHE_DIR, device=DEVICE_NAME)

    def embed_documents(
        self, 
        texts: list[str]
    ) -> list[list[float]]:
        """Embed documents."""
        return self.model.encode(texts)
            
    def embed_query(
        self, 
        text: str
    ) -> list[float]:
        """Embed query text."""
        return self.model.encode(text)
    

class AIRPubmedSearch():
    def __init__(
        self, 
        documents: list[Document] = None, 
        embedding_llm: AIREmbedding = None,
    ):
        self.embedding = embedding_llm
        self.documents = documents
        self.index = None

    def build(self, folder_path: Path = DB_DIR):
        self.index = FAISS.from_documents(self.documents, self.embedding)
        self.index.save_local(folder_path=folder_path)

    @classmethod
    def load(
        cls, 
        db_dir: Path, 
        embedding_llm: AIREmbedding, 
    ):
        instance = cls(embedding_llm=embedding_llm)
        instance.index = FAISS.load_local(
            db_dir, 
            embedding_llm, 
            allow_dangerous_deserialization=True
        )
        return instance

    def retrieve(
        self,
        query: str,
        k: int = 5
    ):
        return self.index.similarity_search(query, k)
       
    def search(
        self, 
        llm: AIRLLM,
        query: str, 
        k: int = 4
    ):
        documents = self.retrieve(query, k)
        context = "\n\n".join([_.page_content for _ in documents])
        prompt = qa_prompt.format(query=query, context=context)
        input_ids, attn_mask = llm.tokenize_input(prompt)
        output = llm.generate_text(input_ids, attn_mask)
        response = llm.decode_output(output, len(input_ids[0]))
        
        return response, documents
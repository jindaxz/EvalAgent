from typing import Callable, Optional, Any
from pydantic import BaseModel
from .base import RetrieverType, BaseRetriever
from .faiss import FaissRetriever
from .bm25 import BM25Retriever

class RetrieverManager(BaseModel):
    retriever_type: RetrieverType
    retriever: BaseRetriever
    
    @classmethod
    def create(
        cls,
        retriever_type: RetrieverType,
        embedding_model: Optional[Any] = None,
        device: str = "cpu",
        tokenizer: Optional[Callable] = None
    ):
        if retriever_type == RetrieverType.FAISS:
            if not embedding_model:
                raise ValueError("Embedding model is required for FAISS retriever")
            return cls(
                retriever_type=retriever_type,
                retriever=FaissRetriever(
                    embedding_model=embedding_model,
                    device=device
                )
            )
        elif retriever_type == RetrieverType.BM25:
            return cls(
                retriever_type=retriever_type,
                retriever=BM25Retriever(tokenizer=tokenizer)
            )

    def load_data(self, data: list[str]) -> None:
        self.retriever.load_data(data)

    def retrieve_top_k(self, query: str, k: int) -> list[int]:
        return self.retriever.retrieve_top_k(query, k)
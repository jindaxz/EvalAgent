from enum import Enum
from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict

class RetrieverType(str, Enum):
    FAISS = "faiss"
    BM25 = "bm25"

class BaseRetriever(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @abstractmethod
    def load_data(self, data: list[str]) -> None:
        pass
    
    @abstractmethod
    def retrieve_top_k(self, query: str, k: int) -> list[int]:
        pass
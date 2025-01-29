from typing import Any, List, Optional
import numpy as np
from pydantic import Field
from base import BaseRetriever
import faiss

class FaissRetriever(BaseRetriever):
    embedding_model: Any = Field(..., description="Embedding model instance")
    device: str = Field("cpu", description="Compute device: 'cpu' or 'gpu'")
    index: Optional[faiss.Index] = Field(None, description="FAISS index")
    data: list[str] = Field(default_factory=list, description="Stored documents")

    def __init__(self, **kwargs):
        if faiss is None:
            raise ImportError("faiss package is required for FaissRetriever")
        super().__init__(**kwargs)
        
        if self.device == "gpu" and not faiss.get_num_gpus():
            raise ValueError("GPU requested but not available")

    def load_data(self, data: list[str]) -> None:
        self.data = data
        embeddings = self.embedding_model.encode(data).astype(np.float32)
        
        if self.device == "gpu":
            res = faiss.StandardGpuResources()
            self.index = faiss.GpuIndexFlatIP(res, embeddings.shape[1])
        else:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            
        self.index.add(embeddings)

    def retrieve_top_k(self, query: str, k: int) -> list[int]:
        query_embedding = self.embedding_model.encode([query]).astype(np.float32)
        _, indices = self.index.search(query_embedding, k)
        return indices[0].tolist()
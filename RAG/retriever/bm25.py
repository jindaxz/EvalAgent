

from typing import Optional
import numpy as np
from pydantic import Field
from rank_bm25 import BM25Okapi


class BM25Retriever(BaseRetriever):
    tokenizer: callable = Field(
        lambda x: x.split(),
        description="Tokenizer function for text processing"
    )
    bm25: Optional[BM25Okapi] = Field(None, description="BM25 instance")
    data: List[str] = Field(default_factory=list, description="Stored documents")

    def __init__(self, **kwargs):
        if BM25Okapi is None:
            raise ImportError("rank_bm25 package is required for BM25Retriever")
        super().__init__(**kwargs)

    def load_data(self, data: List[str]) -> None:
        self.data = data
        tokenized_corpus = [self.tokenizer(doc) for doc in data]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve_top_k(self, query: str, k: int) -> List[int]:
        tokenized_query = self.tokenizer(query)
        scores = self.bm25.get_scores(tokenized_query)
        return np.argsort(scores)[-k:][::-1].tolist()
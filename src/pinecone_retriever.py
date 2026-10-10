"""Minimal Pinecone retriever that avoids langchain-pinecone compatibility limits."""
from typing import Any, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field


class DirectPineconeRetriever(BaseRetriever):
    """Query a Pinecone index and return LangChain Documents."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    index: Any = Field(exclude=True)
    embeddings: Any = Field(exclude=True)
    k: int = 3

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        vector = self.embeddings.embed_query(query)
        result = self.index.query(
            vector=vector,
            top_k=self.k,
            include_metadata=True,
        )
        documents = []
        for match in result.matches or []:
            metadata = dict(match.metadata or {})
            # LangChain's PineconeVectorStore commonly stores page text in `text`.
            content = metadata.pop("text", None) or metadata.pop("page_content", None)
            if not content:
                continue
            documents.append(Document(page_content=str(content), metadata=metadata))
        return documents

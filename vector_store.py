"""
Vector Store Module
====================
Manages the ChromaDB vector store for document embeddings.
Handles creation, persistence, similarity search, and document management.
"""

import os
import logging
from typing import List, Optional, Tuple
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages the ChromaDB vector store lifecycle.

    Responsibilities:
    - Initialize embeddings and the Chroma collection
    - Add documents (with deduplication)
    - Perform similarity search with score filtering
    - Persist and reload the store
    """

    def __init__(self):
        self._embeddings = None
        self._vectorstore = None
        self._collection_name = "rag_documents"

    @property
    def embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """Lazy-initialize the Gemini embedding model."""
        if self._embeddings is None:
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=settings.embedding_model,
                google_api_key=settings.google_api_key,
            )
            logger.info(f"Initialized embeddings: {settings.embedding_model}")
        return self._embeddings

    @property
    def vectorstore(self) -> Chroma:
        """Lazy-initialize or load the persisted Chroma vector store."""
        if self._vectorstore is None:
            os.makedirs(settings.chroma_persist_dir, exist_ok=True)
            self._vectorstore = Chroma(
                collection_name=self._collection_name,
                embedding_function=self.embeddings,
                persist_directory=settings.chroma_persist_dir,
            )
            count = self._vectorstore._collection.count()
            logger.info(
                f"Vector store ready ({count} existing documents) "
                f"at '{settings.chroma_persist_dir}'"
            )
        return self._vectorstore

    def add_documents(self, documents: List[Document]) -> int:
        """
        Add document chunks to the vector store.

        Returns:
            Number of documents added.
        """
        if not documents:
            logger.warning("No documents to add.")
            return 0

        self.vectorstore.add_documents(documents)
        total = self.vectorstore._collection.count()
        logger.info(f"Added {len(documents)} chunks. Total in store: {total}")
        return len(documents)

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = None,
        score_threshold: float = None,
    ) -> List[Tuple[Document, float]]:
        """
        Search for documents similar to the query.

        Returns:
            List of (Document, similarity_score) tuples, filtered by threshold.
            Lower scores = more similar in Chroma's distance metric.
        """
        k = k or settings.top_k_results
        score_threshold = score_threshold or settings.similarity_threshold

        results = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=k
        )

        # Filter by relevance score (higher is better for relevance scores)
        filtered = [
            (doc, score) for doc, score in results
            if score >= score_threshold
        ]

        logger.info(
            f"Query: '{query[:80]}...' → "
            f"{len(results)} results, {len(filtered)} above threshold "
            f"({score_threshold})"
        )
        return filtered

    def get_retriever(self, k: int = None):
        """Return a LangChain retriever interface for the vector store."""
        k = k or settings.top_k_results
        return self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k,
                "score_threshold": settings.similarity_threshold,
            },
        )

    def get_document_count(self) -> int:
        """Return the total number of document chunks in the store."""
        return self.vectorstore._collection.count()

    def clear_store(self):
        """Delete all documents from the vector store."""
        self._vectorstore = None
        import shutil
        if os.path.exists(settings.chroma_persist_dir):
            shutil.rmtree(settings.chroma_persist_dir)
        logger.info("Vector store cleared.")

    def get_collection_stats(self) -> dict:
        """Return stats about the current collection."""
        count = self.get_document_count()
        return {
            "total_chunks": count,
            "collection_name": self._collection_name,
            "persist_directory": settings.chroma_persist_dir,
            "embedding_model": settings.embedding_model,
        }


# Singleton instance
vector_store_manager = VectorStoreManager()

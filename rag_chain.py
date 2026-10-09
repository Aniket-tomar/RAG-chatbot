"""
RAG Chain Module
=================
Orchestrates the full Retrieval-Augmented Generation pipeline:
  1. Query reformulation (for better retrieval)
  2. Document retrieval from ChromaDB
  3. Context formatting with source metadata
  4. LLM generation with anti-hallucination prompts
  5. Conversation memory management

This module ties together all components into a cohesive RAG system.
"""

import logging
from typing import List, Dict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from config import settings
from prompts import (
    RAG_PROMPT_TEMPLATE,
    QUERY_REFORMULATION_PROMPT,
    CONTEXTUALIZE_PROMPT,
    format_context,
)
from vector_store import vector_store_manager

logger = logging.getLogger(__name__)


class RAGChain:
    """
    The core RAG pipeline that processes user queries against
    the document collection.

    Features:
    - Query reformulation for improved retrieval
    - Multi-turn conversation support with history
    - Source-attributed, grounded responses
    - Configurable retrieval parameters
    """

    def __init__(self):
        self._llm = None
        self._conversation_histories: Dict[str, List] = {}

    @property
    def llm(self) -> ChatGoogleGenerativeAI:
        """Lazy-initialize the Gemini LLM."""
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model=settings.gemini_model,
                google_api_key=settings.google_api_key,
                temperature=0.1,       # Low temperature for factual accuracy
                top_p=0.85,            # Focused sampling
                max_output_tokens=2048,
            )
            logger.info(f"Initialized LLM: {settings.gemini_model}")
        return self._llm

    def get_chat_history(self, session_id: str) -> List:
        """Retrieve conversation history for a session."""
        return self._conversation_histories.get(session_id, [])

    def _add_to_history(self, session_id: str, question: str, answer: str):
        """Append a Q&A turn to the conversation history."""
        if session_id not in self._conversation_histories:
            self._conversation_histories[session_id] = []

        history = self._conversation_histories[session_id]
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))

        # Keep only last 10 turns (20 messages) to manage context window
        if len(history) > 20:
            self._conversation_histories[session_id] = history[-20:]

    async def _reformulate_query(
        self, question: str, chat_history: List
    ) -> str:
        """
        Strategy: Query Reformulation
        Reformulate the user's question for better retrieval when
        there's conversation history (follow-up questions).
        """
        if not chat_history:
            # No history → just clean up the query
            chain = QUERY_REFORMULATION_PROMPT | self.llm | StrOutputParser()
            reformulated = await chain.ainvoke({"question": question})
            logger.info(f"Reformulated query: '{question}' → '{reformulated}'")
            return reformulated

        # With history → contextualize the follow-up question
        chain = CONTEXTUALIZE_PROMPT | self.llm | StrOutputParser()
        reformulated = await chain.ainvoke({
            "question": question,
            "chat_history": chat_history,
        })
        logger.info(f"Contextualized query: '{question}' → '{reformulated}'")
        return reformulated

    async def query(
        self,
        question: str,
        session_id: str = "default",
        use_reformulation: bool = True,
    ) -> Dict:
        """
        Execute the full RAG pipeline for a user question.

        Args:
            question: The user's question.
            session_id: Session identifier for conversation history.
            use_reformulation: Whether to reformulate the query.

        Returns:
            Dict with 'answer', 'sources', and 'context_used' keys.
        """
        chat_history = self.get_chat_history(session_id)

        # Step 1: Query reformulation
        search_query = question
        if use_reformulation:
            try:
                search_query = await self._reformulate_query(
                    question, chat_history
                )
            except Exception as e:
                logger.warning(f"Query reformulation failed: {e}")
                search_query = question

        # Step 2: Retrieve relevant documents
        results = vector_store_manager.similarity_search_with_scores(
            search_query,
            k=settings.top_k_results,
        )

        # Extract documents and scores
        retrieved_docs = [doc for doc, score in results]
        scores = [score for _, score in results]

        # Step 3: Format context with source metadata
        context = format_context(retrieved_docs)

        # Step 4: Generate answer using the RAG prompt
        chain = RAG_PROMPT_TEMPLATE | self.llm | StrOutputParser()

        answer = await chain.ainvoke({
            "context": context,
            "question": question,
            "chat_history": chat_history,
        })

        # Step 5: Update conversation history
        self._add_to_history(session_id, question, answer)

        # Step 6: Compile source information
        sources = []
        for doc, score in results:
            sources.append({
                "filename": doc.metadata.get("filename", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "relevance_score": round(score, 4),
                "snippet": doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200 else doc.page_content,
            })

        return {
            "answer": answer,
            "sources": sources,
            "search_query": search_query,
            "num_chunks_retrieved": len(retrieved_docs),
        }

    def clear_history(self, session_id: str = "default"):
        """Clear conversation history for a session."""
        if session_id in self._conversation_histories:
            del self._conversation_histories[session_id]
            logger.info(f"Cleared history for session: {session_id}")

    def clear_all_histories(self):
        """Clear all conversation histories."""
        self._conversation_histories.clear()
        logger.info("Cleared all conversation histories.")


# Singleton instance
rag_chain = RAGChain()

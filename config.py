"""
RAG Chatbot Configuration Module
=================================
Centralizes all configuration with validation and sensible defaults.
"""

import os
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    google_api_key: str = Field(
        default="",
        description="Google Gemini API key"
    )

    # Model Configuration
    gemini_model: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model to use for generation"
    )
    embedding_model: str = Field(
        default="models/embedding-001",
        description="Gemini model to use for embeddings"
    )

    # RAG Configuration
    chunk_size: int = Field(
        default=1000,
        description="Size of text chunks for splitting documents"
    )
    chunk_overlap: int = Field(
        default=200,
        description="Overlap between consecutive chunks"
    )
    top_k_results: int = Field(
        default=5,
        description="Number of top similar chunks to retrieve"
    )
    similarity_threshold: float = Field(
        default=0.3,
        description="Minimum similarity score for retrieved chunks"
    )

    # Paths
    chroma_persist_dir: str = Field(
        default="./chroma_db",
        description="Directory to persist ChromaDB data"
    )
    upload_dir: str = Field(
        default="./uploads",
        description="Directory for uploaded documents"
    )

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

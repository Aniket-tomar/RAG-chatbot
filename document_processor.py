"""
Document Processor Module
==========================
Handles loading, parsing, and chunking documents of multiple formats
(PDF, TXT, DOCX, CSV) for ingestion into the vector store.
"""

import os
import logging
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
)
from config import settings

logger = logging.getLogger(__name__)

# Mapping of file extensions to their LangChain document loaders
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".docx": Docx2txtLoader,
}

SUPPORTED_EXTENSIONS = set(LOADER_MAP.keys())


def get_loader(file_path: str):
    """Return the appropriate document loader based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    loader_class = LOADER_MAP.get(ext)
    if loader_class is None:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return loader_class(file_path)


def load_document(file_path: str) -> List[Document]:
    """
    Load a single document and return a list of LangChain Document objects.

    Each Document contains the text content and metadata (source path,
    page number for PDFs, etc.).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document not found: {file_path}")

    logger.info(f"Loading document: {file_path}")
    loader = get_loader(file_path)
    documents = loader.load()

    # Enrich metadata with the original filename
    filename = os.path.basename(file_path)
    for doc in documents:
        doc.metadata["filename"] = filename
        doc.metadata["source"] = file_path

    logger.info(f"Loaded {len(documents)} page(s)/section(s) from '{filename}'")
    return documents


def load_documents_from_directory(directory: str) -> List[Document]:
    """
    Recursively load all supported documents from a directory.
    """
    all_documents = []
    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                file_path = os.path.join(root, file)
                try:
                    docs = load_document(file_path)
                    all_documents.extend(docs)
                except Exception as e:
                    logger.error(f"Failed to load '{file_path}': {e}")
    return all_documents


def split_documents(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.

    Uses RecursiveCharacterTextSplitter which tries to split on
    natural boundaries (paragraphs, sentences, words) before falling
    back to character-level splitting.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)
    logger.info(
        f"Split {len(documents)} document(s) into {len(chunks)} chunks "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


def process_uploaded_file(file_path: str) -> List[Document]:
    """
    Full pipeline: load a file and split it into chunks ready for embedding.
    """
    documents = load_document(file_path)
    chunks = split_documents(documents)
    return chunks

"""
rag/build_index.py — Build and persist the Chroma vector store for MDT docs.

Indexes:
  - All .txt files in rag/docs_source/
  - Any .pdf files in rag/docs_source/ (e.g., internship MDT walkthrough PDFs)
  - Any .c or .h files in rag/docs_source/ (C source snippets)

Uses Cohere embed-v4.0 embeddings (COHERE_API_KEY from .env) to create dense
vector representations of each document chunk. Chunks are 500 chars with 50-char
overlap (per Section 7.3 of PROJECT_GUIDE.md).

The vector store is persisted to data/chroma/ and can be loaded by the
retrieval tool without re-embedding.

Usage:
    python -m rag.build_index                      # index all docs in docs_source/
    python -m rag.build_index --force              # re-index from scratch
    python -m rag.build_index --docs-dir my/path  # custom docs directory
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get_embeddings():
    """Return a CohereEmbeddings instance using COHERE_API_KEY from .env."""
    from langchain_cohere import CohereEmbeddings
    return CohereEmbeddings(
        model="embed-v4.0",
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )


def get_docs_source_dir() -> Path:
    """Return the default docs_source directory (sibling of this file)."""
    return Path(__file__).parent / "docs_source"


def build_vectorstore(
    doc_paths: list[str] | None = None,
    docs_dir: str | None = None,
    persist_dir: str = "data/chroma",
    force: bool = False,
) -> object:
    """Build (or rebuild) the Chroma vector store from document files.

    Args:
        doc_paths:   Explicit list of file paths to index. If None, uses docs_dir.
        docs_dir:    Directory to scan for .txt, .pdf, .c, .h files.
                     Defaults to rag/docs_source/.
        persist_dir: Directory to persist the Chroma store to.
        force:       If True, delete and rebuild even if persist_dir exists.

    Returns:
        The Chroma vectorstore object (ready for similarity search).
    """
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma

    embedding_fn = _get_embeddings()
    persist_path = Path(persist_dir)

    # If already indexed and not forcing, just load
    if persist_path.exists() and not force:
        print(f"\U0001f4da  Loading existing vector store from: {persist_dir}")
        vectorstore = Chroma(
            persist_directory=str(persist_path),
            embedding_function=embedding_fn,
        )
        count = vectorstore._collection.count()
        print(f"    {count} chunks loaded.")
        return vectorstore

    # Determine files to index
    if doc_paths is None:
        source_dir = Path(docs_dir) if docs_dir else get_docs_source_dir()
        doc_paths = []
        for ext in (".txt", ".pdf", ".c", ".h", ".md"):
            doc_paths.extend(str(p) for p in source_dir.glob(f"**/*{ext}"))

    if not doc_paths:
        raise ValueError(
            "No documents found to index. "
            "Put .txt, .pdf, or .c files in rag/docs_source/ or pass doc_paths."
        )

    print(f"\U0001f4c4  Indexing {len(doc_paths)} document(s)...")

    # Load documents
    docs = []
    for path in doc_paths:
        path_lower = path.lower()
        try:
            if path_lower.endswith(".pdf"):
                loader = PyPDFLoader(path)
            else:
                loader = TextLoader(path, encoding="utf-8")
            loaded = loader.load()
            docs.extend(loaded)
            print(f"    \u2713  {Path(path).name}  ({len(loaded)} page(s)/chunk(s))")
        except Exception as e:
            print(f"    \u26a0\ufe0f  Skipped {path}: {e}")

    if not docs:
        raise ValueError("No documents could be loaded. Check file paths and formats.")

    # Chunk documents (500 chars / 50 overlap per Section 7.3)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,  # character-based
    )
    chunks = splitter.split_documents(docs)
    print(f"\n\u2702\ufe0f   Split into {len(chunks)} chunks")

    # Build and persist vector store
    if force and persist_path.exists():
        shutil.rmtree(persist_path)
        print(f"\U0001f5d1\ufe0f   Cleared existing store at {persist_dir}")

    persist_path.mkdir(parents=True, exist_ok=True)
    print("\U0001f522  Embedding chunks with Cohere embed-v4.0...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=str(persist_path),
    )

    count = vectorstore._collection.count()
    print(f"\u2705  Vector store built: {count} chunks in {persist_dir}")
    return vectorstore


def load_vectorstore(persist_dir: str = "data/chroma") -> object:
    """Load a previously built vector store (no re-embedding)."""
    from langchain_chroma import Chroma

    return Chroma(
        persist_directory=persist_dir,
        embedding_function=_get_embeddings(),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="MDT AI Copilot — build RAG vector index")
    parser.add_argument("--docs-dir", default=None,
                        help="Directory with docs to index (default: rag/docs_source/)")
    parser.add_argument("--persist-dir", default="data/chroma",
                        help="Where to store the Chroma index")
    parser.add_argument("--force", action="store_true",
                        help="Re-index from scratch even if index already exists")
    args = parser.parse_args()

    build_vectorstore(
        docs_dir=args.docs_dir,
        persist_dir=args.persist_dir,
        force=args.force,
    )


if __name__ == "__main__":
    _cli()

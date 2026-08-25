"""
Step 1: Document Ingestion Pipeline for RAG Chatbot
-----------------------------------------------------
Loads PDF / DOCX / TXT files -> chunks text -> embeds with an open-source
sentence-transformers model -> stores vectors + metadata in a local FAISS index.

Usage:
    1. Put your PDF/DOCX/TXT files inside ./documents
    2. Run: python ingest.py
    3. This creates ./faiss_index/index.faiss and ./faiss_index/metadata.pkl
       which the retrieval/chat step will load later.
"""

import os

# Must be set before faiss/torch are imported. On Windows, faiss and torch each
# bundle their own OpenMP/MKL runtime DLLs; loading both in one process can
# cause a DLL initialization failure. This tells the runtime to tolerate it.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer  # import before faiss on Windows
import faiss

from pypdf import PdfReader
import docx  # from python-docx package

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# If HuggingFace Hub downloads are blocked in your environment, download the
# model once elsewhere (see chat instructions) and change this to a local
# path, e.g. "./models/all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 500      # characters per chunk
CHUNK_OVERLAP = 50    # overlap between consecutive chunks
DOCS_DIR = "documents"
INDEX_DIR = "faiss_index"


# ---------------------------------------------------------------------------
# LOADERS - one function per file type
# ---------------------------------------------------------------------------
def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text


def load_docx(path: str) -> str:
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs)


def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_txt,
}


def load_document(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported file type: {ext}")
    return LOADERS[ext](path)


# ---------------------------------------------------------------------------
# CHUNKING - simple fixed-size sliding window (swap for a smarter splitter later)
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = " ".join(text.split())  # normalize whitespace
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# INGESTION PIPELINE
# ---------------------------------------------------------------------------
class DocumentIngestionPipeline:
    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL_NAME):
        print(f"Loading embedding model: {embedding_model_name} ...")
        self.model = SentenceTransformer(embedding_model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        # IndexFlatIP + normalized embeddings == cosine similarity search
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata: List[Dict] = []  # parallel record for each vector in the index

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            chunks,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return embeddings.astype("float32")

    def add_document(self, filepath: str):
        print(f"Processing: {filepath}")
        text = load_document(filepath)
        chunks = chunk_text(text)
        if not chunks:
            print(f"  No text extracted from {filepath}, skipping.")
            return
        embeddings = self.embed_chunks(chunks)
        self.index.add(embeddings)
        for i, chunk in enumerate(chunks):
            self.metadata.append({
                "source": os.path.basename(filepath),
                "chunk_id": i,
                "text": chunk,
            })
        print(f"  Added {len(chunks)} chunks.")

    def ingest_directory(self, directory: str):
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"'{directory}' does not exist. Create it and add your files.")
        files = [f for f in directory.rglob("*") if f.suffix.lower() in LOADERS]
        print(f"Found {len(files)} supported files in {directory}")
        for f in files:
            self.add_document(str(f))

    def save(self, index_dir: str = INDEX_DIR):
        os.makedirs(index_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(index_dir, "index.faiss"))
        with open(os.path.join(index_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Saved FAISS index + metadata to '{index_dir}/' ({self.index.ntotal} vectors)")

    @classmethod
    def load(cls, index_dir: str = INDEX_DIR, embedding_model_name: str = EMBEDDING_MODEL_NAME):
        """Reload a previously saved index (used later in the retrieval step)."""
        pipeline = cls(embedding_model_name)
        pipeline.index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        with open(os.path.join(index_dir, "metadata.pkl"), "rb") as f:
            pipeline.metadata = pickle.load(f)
        return pipeline


if __name__ == "__main__":
    pipeline = DocumentIngestionPipeline()
    pipeline.ingest_directory(DOCS_DIR)
    pipeline.save()
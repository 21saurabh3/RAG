"""
Step 2: Chat API
-----------------
POST /chat
Retrieves relevant chunks from the FAISS index built in ingest.py, classifies
intent, and generates a grounded answer using a small open-source LLM that runs
in-process via the `transformers` library (no separate app/install needed).

Prerequisites:
    1. Run ingest.py first to build faiss_index/
    2. First run of this script will auto-download google/flan-t5-small
       (~300MB) from HuggingFace Hub and cache it locally.

Run the API:
    pip install -r requirements.txt
    python chat_api.py

Test it:
    curl -X POST http://localhost:8000/chat ^
      -H "Content-Type: application/json" ^
      -d "{\"session_id\": \"session_001\", \"message\": \"What is your return policy?\"}"
"""

import os

# Must be set before faiss/torch are imported (see ingest.py for details).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pickle
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer  # import before faiss on Windows
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

# Reuse the same load/chunk logic used in step 1 so both endpoints stay in sync.
from ingest import load_document, chunk_text, LOADERS, DOCS_DIR

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # must match ingest.py
INDEX_DIR = "faiss_index"
TOP_K = 3

# Small open-source generation model (~300MB), runs in-process via transformers.
# No separate app to install - downloads automatically on first run, like the
# embedding model did.
GENERATION_MODEL_NAME = "google/flan-t5-small"
MAX_NEW_TOKENS = 200

# Minimum retrieval similarity score (cosine, since embeddings are normalized)
# required before we trust the top match enough to generate an answer from it.
# Below this, the question is treated as out-of-knowledge rather than risking
# a hallucinated answer from a small generation model. Tune this against your
# own documents/queries - start here and adjust based on false positives/negatives.
SIMILARITY_THRESHOLD = 0.35

OUT_OF_KNOWLEDGE_RESPONSE = (
    "I could not find this information in the available policy document. "
    "Please contact customer support for confirmation."
)

import re

# ---------------------------------------------------------------------------
# INTENT CLASSIFICATION - simple keyword/rule-based
# Extend this dict with your own categories/keywords as needed.
# Order matters: dict is checked top-to-bottom, so more specific phrases
# should come before broader ones if keywords could overlap.
# ---------------------------------------------------------------------------
INTENT_KEYWORDS = {
    "order_status": ["order status", "refund status", "track", "shipment",
                      "delivery status", "where is my", "check my order",
                      "check my refund", "refund"],
    "policy_query": ["return policy", "policy", "exchange", "warranty", "cancel"],
    "product_info": ["product", "specification", "feature", "price", "available", "stock"],
    "complaint": ["complaint", "issue", "problem", "not working", "broken", "unhappy"],
    "greeting": ["hi", "hello", "hey", "good morning", "good evening"],
}


def classify_intent(message: str) -> str:
    text = message.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return intent
    return "general_query"


# ---------------------------------------------------------------------------
# ORDER LOOKUP - slot filling for order/refund-status queries
# ---------------------------------------------------------------------------
# Matches things like ORD12345, AB1234, etc. Adjust to your real order ID format.
ORDER_ID_PATTERN = re.compile(r"\b[A-Za-z]{2,5}\d{3,}\b")


def extract_order_id(message: str) -> Optional[str]:
    match = ORDER_ID_PATTERN.search(message)
    return match.group(0).upper() if match else None


def get_refund_status(order_id: str) -> str:
    """
    Mocked backend lookup. Replace this with a real call to your order/refund
    system (DB query, internal API, etc.) when one is available.
    """
    return "being processed"


# ---------------------------------------------------------------------------
# COMPLAINT REGISTRATION - slot filling for complaint queries
# ---------------------------------------------------------------------------
_complaint_counter = {"next_id": 1001}


def register_complaint(description: str) -> str:
    """
    Mocked complaint registration. Replace with a real call to your ticketing/
    CRM system when available. Returns a generated complaint ID.
    """
    complaint_id = f"CMP{_complaint_counter['next_id']}"
    _complaint_counter["next_id"] += 1
    return complaint_id


# ---------------------------------------------------------------------------
# RETRIEVER - loads the FAISS index built by ingest.py
# ---------------------------------------------------------------------------
class Retriever:
    def __init__(self, index_dir: str = INDEX_DIR, embedding_model_name: str = EMBEDDING_MODEL_NAME):
        print(f"Loading embedding model: {embedding_model_name} ...")
        self.model = SentenceTransformer(embedding_model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

        index_path = os.path.join(index_dir, "index.faiss")
        meta_path = os.path.join(index_dir, "metadata.pkl")

        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                self.metadata: List[Dict] = pickle.load(f)
            print(f"Loaded FAISS index with {self.index.ntotal} vectors.")
        else:
            # No prebuilt index - start empty. This lets the server come up
            # with zero documents; POST /upload will populate it from there.
            print("No existing index found - starting empty. "
                  "Upload documents via POST /upload to populate it.")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata: List[Dict] = []

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        query_vec = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            record = self.metadata[idx]
            results.append({
                "document_name": record["source"],
                "chunk_id": f"chunk_{record['chunk_id'] + 1:03d}",  # e.g. chunk_001
                "text": record["text"],
                "score": float(score),
            })
        return results

    def add_chunks(self, source_name: str, chunks: List[str]) -> int:
        """Embed new chunks and add them to the live index + metadata."""
        embeddings = self.model.encode(
            chunks, convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        self.index.add(embeddings)
        for i, chunk in enumerate(chunks):
            self.metadata.append({
                "source": source_name,
                "chunk_id": i,
                "text": chunk,
            })
        return len(chunks)

    def save(self, index_dir: str = INDEX_DIR):
        """Persist the current index + metadata to disk so uploads survive restarts."""
        os.makedirs(index_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(index_dir, "index.faiss"))
        with open(os.path.join(index_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)


# ---------------------------------------------------------------------------
# GENERATION - small open-source model, runs in-process via transformers
# ---------------------------------------------------------------------------
def build_prompt(message: str, history: List[Dict], context_chunks: List[Dict]) -> str:
    context_text = "\n\n".join(
        f"[Source: {c['document_name']}]\n{c['text']}" for c in context_chunks
    )
    history_text = ""
    for turn in history[-4:]:  # last 4 turns of context, keeps prompt short
        history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

    return f"""Answer the question using ONLY the context below. If the answer isn't
in the context,respond with: "Sorry, I could not find this information in the available
policy document. Please contact customer support for confirmation." Be concise and factual.

Context:
{context_text}

Conversation so far:
{history_text}
Question: {message}

Answer:"""


class Generator:
    def __init__(self, model_name: str = GENERATION_MODEL_NAME):
        print(f"Loading generation model: {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        output_ids = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# SESSION STORE - in-memory (resets when the server restarts)
# Each session holds: conversation history + a "pending_action" marker for
# whatever slot-filling flow is mid-way through (or None if not mid-flow).
# ---------------------------------------------------------------------------
SESSIONS: Dict[str, Dict] = {}


def get_session(session_id: str) -> Dict:
    return SESSIONS.setdefault(session_id, {"history": [], "pending_action": None})


# ---------------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------------
retriever: Optional[Retriever] = None
generator: Optional[Generator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, generator
    retriever = Retriever()
    generator = Generator()
    yield


app = FastAPI(title="RAG Chat API", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str
    message: str


class Source(BaseModel):
    document_name: str
    chunk_id: str


class ChatResponse(BaseModel):
    session_id: str
    intent: str
    response: str
    sources: List[Source]


class UploadResponse(BaseModel):
    document_name: str
    chunks_added: int


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in LOADERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported types: {list(LOADERS.keys())}",
        )

    os.makedirs(DOCS_DIR, exist_ok=True)
    save_path = os.path.join(DOCS_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    text = load_document(save_path)
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No extractable text found in the uploaded file.",
        )

    chunks_added = retriever.add_chunks(file.filename, chunks)
    retriever.save()  # persist so this survives a server restart

    return UploadResponse(document_name=file.filename, chunks_added=chunks_added)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    session = get_session(req.session_id)
    history = session["history"]
    intent = classify_intent(req.message)
    sources: List[Dict] = []
    pending = session["pending_action"]

    # --- Continue a slot-filling flow already in progress --------------------
    if pending == "order_id":
        order_id = extract_order_id(req.message) or req.message.strip()
        status = get_refund_status(order_id)
        answer = f"The refund status for order {order_id} is {status}."
        session["pending_action"] = None
        intent = "order_status"

    elif pending == "complaint_description":
        complaint_id = register_complaint(req.message.strip())
        answer = f"Your complaint has been registered with complaint ID {complaint_id}."
        session["pending_action"] = None
        intent = "complaint"

    # --- Start a new slot-filling flow ---------------------------------------
    elif intent == "order_status":
        order_id = extract_order_id(req.message)
        if order_id:
            status = get_refund_status(order_id)
            answer = f"The refund status for order {order_id} is {status}."
        else:
            answer = "Please provide your order ID."
            session["pending_action"] = "order_id"

    elif intent == "complaint":
        answer = "Please describe your issue."
        session["pending_action"] = "complaint_description"

    # --- Otherwise: normal document-grounded RAG flow ------------------------
    else:
        context_chunks = retriever.search(req.message, top_k=TOP_K)

        # If we found nothing, or the best match isn't actually relevant
        # (score below threshold), don't let the generation model guess -
        # return a fixed, honest fallback instead.
        if not context_chunks or context_chunks[0]["score"] < SIMILARITY_THRESHOLD:
            answer = OUT_OF_KNOWLEDGE_RESPONSE
            history.append({"user": req.message, "assistant": answer, "rag": False})
            return ChatResponse(
                session_id=req.session_id, intent=intent, response=answer, sources=[]
            )

        # Only feed genuine document Q&A turns back into the prompt, not
        # slot-filling exchanges (order IDs, complaint IDs, etc.) - mixing
        # those in confuses a small generation model like flan-t5-small.
        rag_history = [h for h in history if h.get("rag")]
        prompt = build_prompt(req.message, rag_history, context_chunks)
        answer = generator.generate(prompt)
        sources = [
            {"document_name": c["document_name"], "chunk_id": c["chunk_id"]}
            for c in context_chunks
        ]
        history.append({"user": req.message, "assistant": answer, "rag": True})
        return ChatResponse(
            session_id=req.session_id, intent=intent, response=answer, sources=sources
        )

    history.append({"user": req.message, "assistant": answer, "rag": False})

    return ChatResponse(
        session_id=req.session_id,
        intent=intent,
        response=answer,
        sources=sources,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
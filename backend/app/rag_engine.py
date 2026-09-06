import os
from typing import List, Dict

from dotenv import load_dotenv
from google import genai
from google.genai import types
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# Load Environment Variables
# ==========================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not configured in .env"
    )


# ==========================================
# Gemini Client
# ==========================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ==========================================
# Embedding Model
# ==========================================

EMBEDDING_MODEL = "gemini-embedding-001"


# ==========================================
# 1. CV Text Chunking
# ==========================================

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> List[str]:
    """
    Split CV text into overlapping chunks.

    Example:

    Chunk 1 -> characters 0-500
    Chunk 2 -> characters 400-900
    Chunk 3 -> characters 800-1300

    Overlapping chunks help preserve context.
    """

    text = text.strip()

    if not text:
        return []

    chunks = []

    start = 0

    step = chunk_size - overlap

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


# ==========================================
# 2. Create Gemini Embedding
# ==========================================

def create_embedding(
    text: str,
    task_type: str = "RETRIEVAL_DOCUMENT"
) -> List[float]:
    """
    Convert text into a numerical embedding vector
    using Gemini.

    RETRIEVAL_DOCUMENT:
        Used when embedding CV/document chunks.

    RETRIEVAL_QUERY:
        Used when embedding a search query.
    """

    if not text.strip():
        return []

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type
        ),
    )

    return result.embeddings[0].values


# ==========================================
# 3. Create Embeddings for CV Chunks
# ==========================================

def embed_chunks(
    chunks: List[str]
) -> List[List[float]]:
    """
    Create an embedding vector for every CV chunk.
    """

    embeddings = []

    for chunk in chunks:

        embedding = create_embedding(
            chunk,
            task_type="RETRIEVAL_DOCUMENT"
        )

        embeddings.append(embedding)

    return embeddings


# ==========================================
# 4. Retrieve Relevant CV Chunks
# ==========================================

def retrieve_relevant_chunks(
    query: str,
    chunks: List[str],
    embeddings: List[List[float]],
    top_k: int = 3
) -> List[Dict]:
    """
    Retrieve the most relevant CV chunks
    for a given query using cosine similarity.
    """

    if not query.strip():
        return []

    if not chunks or not embeddings:
        return []

    # Create embedding for the user's query
    query_embedding = create_embedding(
        query,
        task_type="RETRIEVAL_QUERY"
    )

    if not query_embedding:
        return []

    # Calculate similarity between query and CV chunks
    similarities = cosine_similarity(
        [query_embedding],
        embeddings
    )[0]

    # Rank chunks from highest similarity to lowest
    ranked_indexes = similarities.argsort()[::-1]

    results = []

    # Make sure top_k is not larger than available chunks
    limit = min(top_k, len(ranked_indexes))

    for index in ranked_indexes[:limit]:

        results.append({
            "chunk": chunks[index],
            "score": round(
                float(similarities[index]),
                4
            )
        })

    return results


# ==========================================
# 5. Build Complete CV RAG
# ==========================================

def build_cv_rag(
    text: str
) -> Dict:
    """
    Prepare a CV for RAG.

    Pipeline:

    CV Text
       ↓
    Chunking
       ↓
    Gemini Embeddings
       ↓
    Vector Data
    """

    chunks = chunk_text(text)

    if not chunks:
        return {
            "chunks": [],
            "embeddings": [],
            "chunk_count": 0
        }

    embeddings = embed_chunks(chunks)

    return {
        "chunks": chunks,
        "embeddings": embeddings,
        "chunk_count": len(chunks)
    }


# ==========================================
# 6. Search CV Using RAG
# ==========================================

def search_cv(
    text: str,
    query: str,
    top_k: int = 3
) -> List[Dict]:
    """
    Complete RAG search.

    This function:
    1. Chunks the CV
    2. Creates embeddings
    3. Searches relevant chunks
    """

    rag_data = build_cv_rag(text)

    results = retrieve_relevant_chunks(
        query=query,
        chunks=rag_data["chunks"],
        embeddings=rag_data["embeddings"],
        top_k=top_k
    )

    return results
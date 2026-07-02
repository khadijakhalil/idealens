"""
IdeaLens RAG (Retrieval-Augmented Generation) Module.
Provides in-memory text embedding, chunking, and cosine similarity retrieval using Google GenAI text-embedding-004 with fallback.
"""

import os
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from dotenv import load_dotenv
from google import genai

logger = logging.getLogger("idealens.rag")

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """
    Chunks text into word-based chunks with overlap.
    
    Args:
        text (str): The raw document text.
        chunk_size (int): Max number of words per chunk.
        overlap (int): Number of overlapping words between consecutive chunks.
        
    Returns:
        List[str]: List of text chunks.
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [text]
        
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += (chunk_size - overlap)
        if i + chunk_size > len(words) and i < len(words):
            chunks.append(" ".join(words[i:]))
            break
    return chunks

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Calculates cosine similarity between two vector lists.
    """
    arr_a = np.array(a)
    arr_b = np.array(b)
    norm_a = np.linalg.norm(arr_a)
    norm_b = np.linalg.norm(arr_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(arr_a, arr_b) / (norm_a * norm_b))

class RAGService:
    """In-memory Vector RAG Service using text-embedding-004 with fallback to gemini-embedding-2."""

    def __init__(self, kb_root: str = "knowledge_base"):
        self.kb_root = os.path.abspath(kb_root)
        self._cached_kbs: Dict[str, List[Tuple[str, List[float], str]]] = {}
        self.embedding_model = None
        # Cache structure: { domain: [(chunk_text, embedding_vector, filename), ...] }

    def _get_client(self) -> genai.Client:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set. Please configure .env.")
        return genai.Client(api_key=api_key)

    def get_kb_path(self, domain: str) -> str:
        """Gets the path to a specific domain's knowledge base."""
        return os.path.join(self.kb_root, domain)

    def _embed_text(self, client: genai.Client, text: str) -> List[float]:
        """Generates embedding for a single text chunk with model discovery/fallback and retry logic."""
        import time
        max_retries = 4
        backoff = 1.5
        
        for attempt in range(max_retries):
            try:
                if self.embedding_model:
                    response = client.models.embed_content(
                        model=self.embedding_model,
                        contents=text
                    )
                    return response.embeddings[0].values

                # First run: auto-discover supported embedding model
                try:
                    response = client.models.embed_content(
                        model="text-embedding-004",
                        contents=text
                    )
                    self.embedding_model = "text-embedding-004"
                    logger.info("RAG: Using model 'text-embedding-004' for embeddings.")
                    return response.embeddings[0].values
                except Exception as e:
                    logger.warning("RAG: 'text-embedding-004' failed, falling back to 'gemini-embedding-2'. Error: %s", e)
                    response = client.models.embed_content(
                        model="gemini-embedding-2",
                        contents=text
                    )
                    self.embedding_model = "gemini-embedding-2"
                    logger.info("RAG: Using model 'gemini-embedding-2' for embeddings.")
                    return response.embeddings[0].values
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning("RAG: Embedding call failed (attempt %d/%d). Retrying in %.1f seconds... Error: %s", 
                                   attempt + 1, max_retries, backoff, e)
                    time.sleep(backoff)
                    backoff *= 2.0
                else:
                    raise e

    def ingest_directory(self, domain: str) -> int:
        """
        Ingests and embeds all .txt files from a domain's knowledge base directory into memory.
        Uses session-based caching.
        """
        if domain in self._cached_kbs:
            logger.info("RAG: Using cached knowledge base for domain '%s'", domain)
            return len(self._cached_kbs[domain])

        dir_path = self.get_kb_path(domain)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.warning("RAG: Directory '%s' does not exist. Created empty folder.", dir_path)
            self._cached_kbs[domain] = []
            return 0

        chunks_data = []
        # Find all .txt files
        for file in os.listdir(dir_path):
            if file.endswith(".txt"):
                file_path = os.path.join(dir_path, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    file_chunks = chunk_text(content, chunk_size=400, overlap=50)
                    for chunk in file_chunks:
                        chunks_data.append((chunk, file))
                except Exception as e:
                    logger.error("RAG: Failed to read file '%s': %s", file, e)

        if not chunks_data:
            logger.info("RAG: No .txt files found to ingest in '%s'", dir_path)
            self._cached_kbs[domain] = []
            return 0

        client = self._get_client()
        logger.info("RAG: Generating embeddings for %d chunks in domain '%s'", len(chunks_data), domain)
        
        try:
            loaded_data = []
            for chunk, filename in chunks_data:
                embedding = self._embed_text(client, chunk)
                loaded_data.append((chunk, embedding, filename))
                
            self._cached_kbs[domain] = loaded_data
            logger.info("RAG: Successfully cached %d chunks for domain '%s'", len(loaded_data), domain)
            return len(loaded_data)
            
        except Exception as e:
            logger.error("RAG: Failed to generate embeddings for domain '%s': %s", domain, e)
            raise e

    def retrieve(self, domain: str, query: str, top_k: int = 4) -> str:
        """
        Retrieves the top_k most relevant chunks using cosine similarity.
        
        Args:
            domain (str): The domain knowledge base (e.g., 'culture').
            query (str): The query string.
            top_k (int): Number of chunks to retrieve (defaults to 4).
            
        Returns:
            str: Formatted string containing the relevant chunks and their source filenames.
        """
        # Ensure directory is ingested/cached
        self.ingest_directory(domain)
        
        kb_chunks = self._cached_kbs.get(domain, [])
        if not kb_chunks:
            return f"No verified knowledge base records found in '{domain}'."

        client = self._get_client()
        try:
            query_vector = self._embed_text(client, query)
        except Exception as e:
            logger.error("RAG: Failed to embed query: %s", e)
            return f"RAG Retrieval Error: Failed to embed query due to: {e}"

        # Calculate similarity scores
        scores = []
        for chunk, emb, filename in kb_chunks:
            sim = cosine_similarity(query_vector, emb)
            scores.append((sim, chunk, filename))

        # Sort descending by similarity score
        scores.sort(key=lambda x: x[0], reverse=True)
        top_scores = scores[:top_k]

        # Format output
        formatted_results = []
        for i, (score, chunk, filename) in enumerate(top_scores):
            formatted_results.append(
                f"--- Chunk {i+1} [Source: {filename}] [Similarity: {score:.4f}] ---\n"
                f"{chunk.strip()}\n"
            )

        return "\n".join(formatted_results)

if __name__ == '__main__':
    # Run simple retrieval test
    print("Testing RAG Engine...")
    load_dotenv()
    
    rag = RAGService()
    try:
        # Ingest culture directory
        print("Ingesting culture knowledge base...")
        count = rag.ingest_directory("culture")
        print(f"Ingested {count} chunks.")
        
        # Test query
        test_query = "Japanese coffee loyalty seasonal"
        print(f"\nRetrieving for query: '{test_query}'...")
        output = rag.retrieve("culture", test_query)
        print("\n--- RETRIEVED RESULT ---")
        print(output)
    except Exception as e:
        print(f"RAG Test failed with error: {e}")

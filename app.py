from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from pdf_chunker import PDFChunker
from embed_store import HybridEmbedStore, FaissHybridEmbedStore
from llm_answer import LLMAnswer
import os
import logging

print("[INIT] Starting app initialization...")

app = FastAPI()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class QueryRequest(BaseModel):
    query: str

# Initialize components (load once)
PDF_PATH = "medicare-and-you.pdf"  # Using Medicare PDF for Medicare-related questions
CHROMA_DIR = "chroma_db"
llm = None
try:
    print("[INIT] Loading LLMAnswer...")
    llm = LLMAnswer()  # Uses default: llama2:7b
    print("[INIT] LLMAnswer loaded successfully.")
except Exception as e:
    print(f"[INIT] LLM not loaded: {e}")

try:
    print(f"[INIT] Loading and chunking PDF: {PDF_PATH}")
    chunker = PDFChunker(PDF_PATH)
    chunks = chunker.chunk_pdf()
    print(f"[INIT] PDF chunked into {len(chunks)} chunks.")
except Exception as e:
    print(f"[INIT] PDF chunking failed: {e}")
    chunks = []

try:
    print(f"[INIT] Initializing HybridEmbedStore at {CHROMA_DIR}")
    store = HybridEmbedStore(CHROMA_DIR)
    store.add_chunks(chunks)
    print("[INIT] Chunks added to HybridEmbedStore.")
except Exception as e:
    print(f"[INIT] HybridEmbedStore initialization failed: {e}")
    store = None

faiss_store = None
try:
    print(f"[INIT] Initializing FaissHybridEmbedStore")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    faiss_store = FaissHybridEmbedStore(dim=384)
    faiss_store.add_chunks(chunks, embedder)
    print("[INIT] Chunks added to FaissHybridEmbedStore.")
except Exception as e:
    print(f"[INIT] FaissHybridEmbedStore initialization failed: {e}")
    faiss_store = None

@app.get("/")
def read_root():
    return {"message": "Medicare RAG API is running", "status": "healthy"}

@app.get("/test")
def test_system():
    """Test endpoint to verify all components are working"""
    try:
        # Test LLM
        if llm is None:
            return {"error": "LLM not loaded"}
        
        # Test embedding store
        test_query = "test"
        test_hits = store.query(test_query, top_k=1)
        
        return {
            "status": "healthy",
            "llm_loaded": llm is not None,
            "store_working": len(test_hits) > 0,
            "total_chunks_indexed": len(test_hits) if test_hits else 0
        }
    except Exception as e:
        return {"error": f"System test failed: {str(e)}"}

@app.get("/debug/chunks")
def debug_chunks():
    """Debug endpoint to see what chunks are available"""
    try:
        # Get a few sample chunks
        test_queries = ["coverage", "medicare", "options", "benefits"]
        results = {}
        
        for query in test_queries:
            hits = store.query(query, top_k=2)
            results[query] = []
            for hit in hits:
                results[query].append({
                    "page": hit["page"],
                    "method": hit.get("method", "unknown"),
                    "similarity": hit.get("similarity", 0.0),
                    "preview": hit["chunk"][:200] + "..." if len(hit["chunk"]) > 200 else hit["chunk"]
                })
        
        return {
            "total_chunks": "405 (from logs)",
            "sample_queries": results
        }
    except Exception as e:
        return {"error": f"Debug failed: {str(e)}"}

# --- Retrieval validation node ---
def validate_retrieval(query, hits, min_overlap=1):
    query_words = set(query.lower().split())
    for hit in hits:
        chunk_words = set(hit['chunk'].lower().split())
        if len(query_words & chunk_words) >= min_overlap:
            return True
    return False

# --- LLM-based query rephrasing ---
def rephrase_query_with_llm(llm, query):
    prompt = f"Rephrase the following question to maximize the chance of finding relevant information in a Medicare handbook.\n\nQuestion: {query}\n\nRephrased:"
    try:
        response = llm.get_answer(prompt, [])
        return response['answer'].strip()
    except Exception as e:
        return query  # fallback

@app.post("/ask")
def ask_question(request: QueryRequest) -> Dict:
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty.")
        logger.info(f"[ASK] Processing query: {query}")
        max_attempts = 2
        attempt = 0
        validated = False
        orig_query = query
        chroma_hits = []
        faiss_hits = []
        while attempt < max_attempts and not validated:
            chroma_hits = store.query(query, top_k=5)
            faiss_hits = faiss_store.query(query, embedder, top_k=5) if faiss_store else []
            # Log Chroma retrieval
            logger.info(f"[ASK] ChromaDB retrieval for query: '{query}' - {len(chroma_hits)} hits")
            for i, hit in enumerate(chroma_hits):
                preview = hit['chunk'][:150] + "..." if len(hit['chunk']) > 150 else hit['chunk']
                logger.info(f"[ASK][Chroma] Hit {i+1} (Page {hit['page']}, Score: {hit.get('similarity', 0.0):.3f}, Type: {hit.get('type', 'unknown')}): {preview}")
            # Log FAISS retrieval
            logger.info(f"[ASK] FAISS retrieval for query: '{query}' - {len(faiss_hits)} hits")
            for i, hit in enumerate(faiss_hits):
                preview = hit['chunk'][:150] + "..." if len(hit['chunk']) > 150 else hit['chunk']
                logger.info(f"[ASK][FAISS] Hit {i+1} (Page {hit['page']}, Score: {hit.get('similarity', 0.0):.3f}, Type: {hit.get('type', 'unknown')}): {preview}")
            combined_hits = chroma_hits + faiss_hits
            validated = validate_retrieval(query, combined_hits)
            if not validated:
                logger.info(f"[ASK] Retrieval not validated for query: '{query}'. Rephrasing and retrying...")
                query = rephrase_query_with_llm(llm, orig_query)
            attempt += 1
        if not chroma_hits and not faiss_hits:
            logger.warning("[ASK] No relevant chunks found in either store")
            return {
                "answer": "No relevant information found in the document.",
                "source_pages": [],
              #  #"primary_source_page": None,
                "confidence_score": 0.0,
                "chunk_size": 0,
                "total_chunks_used": 0
            }
        if llm is None:
            logger.error("[ASK] LLM not loaded")
            return {
                "answer": "LLM model not loaded. Please check the server logs.",
                "source_pages": [chroma_hits[0]["page"]] if chroma_hits else [],
               # #"primary_source_page": chroma_hits[0]["page"] if chroma_hits else None,
                "confidence_score": 0.0,
                "chunk_size": chroma_hits[0].get("chunk_size", 0) if chroma_hits else 0,
                "total_chunks_used": 0
            }
        logger.info("[ASK] Calling LLM for answer generation with combined retrievals")
        logger.info(f"[ASK] Total combined retrievals sent to LLM: {len(chroma_hits) + len(faiss_hits)}")
        result = llm.get_answer(orig_query, (chroma_hits + faiss_hits))
        logger.info(f"[ASK] LLM returned answer: {result['answer'][:100]}...")
        return result
    except Exception as e:
        logger.error(f"[ASK] Exception in ask_question: {e}")
        return {
            "answer": f"Error processing your question: {str(e)}",
            "source_pages": [],
            #"primary_source_page": None,
            "confidence_score": 0.0,
            "chunk_size": 0,
            "total_chunks_used": 0
        } 
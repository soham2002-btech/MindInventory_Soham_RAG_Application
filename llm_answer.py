import requests
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMAnswer:
    def __init__(self, ollama_url="http://localhost:11434/api/generate", model_name="llama2:7b"):
        logger.info(f"Initializing LLMAnswer with Ollama URL: {ollama_url}")
        logger.info(f"Using model: {model_name}")
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Loaded sentence transformer for confidence scoring")
        
        # Test Ollama connection
        self._test_ollama_connection()

    def _test_ollama_connection(self):
        """Test if Ollama is running and accessible"""
        try:
            logger.info("Testing Ollama connection...")
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Ollama is running and accessible")
            else:
                logger.warning("⚠️ Ollama responded but with unexpected status")
        except Exception as e:
            logger.error(f"❌ Cannot connect to Ollama: {e}")
            logger.error("Make sure Ollama is running: ollama serve")

    def rerank_chunks(self, query, chunks, top_k=5):
        # Compute embedding for query
        query_emb = self.embedder.encode([query])[0]
        # Compute embedding for each chunk
        chunk_embs = self.embedder.encode([c['chunk'] for c in chunks])
        # Compute cosine similarity
        scores = [float(np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))) for emb in chunk_embs]
        # Attach scores and sort
        for c, s in zip(chunks, scores):
            c['rerank_score'] = s
        reranked = sorted(chunks, key=lambda c: c['rerank_score'], reverse=True)
        return reranked[:top_k]

    def build_prompt(self, query, chunks):
        # Use all reranked top 5 chunks for better context
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(f"Page {chunk['page']}: {chunk['chunk']}")
        context = "\n\n".join(context_parts)
        prompt = (
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            "Instructions: Using ONLY the information in the context above, provide a complete and detailed answer. "
            f"Always generate the answer for part which is related to {query} Don't generate answer for other parts. even if it is present in Context"
            "List ALL requirements, conditions, or relevant facts mentioned in the context. "
            "If the answer is not in the context, respond with 'Not found in context'.\n\n"
            "Answer:"
        )
        logger.info(f"Built prompt with context from pages {[c['page'] for c in chunks]}")
        return prompt

    def _clean_response(self, answer: str) -> str:
        """Clean up generic AI responses"""
        # Simple cleaning - just remove obvious generic phrases
        generic_phrases = [
            "i'm happy to help",
            "i'd be happy to help", 
            "i'm here to help",
            "of course!",
            "of course,",
            "i'm sorry, but",
            "unfortunately,",
            "i cannot provide",
            "i don't have access to",
            "i'm an ai",
            "i'm an ai assistant",
            "as an ai",
            "as an ai assistant"
        ]
        
        answer_lower = answer.lower()
        
        # Remove generic phrases
        for phrase in generic_phrases:
            answer = answer.replace(phrase, "").replace(phrase.title(), "")
        
        # Clean up extra whitespace
        answer = " ".join(answer.split())
        
        return answer

    def _validate_answer(self, answer: str, query: str) -> str:
        """Validate that the answer addresses the original question"""
        answer_lower = answer.lower()
        query_lower = query.lower()
        
        # If answer is empty or just whitespace
        if not answer.strip():
            return "Not found in context"
        
        # If answer is too short (less than 5 characters), it's probably not useful
        if len(answer.strip()) < 5:
            return "Not found in context"
        
        # Check if the LLM is trying to change the question format
        if "question:" in answer_lower and "answer:" in answer_lower:
            # Extract just the answer part
            if "answer:" in answer_lower:
                answer = answer_lower.split("answer:")[-1].strip()
                if not answer:
                    return "Not found in context"
        
        # If the answer contains "not found" but is longer than that phrase,
        # it might be a valid answer that happens to contain those words
        if "not found" in answer_lower and len(answer.strip()) > 20:
            # Check if the answer contains key terms from the query
            query_words = [word for word in query_lower.split() if len(word) > 3]
            if query_words and any(word in answer_lower for word in query_words):
                # This might be a valid answer, don't reject it
                return answer
        
        return answer

    def get_answer(self, query, chunks):
        try:
            logger.info(f"Processing query: '{query}'")
            logger.info(f"Using {len(chunks)} chunks, reranking for top 5")
            # Rerank and select top 5
            reranked_chunks = self.rerank_chunks(query, chunks, top_k=5)
            logger.info(f"Top 5 reranked pages: {[c['page'] for c in reranked_chunks]}")
            prompt = self.build_prompt(query, reranked_chunks)
            logger.info("Sending request to Ollama...")
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 1024,  # or higher, up to the model's context window
                    "temperature": 0,
                    "top_p": 0.9
                }
            }
            
            # Try with shorter timeout first
            try:
                response = requests.post(self.ollama_url, json=payload, timeout=30)
                response.raise_for_status()
            except requests.exceptions.Timeout:
                logger.warning("First attempt timed out, trying with longer timeout...")
                response = requests.post(self.ollama_url, json=payload, timeout=60)
                response.raise_for_status()
            
            result_json = response.json()
            logger.info("Received response from Ollama")
            
            answer = result_json.get("response", "").strip()
            if not answer:
                logger.warning("Empty response from Ollama, using fallback")
                answer = "Not found in context"
            
            # Log the raw response for debugging
            logger.info(f"Raw LLM response: {answer[:200]}...")
            
            # Clean up the response
            answer = self._clean_response(answer)
            
            # Validate the answer
            answer = self._validate_answer(answer, query)
            
            logger.info(f"Final answer: {answer[:100]}...")
            
            # Calculate confidence score
            try:
                query_emb = self.embedder.encode(query)
                chunk_emb = self.embedder.encode(reranked_chunks[0]["chunk"])
                confidence = float(np.dot(query_emb, chunk_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb)))
            except Exception as e:
                logger.warning(f"Could not calculate confidence score: {e}")
                confidence = 0.5
            
            logger.info(f"Confidence score: {confidence:.4f}")
            
            # If answer contains 'not found in context' (case-insensitive), always clear all source metadata
            if 'not found in context' in answer.strip().lower():
                result = {
                    "answer": answer,
                    "source_pages": [],
                    #"primary_source_page": None,
                    "confidence_score": 0.0,
                    "chunk_size": 0,
                    "total_chunks_used": 0
                }
                logger.info("No relevant context found (final robust check), returning empty source metadata.")
                return result

            # Get all unique pages used in the context
            pages_used = list(set([chunk['page'] for chunk in reranked_chunks]))
            pages_used.sort()  # Sort for consistent ordering
            
            result = {
                "answer": answer,
                "source_pages": pages_used,  # Return all pages used
                #"primary_source_page": reranked_chunks[0]["page"],  # Keep the top page as primary
                "confidence_score": confidence,
                "chunk_size": reranked_chunks[0].get("chunk_size", len(reranked_chunks[0]["chunk"].split())),
                "total_chunks_used": len(reranked_chunks)  # Number of chunks used for context
            }
            
            logger.info("Successfully generated answer")
            return result
            
        except requests.exceptions.ConnectionError:
            logger.error("❌ Cannot connect to Ollama. Make sure it's running: ollama serve")
            return {
                "answer": "Error: Cannot connect to Ollama. Please make sure it's running.",
                "source_pages": [chunks[0]["page"]] if chunks else [],
                #"primary_source_page": chunks[0]["page"] if chunks else None,
                "confidence_score": 0.0,
                "chunk_size": 0,
                "total_chunks_used": 0
            }
        except requests.exceptions.Timeout:
            logger.error("❌ Ollama request timed out")
            return {
                "answer": "Error: Request timed out. The model might be too slow.",
                "source_pages": [chunks[0]["page"]] if chunks else [],
                #"primary_source_page": chunks[0]["page"] if chunks else None,
                "confidence_score": 0.0,
                "chunk_size": 0,
                "total_chunks_used": 0
            }
        except Exception as e:
            logger.error(f"❌ Error in get_answer: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "source_pages": [chunks[0]["page"]] if chunks else [],
                #"primary_source_page": chunks[0]["page"] if chunks else None,
                "confidence_score": 0.0,
                "chunk_size": 0,
                "total_chunks_used": 0
            } 
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Tuple
import os
import logging
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import faiss

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridEmbedStore:
    def __init__(self, persist_dir: str = "chroma_db"):
        logger.info(f"Initializing HybridEmbedStore with persist directory: {persist_dir}")
        
        # Multiple embedding models for better coverage
        self.models = {
            'dense_small': SentenceTransformer('all-MiniLM-L6-v2'),
            'dense_large': SentenceTransformer('all-mpnet-base-v2'),  # Better quality
            'dense_multilingual': SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        }
        logger.info("Loaded multiple sentence transformer models")
        
        # TF-IDF for sparse vectors
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2
        )
        self.tfidf_matrix = None
        self.tfidf_feature_names = None
        
        # ChromaDB client
        self.client = chromadb.Client(Settings(
            persist_directory=persist_dir
        ))
        
        # Multiple collections for different embedding types
        self.collections = {}
        for model_name in self.models.keys():
            collection_name = f"medicare_chunks_{model_name}"
            self.collections[model_name] = self.client.get_or_create_collection(collection_name)
            logger.info(f"Created/loaded ChromaDB collection: {collection_name}")
        
        # Store all chunks for TF-IDF
        self.all_chunks = []
        self.all_chunk_texts = []

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for better matching"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\-\.\,\!\?\(\)]', '', text)
        return text.lower()

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        # Extract words and filter
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Add Medicare-specific terms
        medicare_terms = ['medicare', 'part', 'advantage', 'supplement', 'coverage', 'enrollment', 'premium', 'deductible', 'coinsurance', 'hospital', 'doctor', 'prescription', 'drug', 'benefit', 'plan', 'insurance']
        keywords.extend([term for term in medicare_terms if term in query.lower()])
        
        return list(set(keywords))

    def _generate_search_terms(self, query: str) -> List[str]:
        """Generate multiple search terms for better retrieval"""
        original_query = query.strip()
        keywords = self._extract_keywords(query)
        
        search_terms = [original_query]
        
        # Add keyword combinations
        if len(keywords) >= 2:
            search_terms.append(' '.join(keywords[:3]))  # Top 3 keywords
            search_terms.append(' '.join(keywords[:2]))  # Top 2 keywords
        
        # Add Medicare-specific variations
        if 'medicare' in query.lower():
            search_terms.extend([
                'medicare coverage',
                'medicare benefits',
                'medicare enrollment',
                'medicare plans'
            ])
        
        # Add specific Medicare parts if mentioned
        if 'part a' in query.lower() or 'parta' in query.lower():
            search_terms.extend(['part a', 'hospital insurance', 'inpatient care'])
        if 'part b' in query.lower() or 'partb' in query.lower():
            search_terms.extend(['part b', 'medical insurance', 'outpatient care'])
        if 'part c' in query.lower() or 'partc' in query.lower():
            search_terms.extend(['part c', 'medicare advantage', 'advantage plan'])
        if 'part d' in query.lower() or 'partd' in query.lower():
            search_terms.extend(['part d', 'prescription drug', 'drug coverage'])
        
        return list(set(search_terms))

    def add_chunks(self, chunks: List[Dict]):
        """Add chunks to all embedding stores"""
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        logger.info(f"Adding {len(chunks)} chunks to the hybrid embedding store")
        
        # Store chunks for TF-IDF
        self.all_chunks = chunks
        self.all_chunk_texts = [chunk['chunk'] for chunk in chunks]
        
        # Prepare batch data for each model
        batch_size = 50
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        for model_name, model in self.models.items():
            logger.info(f"Indexing with {model_name} model...")
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(chunks))
                
                batch_chunks = chunks[start_idx:end_idx]
                
                # Prepare batch data
                documents = []
                metadatas = []
                ids = []
                
                for i, chunk in enumerate(batch_chunks):
                    documents.append(chunk['chunk'])
                    metadatas.append({
                        'page': chunk['page'],
                        'word_count': len(chunk['chunk'].split()),
                        'model': model_name
                    })
                    ids.append(f"chunk_{start_idx + i}_{model_name}")
                
                logger.info(f"Indexing batch {batch_idx + 1}/{total_batches} ({len(documents)} chunks) with {model_name}")
                
                try:
                    self.collections[model_name].add(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids
                    )
                except Exception as e:
                    logger.error(f"Error adding batch {batch_idx + 1} with {model_name}: {e}")
                    raise
        
        # Build TF-IDF matrix
        logger.info("Building TF-IDF matrix for sparse retrieval...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.all_chunk_texts)
        self.tfidf_feature_names = self.tfidf_vectorizer.get_feature_names_out()
        logger.info(f"TF-IDF matrix built with {self.tfidf_matrix.shape[1]} features")
        
        logger.info(f"Successfully indexed all {len(chunks)} chunks with hybrid approach")

    def _sparse_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Perform sparse vector search using TF-IDF"""
        if self.tfidf_matrix is None:
            return []
        
        # Transform query
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        hits = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include relevant results
                hit = {
                    'chunk': self.all_chunks[idx]['chunk'],
                    'page': self.all_chunks[idx]['page'],
                    'word_count': len(self.all_chunks[idx]['chunk'].split()),
                    'similarity': float(similarities[idx]),
                    'method': 'sparse'
                }
                hits.append(hit)
        
        return hits

    def _dense_search(self, query: str, model_name: str, top_k: int = 3) -> List[Dict]:
        """Perform dense vector search using specified model"""
        try:
            results = self.collections[model_name].query(
                query_texts=[query],
                n_results=top_k
            )
            
            hits = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    hit = {
                        'chunk': doc,
                        'page': results['metadatas'][0][i]['page'],
                        'word_count': results['metadatas'][0][i]['word_count'],
                        'similarity': 1.0,  # ChromaDB doesn't return similarity scores by default
                        'method': f'dense_{model_name}'
                    }
                    hits.append(hit)
            
            return hits
            
        except Exception as e:
            logger.error(f"Error in dense search with {model_name}: {e}")
            return []

    def _keyword_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Perform keyword-based search"""
        keywords = self._extract_keywords(query)
        if not keywords:
            return []
        
        hits = []
        keyword_scores = {}
        
        for i, chunk in enumerate(self.all_chunks):
            chunk_text = chunk['chunk'].lower()
            score = 0
            
            # Count keyword matches
            for keyword in keywords:
                if keyword in chunk_text:
                    score += chunk_text.count(keyword)
            
            if score > 0:
                keyword_scores[i] = score
        
        # Sort by score and get top k
        sorted_indices = sorted(keyword_scores.keys(), key=lambda x: keyword_scores[x], reverse=True)[:top_k]
        
        for idx in sorted_indices:
            hit = {
                'chunk': self.all_chunks[idx]['chunk'],
                'page': self.all_chunks[idx]['page'],
                'word_count': len(self.all_chunks[idx]['chunk'].split()),
                'similarity': keyword_scores[idx],
                'method': 'keyword'
            }
            hits.append(hit)
        
        return hits

    def query(self, query: str, top_k: int = 3) -> List[Dict]:
        """Hybrid query combining multiple search strategies"""
        logger.info(f"Hybrid querying for: '{query}' (top_k={top_k})")
        
        # Generate multiple search terms
        search_terms = self._generate_search_terms(query)
        logger.info(f"Generated search terms: {search_terms}")
        
        all_hits = []
        
        # 1. Dense search with multiple models
        for model_name in self.models.keys():
            for term in search_terms[:2]:  # Use top 2 search terms per model
                hits = self._dense_search(term, model_name, top_k=2)
                all_hits.extend(hits)
        
        # 2. Sparse search
        for term in search_terms[:2]:
            hits = self._sparse_search(term, top_k=2)
            all_hits.extend(hits)
        
        # 3. Keyword search
        hits = self._keyword_search(query, top_k=3)
        all_hits.extend(hits)
        
        # 4. Direct query search
        direct_hits = self._dense_search(query, 'dense_small', top_k=3)
        all_hits.extend(direct_hits)
        
        # Deduplicate and rank results
        unique_hits = {}
        for hit in all_hits:
            # Create unique key based on chunk content
            chunk_key = hit['chunk'][:100]  # First 100 chars as key
            
            if chunk_key not in unique_hits:
                unique_hits[chunk_key] = hit
            else:
                # If we have multiple methods finding the same chunk, boost the score
                unique_hits[chunk_key]['similarity'] = max(
                    unique_hits[chunk_key]['similarity'], 
                    hit['similarity']
                )
                unique_hits[chunk_key]['method'] += f"+{hit['method']}"
        
        # Sort by similarity and get top k
        final_hits = sorted(
            unique_hits.values(), 
            key=lambda x: x['similarity'], 
            reverse=True
        )[:top_k]
        
        logger.info(f"Found {len(final_hits)} relevant chunks using hybrid search")
        logger.info(f"Search methods used: {[hit['method'] for hit in final_hits]}")
        
        return final_hits

# Backward compatibility
class EmbedStore(HybridEmbedStore):
    """Backward compatibility wrapper"""
    pass

class MilvusHybridEmbedStore:
    def __init__(self, host="localhost", port="19530", collection_name="medicare_chunks_milvus"):
        self.collection_name = collection_name
        self.dim = 384  # Assuming MiniLM or similar
        self._connect(host, port)
        self._create_collection_if_not_exists()

    def _connect(self, host, port):
        connections.connect(host=host, port=port)

    def _create_collection_if_not_exists(self):
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            return
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="page", dtype=DataType.INT64),
            FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="chunk_size", dtype=DataType.INT64),
            FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=32)
        ]
        schema = CollectionSchema(fields, description="Medicare hybrid chunks")
        self.collection = Collection(self.collection_name, schema)
        self.collection.load()

    def add_chunks(self, chunks, embedder):
        # Chunks: list of dicts with 'chunk', 'page', 'chunk_size', 'type'
        embeddings = embedder.encode([c['chunk'] for c in chunks])
        data = [
            [None] * len(chunks),  # id (auto)
            [list(map(float, emb)) for emb in embeddings],
            [c['page'] for c in chunks],
            [c['chunk'] for c in chunks],
            [c['chunk_size'] for c in chunks],
            [c['type'] for c in chunks]
        ]
        self.collection.insert(data)
        self.collection.flush()

    def query(self, query, embedder, top_k=5):
        query_emb = embedder.encode([query])[0]
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[list(map(float, query_emb))],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["page", "chunk", "chunk_size", "type"]
        )
        hits = []
        for hit in results[0]:
            hits.append({
                "chunk": hit.entity.get("chunk"),
                "page": hit.entity.get("page"),
                "chunk_size": hit.entity.get("chunk_size"),
                "type": hit.entity.get("type"),
                "similarity": hit.distance
            })
        return hits

class FaissHybridEmbedStore:
    def __init__(self, dim=384):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []  # List of dicts: chunk, page, chunk_size, type

    def add_chunks(self, chunks, embedder):
        # Chunks: list of dicts with 'chunk', 'page', 'chunk_size', 'type'
        embeddings = embedder.encode([c['chunk'] for c in chunks])
        embeddings = np.array(embeddings).astype('float32')
        self.index.add(embeddings)
        self.metadata.extend([
            {
                'chunk': c['chunk'],
                'page': c['page'],
                'chunk_size': c['chunk_size'],
                'type': c['type']
            } for c in chunks
        ])

    def query(self, query, embedder, top_k=5):
        query_emb = embedder.encode([query])
        query_emb = np.array(query_emb).astype('float32')
        D, I = self.index.search(query_emb, top_k)
        hits = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            hits.append({
                'chunk': meta['chunk'],
                'page': meta['page'],
                'chunk_size': meta['chunk_size'],
                'type': meta['type'],
                'similarity': float(score)
            })
        return hits

if __name__ == "__main__":
    from pdf_chunker import PDFChunker
    chunker = PDFChunker("medicare-and-you.pdf")
    chunks = chunker.chunk_pdf()
    store = HybridEmbedStore()
    store.add_chunks(chunks)
    print(store.query("What are the eligibility criteria for Medicare Advantage Plans?")) 
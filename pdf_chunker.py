import pdfplumber
import nltk
from typing import List, Dict
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download('punkt')

from nltk.tokenize import sent_tokenize

class PDFChunker:
    def __init__(self, pdf_path: str, chunk_size: int = 1000, overlap: int = 100):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size  # Increased chunk size for fewer chunks
        self.overlap = overlap  # Overlap between chunks

    def extract_text_by_page(self) -> List[str]:
        pages = []
        logger.info(f"Extracting text from PDF: {self.pdf_path}")
        
        with pdfplumber.open(self.pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"PDF has {total_pages} pages")
            
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    pages.append(text.strip())
                    if page_num % 10 == 0:  # Log every 10 pages
                        logger.info(f"Processed {page_num}/{total_pages} pages")
                else:
                    logger.warning(f"Page {page_num} is empty, skipping")
        
        logger.info(f"Extracted text from {len(pages)} non-empty pages")
        return pages

    def semantic_chunk_text(self, text: str, page_num: int, para_per_chunk: int = 2) -> List[Dict]:
        """Create semantic (paragraph-based) chunks, grouping paragraphs together."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        i = 0
        while i < len(paragraphs):
            chunk_paras = paragraphs[i:i+para_per_chunk]
            chunk_text = '\n\n'.join(chunk_paras)
            if chunk_text:
                word_count = len(chunk_text.split())
                chunks.append({
                    'chunk': chunk_text,
                    'page': page_num,
                    'chunk_size': word_count,
                    'type': 'semantic'
                })
            i += para_per_chunk
        return chunks

    def chunk_text(self, text: str, page_num: int) -> List[Dict]:
        """Create fixed-size chunks with overlap (sliding window)."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            # If this is not the last chunk, try to break at a sentence boundary
            if end < len(text):
                for i in range(end, max(start + self.chunk_size - 200, start), -1):
                    if i < len(text) and text[i] in '.!?':
                        end = i + 1
                        break
            chunk_text = text[start:end].strip()
            if chunk_text:
                word_count = len(chunk_text.split())
                chunks.append({
                    'chunk': chunk_text,
                    'page': page_num,
                    'chunk_size': word_count,
                    'type': 'sliding'
                })
            start = end - self.overlap
            if start >= len(text):
                break
        return chunks

    def chunk_pdf(self) -> List[Dict]:
        """Semantic chunking: only paragraph-based chunks for each page."""
        pages = self.extract_text_by_page()
        all_chunks = []
        for page_num, page_text in enumerate(pages, 1):
            semantic_chunks = self.semantic_chunk_text(page_text, page_num)
            all_chunks.extend(semantic_chunks)
            # Log progress every 20 pages
            if page_num % 20 == 0 or page_num == len(pages):
                logger.info(f"Processed page {page_num}/{len(pages)}: {len(semantic_chunks)} semantic chunks")
        logger.info(f"Total chunks created: {len(all_chunks)} (semantic only)")
        return all_chunks

if __name__ == "__main__":
    chunker = PDFChunker("medicare-and-you.pdf")
    chunks = chunker.chunk_pdf()
    print(f"Extracted {len(chunks)} chunks.")
    # Print first few chunks for inspection
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i + 1} (Page {chunk['page']}):")
        print(f"Words: {chunk['chunk_size']}, Chars: {len(chunk['chunk'])}")
        print(f"Text: {chunk['chunk'][:200]}...") 
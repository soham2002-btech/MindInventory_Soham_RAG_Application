import os
from pdf_chunker import PDFChunker
from collections import Counter
import numpy as np
import csv

# Expanded, rigorous test queries (positive, negative, multi-hop, ambiguous, adversarial, paraphrased)
TEST_QUERIES = [
    {
        'query': 'What is an appeal?',
        'answer_keywords': ['appeal', 'disagree', 'decision', 'medicare'],
        'type': 'positive'
    },
    {
        'query': 'What is TRICARE?',
        'answer_keywords': ['tricare', 'military', 'defense'],
        'type': 'positive'
    },
    {
        'query': 'What is Medicaid?',
        'answer_keywords': ['medicaid', 'low-income', 'state', 'coverage'],
        'type': 'positive'
    },
    {
        'query': 'How do I appeal a denied claim?',
        'answer_keywords': ['appeal', 'denied', 'claim', 'process'],
        'type': 'multi-hop'
    },
    {
        'query': 'What is the capital of India?',
        'answer_keywords': ['delhi', 'new delhi'],
        'type': 'negative'
    },
    {
        'query': 'Explain the difference between Medicare and Medicaid.',
        'answer_keywords': ['difference', 'medicare', 'medicaid'],
        'type': 'ambiguous'
    },
    {
        'query': 'How to get from Pune to Mumbai?',
        'answer_keywords': ['train', 'bus', 'car', 'distance'],
        'type': 'negative'
    },
    {
        'query': 'What is the process for a fast appeal?',
        'answer_keywords': ['fast appeal', 'notice', 'reviewer', 'decision'],
        'type': 'adversarial'
    },
    {
        'query': 'How can I challenge a Medicare decision?',
        'answer_keywords': ['challenge', 'appeal', 'decision', 'medicare'],
        'type': 'paraphrased'
    },
    # Add more as needed
]

PDF_PATH = 'medicare-and-you.pdf'

# Helper: Calculate redundancy (fraction of repeated text across chunks)
def calc_redundancy(chunks):
    seen = set()
    repeated = 0
    for c in chunks:
        text = c['chunk']
        if text in seen:
            repeated += 1
        else:
            seen.add(text)
    return repeated / len(chunks) if chunks else 0

# Helper: Average chunk size (words)
def avg_chunk_size(chunks):
    return np.mean([c['chunk_size'] for c in chunks]) if chunks else 0

def chunk_size_stats(chunks):
    sizes = [c['chunk_size'] for c in chunks]
    return {
        'min': int(np.min(sizes)) if sizes else 0,
        'max': int(np.max(sizes)) if sizes else 0,
        'std': round(float(np.std(sizes)), 2) if sizes else 0
    }

# Helper: Unique pages covered
def unique_pages(chunks):
    return len(set(c['page'] for c in chunks))

# Helper: For a query, does any chunk contain an answer keyword?
def retrieval_coverage(chunks, answer_keywords, k=5):
    found = 0
    for c in chunks[:k]:
        text = c['chunk'].lower()
        if any(kw in text for kw in answer_keywords):
            found += 1
    return found > 0

# Helper: Recall@k for all queries
def recall_at_k(chunks, queries, k=5):
    recall = []
    for q in queries:
        recall.append(retrieval_coverage(chunks, q['answer_keywords'], k=k))
    return sum(recall) / len(recall) if recall else 0

# Helper: Precision@k for all queries
def precision_at_k(chunks, queries, k=5):
    precisions = []
    for q in queries:
        relevant = 0
        for c in chunks[:k]:
            text = c['chunk'].lower()
            if any(kw in text for kw in q['answer_keywords']):
                relevant += 1
        precisions.append(relevant / k)
    return sum(precisions) / len(precisions) if precisions else 0

# Helper: F1@k for all queries (if ground-truth available, here we use answer_keywords as proxy)
def f1_at_k(chunks, queries, k=5):
    f1s = []
    for q in queries:
        relevant = 0
        for c in chunks[:k]:
            text = c['chunk'].lower()
            if any(kw in text for kw in q['answer_keywords']):
                relevant += 1
        precision = relevant / k
        recall = 1.0 if relevant > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0

# Helper: Overlap between chunks (fraction of tokens in >1 chunk)
def calc_overlap(chunks):
    token_counts = Counter()
    for c in chunks:
        for word in set(c['chunk'].split()):
            token_counts[word] += 1
    overlap = sum(1 for v in token_counts.values() if v > 1)
    total = len(token_counts)
    return round(overlap / total, 3) if total else 0

# Print top failing queries for each chunker
def print_failing_queries(chunker_name, chunks, queries, k=5):
    print(f"\nTop failing queries for {chunker_name} (not found in top-{k}):")
    for q in queries:
        if not retrieval_coverage(chunks, q['answer_keywords'], k=k):
            print(f"  - {q['query']}")

# Export results to CSV
def export_to_csv(results, filename="chunker_metrics.csv"):
    if not results:
        return
    keys = list(results[0].keys())
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\nResults exported to {filename}")

# Run evaluation for each chunker
def evaluate_chunkers():
    results = []
    chunker = PDFChunker(PDF_PATH)
    pages = chunker.extract_text_by_page()

    # 1. Sliding window
    sliding_chunks = []
    for page_num, page_text in enumerate(pages, 1):
        sliding_chunks.extend(chunker.chunk_text(page_text, page_num))
    
    # 2. Semantic
    semantic_chunks = []
    for page_num, page_text in enumerate(pages, 1):
        semantic_chunks.extend(chunker.semantic_chunk_text(page_text, page_num))
    
    # 3. Hybrid (both)
    hybrid_chunks = sliding_chunks + semantic_chunks
    # 4. Current (complex/hybrid) chunker
    current_chunks = chunker.chunk_pdf()

    chunkers = [
        ('sliding', sliding_chunks),
        ('semantic', semantic_chunks),
        ('hybrid', hybrid_chunks),
        ('current', current_chunks)
    ]

    for name, chunks in chunkers:
        stats = chunk_size_stats(chunks)
        metric = {
            'chunker': name,
            'num_chunks': len(chunks),
            'avg_chunk_size': round(avg_chunk_size(chunks), 2),
            'min_chunk': stats['min'],
            'max_chunk': stats['max'],
            'std_chunk': stats['std'],
            'redundancy': round(calc_redundancy(chunks), 3),
            'overlap': calc_overlap(chunks),
            'unique_pages': unique_pages(chunks),
            'recall@1': round(recall_at_k(chunks, TEST_QUERIES, k=1), 2),
            'recall@3': round(recall_at_k(chunks, TEST_QUERIES, k=3), 2),
            'recall@5': round(recall_at_k(chunks, TEST_QUERIES, k=5), 2),
            'recall@10': round(recall_at_k(chunks, TEST_QUERIES, k=10), 2),
            'precision@5': round(precision_at_k(chunks, TEST_QUERIES, k=5), 2),
            'f1@5': round(f1_at_k(chunks, TEST_QUERIES, k=5), 2)
        }
        # Coverage for positive, negative, multi-hop, ambiguous, adversarial, paraphrased
        for t in ['positive', 'negative', 'multi-hop', 'ambiguous', 'adversarial', 'paraphrased']:
            subset = [q for q in TEST_QUERIES if q['type'] == t]
            if subset:
                metric[f'coverage_{t}'] = f"{sum(retrieval_coverage(chunks, q['answer_keywords'], 5) for q in subset)}/{len(subset)}"
        results.append(metric)
        print_failing_queries(name, chunks, TEST_QUERIES, k=5)

    # Print results as a detailed table
    print("\nUltra-Rigorous Chunker Performance Comparison:")
    print(f"{'Chunker':<10} {'#Chunks':<8} {'AvgSz':<7} {'Min':<5} {'Max':<5} {'Std':<5} {'Redund.':<8} {'Ovlp':<6} {'Pages':<6} {'R@1':<5} {'R@3':<5} {'R@5':<5} {'R@10':<6} {'P@5':<5} {'F1@5':<6} {'PosCov':<8} {'NegCov':<8} {'MultiCov':<9} {'AmbigCov':<9} {'AdvCov':<8} {'ParaCov':<8}")
    for m in results:
        print(f"{m['chunker']:<10} {m['num_chunks']:<8} {m['avg_chunk_size']:<7} {m['min_chunk']:<5} {m['max_chunk']:<5} {m['std_chunk']:<5} {m['redundancy']:<8} {m['overlap']:<6} {m['unique_pages']:<6} {m['recall@1']:<5} {m['recall@3']:<5} {m['recall@5']:<5} {m['recall@10']:<6} {m['precision@5']:<5} {m['f1@5']:<6} {m.get('coverage_positive','-'):<8} {m.get('coverage_negative','-'):<8} {m.get('coverage_multi-hop','-'):<9} {m.get('coverage_ambiguous','-'):<9} {m.get('coverage_adversarial','-'):<8} {m.get('coverage_paraphrased','-'):<8}")

    export_to_csv(results)

if __name__ == "__main__":
    evaluate_chunkers() 
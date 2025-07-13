# Medicare RAG Project

This project is a Retrieval-Augmented Generation (RAG) system for answering Medicare-related questions using a PDF document as the knowledge base. It uses FastAPI, hybrid vector search (ChromaDB + FAISS), and an LLM (Ollama/llama2:7b) for answer generation.

## RAG Pipeline Flow

![RAG Pipeline Flow](images/rag_pipeline_flow.png)

## How It Works
- **User Query:** User sends a question to the FastAPI `/ask` endpoint.
- **Text Chunking (Semantic):** The PDF is split into semantic chunks (e.g., paragraphs) for efficient retrieval.
- **ChromaDB & FAISS Retrieval:** The query is used to retrieve relevant chunks from both ChromaDB and FAISS vector stores.
- **Combine Results:** Results from both stores are combined.
- **Rerank:** All retrieved chunks are reranked by semantic similarity to the query; the top 5 are selected.
- **Validation Node:** Checks if the retrieved content matches the query intent. If not, the query is rephrased using the LLM and retrieval is retried (loop).
- **LLM Generation:** Once validated, the top reranked chunks are sent to the LLM to generate the answer.
- **Final Answer:** The answer is returned to the user.

## How to Run the Project

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Start Ollama (for LLM):**
   ```bash
   ollama serve
   ollama run llama2:7b
   ```
   (Or use your preferred LLM backend.)
3. **Start the FastAPI server:**
   ```bash
   uvicorn app:app --reload
   ```
4. **Ask questions:**
   - Use the `/ask` endpoint (e.g., via Swagger UI at `http://localhost:8000/docs` or with `curl`/Postman).
5. **Logs:**
   - Retrievals from both ChromaDB and FAISS, reranking, validation, and LLM generation are all logged for explainability.

## Requirements & Compatibility
- **Python 3.11 is recommended.**
- The following package versions are required for compatibility:
  - `marshmallow==3.20.1` (required for `environs` and `pymilvus`)
  - `huggingface_hub==0.15.1` (required for `sentence-transformers==2.2.2`)
- If you encounter errors related to `marshmallow` or `huggingface_hub`, ensure these versions are installed (see `requirements.txt`).

### Hugging Face Token (if required)
- Some models from the Hugging Face Hub require authentication with a token (for example, if the model is gated or not fully public).
- If you see errors about authentication or permission denied when downloading models, you need a Hugging Face token.
- To get a token:
  1. Go to https://huggingface.co/settings/tokens and create a new token (with "read" access).
  2. Set the token as an environment variable before running your code:
     ```sh
     export HUGGINGFACE_HUB_TOKEN=your_token_here
     ```
     Or, use the Hugging Face CLI:
     ```sh
     huggingface-cli login
     ```
- The code and libraries will automatically use this token when accessing the Hugging Face Hub.

## Troubleshooting
- **ChromaDB Telemetry Error:**
  - You may see errors like `capture() takes 1 positional argument but 3 were given` from `chromadb.telemetry.product.posthog`. This is harmless and does not affect retrieval or results.
  - To disable ChromaDB telemetry, set the environment variable:
    ```sh
    export CHROMA_TELEMETRY_ENABLED=FALSE
    ```
    Or in Python:
    ```python
    import os
    os.environ["CHROMA_TELEMETRY_ENABLED"] = "FALSE"
    ```
- **ONNX/CoreML Warnings:**
  - On Apple Silicon, you may see warnings about CoreML not supporting certain input dimensions. These are expected and do not affect the pipeline; computation will fall back to CPU as needed.
- **General:**
  - If you see `distutils` errors on macOS, run: `brew install python-setuptools`
  - If you have issues with virtual environments, ensure you are using Python 3.11 and have activated the correct environment.

## Sample RAG Results

See [`Result.txt`](./Result.txt) for example RAG answers. Here are a few samples:

```json
[
  {
    "query": "Lung cancer?",
    "answer": "Lung cancer screening is covered by Medicare for individuals between the ages of 50-77 who have a history of smoking and meet certain criteria. The Part B deductible applies, and you pay nothing for this screening test if your doctor accepts assignment. Before your first lung cancer screening, you'll need to schedule an appointment with a health care provider to discuss the benefits and risks of lung cancer screening to decide if the screening is right for you. Requirements/conditions: * You're between 50-77 years old. * You don't have signs or symptoms of lung cancer (you're asymptomatic). * You've had a history of smoking. * You have a tobacco smoking history of at least 20 'pack years' (an average of one pack—20 cigarettes—per day for 20 years). * You get an order from your doctor.",
    "source_pages": [34, 37, 45],
    "confidence_score": 0.45,
    "chunk_size": 386,
    "total_chunks_used": 5
  },
  {
    "query": "What’s the Medicare drug coverage (Part D) late enrollment penalty?",
    "answer": "The Medicare drug coverage (Part D) late enrollment penalty is calculated by multiplying 1% of the 'national base beneficiary premium' ($36.78 in 2025) by the number of full, uncovered months that you were eligible but didn't have Medicare drug coverage (Part D) and went without other creditable prescription drug coverage. The final amount is rounded to the nearest $0.10 and added to your monthly premium. Requirements or conditions mentioned in the context for calculating the late enrollment penalty include: * How long you didn't have credible prescription drug coverage (Part D) after your Initial Enrollment Period ended * Whether you had other credible prescription drug coverage (Part D) * Your eligibility for Medicare drug coverage (Part D) * The national base beneficiary premium amount ($36.78 in 2025) * The rounding rule to the nearest $0.10 for the final amount.",
    "source_pages": [79, 84, 85],
    "confidence_score": 0.74,
    "chunk_size": 554,
    "total_chunks_used": 5
  },
  {
    "query": "What is Capital of India?",
    "answer": "Not found in context. The question 'What is Capital of India?' does not appear in the provided context.",
    "source_pages": [],
    "confidence_score": 0.0,
    "chunk_size": 0,
    "total_chunks_used": 0
  }
]
```

- The system returns detailed, context-grounded answers for Medicare-related questions.
- If the answer is not found in the context, it returns a clear "Not found in context" response.

## Notes
- The system uses semantic chunking for the PDF.
- Both ChromaDB and FAISS are used for hybrid retrieval.
- The pipeline includes reranking, validation, and LLM-based query rephrasing for robust answer generation. 

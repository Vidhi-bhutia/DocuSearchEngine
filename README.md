# Multi-Document Embedding Search Engine

A lightweight embedding-based search engine over 100-200 text documents with efficient caching, vector search using FAISS, and comprehensive ranking explanations.

## Features

- **Efficient Embedding Generation**: Uses sentence-transformers (all-MiniLM-L6-v2) for high-quality semantic embeddings
- **Smart Caching**: SQLite-based cache prevents recomputing embeddings for unchanged documents
- **Fast Vector Search**: FAISS IndexFlatIP for efficient similarity search with normalized embeddings
- **Ranking Explanations**: Detailed explanations including keyword overlap, semantic similarity scores, and document statistics
- **REST API**: FastAPI-based API with automatic documentation
- **20 Newsgroups Dataset**: Ready-to-use dataset loader with 200 documents

## Project Structure

```
.
├── src/
│   ├── api.py              # FastAPI application with /search endpoint
│   ├── embedder.py         # Embedding generation using sentence-transformers
│   ├── cache_manager.py    # SQLite cache for embeddings
│   ├── preprocessor.py     # Document loading and text cleaning
│   ├── search_engine.py    # FAISS-based vector search with ranking
│   └── load_dataset.py     # 20 Newsgroups dataset loader
├── data/
│   ├── docs/               # Text documents (created by dataset loader)
│   └── cache/              # SQLite cache database
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `sentence-transformers` - For generating semantic embeddings
- `faiss-cpu` - For efficient vector similarity search
- `fastapi` - For REST API
- `uvicorn` - ASGI server
- `scikit-learn` - For 20 Newsgroups dataset
- `numpy` - For numerical operations
- `pydantic` - For request/response validation

### 2. Load Dataset

Load the 20 Newsgroups dataset and save as text files:

```bash
cd src
python load_dataset.py
```

This will download and save 200 documents to `data/docs/`.

## How to Run

### 1. Start the API Server

```bash
cd src
python api.py
```

The API will be available at `http://0.0.0.0:5000`

### 2. Search Documents

**Using curl:**

```bash
curl -X POST "http://0.0.0.0:5000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "quantum physics basics",
    "top_k": 5
  }'
```

**Example Response:**

```json
{
  "query": "quantum physics basics",
  "results": [
    {
      "doc_id": "doc_0014",
      "score": 0.88,
      "preview": "Quantum theory is concerned with the behavior of matter and energy at the molecular, atomic, nuclear, and even smaller microscopic levels...",
      "explanation": {
        "reasons": [
          "High semantic similarity to query",
          "Partial keyword overlap (3 matching terms)"
        ],
        "overlapping_keywords": ["quantum", "physics", "theory"],
        "overlap_ratio": 0.75,
        "overlap_count": 3,
        "document_length": 1523,
        "length_normalization_score": 1.0,
        "semantic_similarity_score": 0.88
      }
    }
  ],
  "total_results": 5
}
```

### 3. API Documentation

Visit `http://0.0.0.0:5000/docs` for interactive API documentation (Swagger UI).

## How Caching Works

The caching system is designed to avoid recomputing embeddings for documents that haven't changed:

1. **Hash-Based Change Detection**: Each document's text is hashed using SHA256
2. **Cache Lookup**: Before generating an embedding, the system checks if:
   - The document ID exists in the cache
   - The cached hash matches the current document hash
3. **Cache Hit**: If both conditions are met, the cached embedding is used
4. **Cache Miss**: If the document is new or has changed, a new embedding is generated and cached
5. **Storage**: Embeddings are stored in SQLite as binary blobs (BLOB type) for efficient storage

**Cache Database Schema:**
```sql
CREATE TABLE embeddings (
    doc_id TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    hash TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    dimension INTEGER NOT NULL
)
```

## How Embedding Generation Works

1. **Model Loading**: On startup, the `sentence-transformers/all-MiniLM-L6-v2` model is loaded
2. **Document Processing**: Each document is:
   - Cleaned (lowercase, HTML removal, whitespace normalization)
   - Hashed for cache lookup
3. **Embedding Generation**: 
   - If cached: Load from SQLite
   - If not cached: Generate using the model and save to cache
4. **Normalization**: All embeddings are L2-normalized for cosine similarity via inner product

## How Search Works

1. **Query Embedding**: The search query is converted to an embedding vector
2. **FAISS Search**: Using IndexFlatIP (inner product), FAISS finds the top-k most similar documents
3. **Ranking**: Results are scored based on cosine similarity (via normalized inner product)
4. **Explanation Generation**: For each result:
   - Keyword overlap is computed (simple word matching)
   - Overlap ratio = (matching keywords) / (total query keywords)
   - Semantic similarity score from FAISS
   - Reasons are generated based on thresholds

## Design Choices

### 1. **Embedding Model: all-MiniLM-L6-v2**
- **Why**: Excellent balance between quality and speed
- **Dimension**: 384 (compact, efficient)
- **Performance**: Fast inference, suitable for real-time search
- **Quality**: Strong performance on semantic similarity tasks

### 2. **Cache: SQLite**
- **Why**: Built-in Python support, no external dependencies
- **Benefits**: ACID compliance, efficient binary storage (BLOB)
- **Trade-off**: Single-file database, easy to backup/transfer

### 3. **Vector Index: FAISS IndexFlatIP**
- **Why**: Exact search, no approximation errors
- **Method**: Inner product (equivalent to cosine similarity with normalized vectors)
- **Scalability**: Works well for 100-200 documents; for larger datasets, consider IndexIVFFlat

### 4. **Normalization: L2 Normalization**
- **Why**: Converts cosine similarity to inner product for faster FAISS search
- **Formula**: `embedding / ||embedding||_2`
- **Benefit**: IndexFlatIP is faster than custom cosine similarity

### 5. **Ranking Explanation**
- **Hybrid Approach**: Combines semantic similarity (embeddings) with keyword overlap (BM25-like)
- **Interpretability**: Users understand why a document was retrieved
- **Components**:
  - Semantic similarity score (from FAISS)
  - Keyword overlap (simple word matching)
  - Document length normalization
  - Human-readable reasons

### 6. **Modular Architecture**
- **Separation of Concerns**: Each module has a single responsibility
- **Testability**: Easy to unit test individual components
- **Maintainability**: Changes to one component don't affect others

## API Endpoints

### `POST /search`
Search for documents similar to a query.

**Request Body:**
```json
{
  "query": "string",
  "top_k": 5
}
```

**Response:**
```json
{
  "query": "string",
  "results": [
    {
      "doc_id": "string",
      "score": 0.0,
      "preview": "string",
      "explanation": {
        "reasons": ["string"],
        "overlapping_keywords": ["string"],
        "overlap_ratio": 0.0,
        "overlap_count": 0,
        "document_length": 0,
        "length_normalization_score": 0.0,
        "semantic_similarity_score": 0.0
      }
    }
  ],
  "total_results": 0
}
```

### `GET /health`
Check API health and readiness.

### `GET /`
API information and available endpoints.

### `GET /docs`
Interactive API documentation (Swagger UI).

## Performance Notes

- **First Run**: Slower due to model loading and embedding generation
- **Subsequent Runs**: Fast due to caching (only new/changed documents are processed)
- **Search Latency**: ~10-50ms for 200 documents
- **Scalability**: For >10,000 documents, consider IndexIVFFlat or IndexHNSWFlat

## Extending the System

### Add Your Own Documents

1. Place `.txt` files in `data/docs/`
2. Restart the API server
3. Embeddings will be generated automatically

### Change Embedding Model

In `src/embedder.py`, modify the `model_name` parameter:
```python
embedder = EmbeddingGenerator(model_name='your-model-name')
```

### Adjust Search Parameters

- **top_k**: Change in search request (max 20)
- **Normalization**: Modify `normalize` parameter in `embedder.py`
- **Preview length**: Change `max_chars` in `search_engine.py`

## Troubleshooting

### Issue: "No documents found"
- **Solution**: Run `python src/load_dataset.py` to download the dataset

### Issue: "Search engine not initialized"
- **Solution**: Wait a few seconds for startup to complete, then retry

### Issue: Slow first search
- **Solution**: Normal behavior - model loading and initial embedding generation take time

### Issue: Cache not working
- **Solution**: Check that `data/cache/` directory exists and is writable

## License

This project is created for educational purposes as part of the CodeAtRandom AI Engineer Intern Assignment.

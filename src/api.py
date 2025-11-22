from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from contextlib import asynccontextmanager
import uvicorn

from embedder import EmbeddingGenerator
from cache_manager import EmbeddingCache
from preprocessor import DocumentPreprocessor
from search_engine import VectorSearchEngine

import numpy as np


search_engine = None
embedder = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global search_engine, embedder
    
    print("Starting up search engine...")
    
    preprocessor = DocumentPreprocessor(docs_dir="data/docs")
    cache = EmbeddingCache(cache_db_path="data/cache/embeddings.db")
    embedder = EmbeddingGenerator()
    
    print("Loading documents...")
    documents = preprocessor.load_all_documents()
    
    if documents:
        print(f"Loaded {len(documents)} documents")
        
        embeddings_list = []
        doc_ids = []
        
        cache_hits = 0
        cache_misses = 0
        
        print("Generating/loading embeddings...")
        for doc in documents:
            doc_id = doc['doc_id']
            doc_hash = doc['hash']
            
            cached_embedding = cache.get_embedding(doc_id, doc_hash)
            
            if cached_embedding is not None:
                embedding = cached_embedding
                cache_hits += 1
            else:
                embedding = embedder.embed_text(doc['clean_text'], normalize=True)
                cache.save_embedding(doc_id, embedding, doc_hash)
                cache_misses += 1
            
            embeddings_list.append(embedding)
            doc_ids.append(doc_id)
        
        print(f"Cache hits: {cache_hits}, Cache misses: {cache_misses}")
        
        embeddings_array = np.array(embeddings_list)
        
        search_engine = VectorSearchEngine(dimension=embedder.get_embedding_dimension())
        search_engine.build_index(embeddings_array, doc_ids, documents)
        
        print("Search engine ready!")
    else:
        print("WARNING: No documents found! Search will not work.")
    
    yield
    
    print("Shutting down search engine...")


app = FastAPI(
    title="Multi-Document Embedding Search Engine",
    description="Semantic search over documents using sentence embeddings and FAISS",
    version="1.0.0",
    lifespan=lifespan
)


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")


class SearchResult(BaseModel):
    doc_id: str
    score: float
    preview: str
    explanation: dict


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Multi-Document Embedding Search Engine API",
        "endpoints": {
            "/search": "POST - Search documents",
            "/docs": "GET - API documentation",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if search_engine is None or embedder is None:
        return {
            "status": "initializing",
            "ready": False
        }
    
    return {
        "status": "healthy",
        "ready": True,
        "indexed_documents": len(search_engine.doc_ids) if search_engine else 0
    }


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search documents using semantic similarity.
    
    Args:
        request: SearchRequest with query and top_k
        
    Returns:
        SearchResponse with ranked results and explanations
    """
    if search_engine is None or embedder is None:
        raise HTTPException(
            status_code=503,
            detail="Search engine not initialized. Please wait for startup to complete."
        )
    
    if not search_engine.doc_ids:
        raise HTTPException(
            status_code=404,
            detail="No documents available for search"
        )
    
    try:
        query_embedding = embedder.embed_text(request.query, normalize=True)
        
        search_results = search_engine.search(query_embedding, top_k=request.top_k)
        
        results = []
        for doc_id, score in search_results:
            preview = search_engine.get_document_preview(doc_id, max_chars=200)
            explanation = search_engine.get_ranking_explanation(
                request.query, doc_id, score
            )
            
            results.append(SearchResult(
                doc_id=doc_id,
                score=round(score, 4),
                preview=preview,
                explanation=explanation
            ))
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)

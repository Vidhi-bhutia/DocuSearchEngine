import faiss
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import re


class VectorSearchEngine:
    """FAISS-based vector search engine with ranking explanations."""
    
    def __init__(self, dimension: int):
        """
        Initialize the search engine.
        
        Args:
            dimension: Dimensionality of embeddings
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.doc_ids = []
        self.documents = {}
    
    def build_index(self, embeddings: np.ndarray, doc_ids: List[str], documents: List[Dict]):
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: Array of embeddings (num_docs x dimension)
            doc_ids: List of document IDs
            documents: List of document dictionaries
        """
        if len(embeddings) != len(doc_ids):
            raise ValueError("Number of embeddings must match number of doc_ids")
        
        embeddings = embeddings.astype('float32')
        
        if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5):
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
        
        self.index.add(embeddings)
        self.doc_ids = doc_ids
        
        self.documents = {doc['doc_id']: doc for doc in documents}
        
        print(f"Built FAISS index with {len(doc_ids)} documents")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of (doc_id, score) tuples
        """
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.doc_ids)))
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.doc_ids):
                results.append((self.doc_ids[idx], float(score)))
        
        return results
    
    def extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """
        Extract keywords from text (simple word-based approach).
        
        Args:
            text: Input text
            min_length: Minimum word length to consider
            
        Returns:
            List of keywords
        """
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if len(w) >= min_length]
        return keywords
    
    def compute_keyword_overlap(self, query_text: str, doc_text: str) -> Dict:
        """
        Compute keyword overlap between query and document.
        
        Args:
            query_text: Query text
            doc_text: Document text
            
        Returns:
            Dictionary with overlap statistics
        """
        query_keywords = set(self.extract_keywords(query_text))
        doc_keywords = set(self.extract_keywords(doc_text))
        
        if not query_keywords:
            return {
                'overlapping_keywords': [],
                'overlap_count': 0,
                'overlap_ratio': 0.0
            }
        
        overlap = query_keywords & doc_keywords
        overlap_ratio = len(overlap) / len(query_keywords)
        
        return {
            'overlapping_keywords': sorted(list(overlap)),
            'overlap_count': len(overlap),
            'overlap_ratio': round(overlap_ratio, 3)
        }
    
    def get_ranking_explanation(self, query_text: str, doc_id: str, score: float) -> Dict:
        """
        Generate explanation for why a document was ranked highly.
        
        Args:
            query_text: Original query text
            doc_id: Document ID
            score: Similarity score
            
        Returns:
            Dictionary with ranking explanation
        """
        doc = self.documents.get(doc_id, {})
        doc_text = doc.get('clean_text', '')
        doc_length = doc.get('length', 0)
        
        overlap_info = self.compute_keyword_overlap(query_text, doc_text)
        
        reasons = []
        if score > 0.7:
            reasons.append("High semantic similarity to query")
        elif score > 0.5:
            reasons.append("Moderate semantic similarity to query")
        else:
            reasons.append("Low semantic similarity to query")
        
        if overlap_info['overlap_ratio'] > 0.5:
            reasons.append(f"Strong keyword overlap ({overlap_info['overlap_count']} matching terms)")
        elif overlap_info['overlap_count'] > 0:
            reasons.append(f"Partial keyword overlap ({overlap_info['overlap_count']} matching terms)")
        
        length_norm_score = min(1.0, doc_length / 1000)
        
        return {
            'reasons': reasons,
            'overlapping_keywords': overlap_info['overlapping_keywords'],
            'overlap_ratio': overlap_info['overlap_ratio'],
            'overlap_count': overlap_info['overlap_count'],
            'document_length': doc_length,
            'length_normalization_score': round(length_norm_score, 3),
            'semantic_similarity_score': round(float(score), 3)
        }
    
    def get_document_preview(self, doc_id: str, max_chars: int = 200) -> str:
        """
        Get a preview of the document text.
        
        Args:
            doc_id: Document ID
            max_chars: Maximum characters to return
            
        Returns:
            Preview text
        """
        doc = self.documents.get(doc_id, {})
        text = doc.get('clean_text', '')
        
        if len(text) <= max_chars:
            return text
        
        return text[:max_chars] + "..."

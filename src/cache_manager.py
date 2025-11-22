import sqlite3
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List


class EmbeddingCache:
    """SQLite-based cache for document embeddings."""
    
    def __init__(self, cache_db_path: str = "data/cache/embeddings.db"):
        self.cache_db_path = Path(cache_db_path)
        self.cache_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database with embeddings table."""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                doc_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                hash TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                dimension INTEGER NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_embedding(self, doc_id: str, current_hash: str) -> Optional[np.ndarray]:
        """
        Retrieve cached embedding if hash matches.
        
        Args:
            doc_id: Document identifier
            current_hash: Current hash of the document text
            
        Returns:
            Cached embedding array if valid, None otherwise
        """
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT embedding, hash FROM embeddings WHERE doc_id = ?',
            (doc_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result is None:
            return None
        
        cached_embedding_blob, cached_hash = result
        
        if cached_hash != current_hash:
            return None
        
        embedding = np.frombuffer(cached_embedding_blob, dtype=np.float32)
        return embedding
    
    def save_embedding(self, doc_id: str, embedding: np.ndarray, doc_hash: str):
        """
        Save embedding to cache.
        
        Args:
            doc_id: Document identifier
            embedding: Embedding vector
            doc_hash: Hash of the document text
        """
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        embedding_blob = embedding.astype(np.float32).tobytes()
        timestamp = datetime.utcnow().isoformat()
        dimension = len(embedding)
        
        cursor.execute('''
            INSERT OR REPLACE INTO embeddings 
            (doc_id, embedding, hash, updated_at, dimension)
            VALUES (?, ?, ?, ?, ?)
        ''', (doc_id, embedding_blob, doc_hash, timestamp, dimension))
        
        conn.commit()
        conn.close()
    
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Retrieve all cached embeddings.
        
        Returns:
            Dictionary mapping doc_id to embedding array
        """
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT doc_id, embedding FROM embeddings')
        results = cursor.fetchall()
        conn.close()
        
        embeddings = {}
        for doc_id, embedding_blob in results:
            embeddings[doc_id] = np.frombuffer(embedding_blob, dtype=np.float32)
        
        return embeddings
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about the cache."""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*), AVG(dimension) FROM embeddings')
        count, avg_dim = cursor.fetchone()
        
        cursor.execute('SELECT MIN(updated_at), MAX(updated_at) FROM embeddings')
        min_date, max_date = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_cached': count or 0,
            'avg_dimension': int(avg_dim) if avg_dim else 0,
            'oldest_entry': min_date,
            'newest_entry': max_date
        }
    
    def clear_cache(self):
        """Clear all cached embeddings."""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM embeddings')
        conn.commit()
        conn.close()

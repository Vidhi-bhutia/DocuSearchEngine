from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
from tqdm import tqdm


class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.dimension}")
    
    def embed_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            normalize: Whether to L2-normalize the embedding
            
        Returns:
            Embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        if normalize:
            embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def embed_batch(self, texts: List[str], normalize: bool = True, 
                   batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            normalize: Whether to L2-normalize embeddings
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings (num_texts x embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=show_progress
        )
        
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        if self.dimension is None:
            return 384
        return int(self.dimension)

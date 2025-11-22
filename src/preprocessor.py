import re
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional


class DocumentPreprocessor:
    """Handles document loading, cleaning, and metadata extraction."""
    
    def __init__(self, docs_dir: str = "data/docs"):
        self.docs_dir = Path(docs_dir)
        
    def clean_text(self, text: str) -> str:
        """
        Clean text by:
        - Converting to lowercase
        - Removing HTML tags
        - Removing extra whitespace
        """
        text = re.sub(r'<[^>]+>', '', text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def compute_hash(self, text: str) -> str:
        """Compute SHA256 hash of the text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def load_document(self, file_path: Path) -> Optional[Dict]:
        """
        Load a single document and extract metadata.
        
        Returns:
            Dict with keys: doc_id, text, clean_text, hash, length, file_path
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = f.read()
            
            clean_text = self.clean_text(raw_text)
            
            return {
                'doc_id': file_path.stem,
                'text': raw_text,
                'clean_text': clean_text,
                'hash': self.compute_hash(clean_text),
                'length': len(clean_text),
                'file_path': str(file_path)
            }
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def load_all_documents(self) -> List[Dict]:
        """
        Load all .txt files from the documents directory.
        
        Returns:
            List of document dictionaries
        """
        documents = []
        
        if not self.docs_dir.exists():
            print(f"Directory {self.docs_dir} does not exist!")
            return documents
        
        txt_files = list(self.docs_dir.glob('*.txt'))
        print(f"Found {len(txt_files)} text files")
        
        for file_path in txt_files:
            doc = self.load_document(file_path)
            if doc:
                documents.append(doc)
        
        print(f"Successfully loaded {len(documents)} documents")
        return documents
    
    def get_document_stats(self, documents: List[Dict]) -> Dict:
        """Get statistics about the document collection."""
        if not documents:
            return {}
        
        lengths = [doc['length'] for doc in documents]
        return {
            'total_documents': len(documents),
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths)
        }

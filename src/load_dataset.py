from sklearn.datasets import fetch_20newsgroups
from pathlib import Path
import re


def clean_filename(text: str) -> str:
    """Create a clean filename from text."""
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '_', text)
    return text[:50]


def load_20newsgroups_dataset(output_dir: str = "data/docs", max_docs: int = 200):
    """
    Load 20 Newsgroups dataset and save as text files.
    
    Args:
        output_dir: Directory to save text files
        max_docs: Maximum number of documents to save
    """
    print("Fetching 20 Newsgroups dataset...")
    dataset = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving up to {max_docs} documents to {output_dir}")
    
    docs_saved = 0
    for idx, (text, category_id) in enumerate(zip(dataset.data, dataset.target)):
        if docs_saved >= max_docs:
            break
        
        if len(text.strip()) < 50:
            continue
        
        category_name = dataset.target_names[category_id]
        
        doc_filename = f"doc_{docs_saved:04d}_{clean_filename(category_name)}.txt"
        doc_path = output_path / doc_filename
        
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        docs_saved += 1
    
    print(f"Successfully saved {docs_saved} documents")
    return docs_saved


if __name__ == "__main__":
    load_20newsgroups_dataset(max_docs=200)

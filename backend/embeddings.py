import os
import json
import logging
from tqdm import tqdm
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
INDEX_DIR = os.path.join(DATA_DIR, "index")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")

# Ensure index directory exists
os.makedirs(INDEX_DIR, exist_ok=True)

def load_chunks(chunks_file):
    """Load preprocessed chunks from JSON file"""
    if not os.path.exists(chunks_file):
        logger.warning(f"Chunks file not found: {chunks_file}")
        return []
    
    try:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        logger.info(f"Loaded {len(chunks)} chunks from {chunks_file}")
        return chunks
    except Exception as e:
        logger.error(f"Error loading chunks file: {str(e)}")
        return []

def create_documents(chunks):
    """Convert chunks to LangChain Document objects with progress bar"""
    documents = []
    for chunk in tqdm(chunks, desc="Creating documents", unit="chunk"):
        try:
            doc = Document(
                page_content=chunk["text"],
                metadata={
                    "source": chunk["source"],
                    "page": chunk["page"]
                }
            )
            documents.append(doc)
        except Exception as e:
            logger.error(f"Error creating document from chunk: {str(e)}")
    return documents

class ProgressCallback:
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="Creating embeddings", unit="doc")
    
    def update(self, _):
        self.pbar.update(1)

def batch_documents(documents: List[Document], batch_size: int = 32):
    """Split documents into batches"""
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]

class EmbeddingProgress:
    """Custom progress tracker for embeddings"""
    def __init__(self, total):
        self.progress_bar = tqdm(total=total, desc="Creating embeddings", unit="doc")
        
    def __call__(self, *args, **kwargs):
        """Update progress bar"""
        self.progress_bar.update(1)
    
    def close(self):
        """Close progress bar"""
        self.progress_bar.close()

def create_embeddings(
    chunks_file=CHUNKS_FILE,
    index_dir=INDEX_DIR,
    documents=None,
    batch_size=32
):
    """Create embeddings with progress tracking"""
    if documents is None:
        # Load and create documents with progress bar
        chunks = load_chunks(chunks_file)
        documents = []
        for chunk in tqdm(chunks, desc="Creating documents", unit="chunk"):
            try:
                doc = Document(
                    page_content=chunk["text"],
                    metadata={
                        "source": chunk["source"],
                        "page": chunk["page"]
                    }
                )
                documents.append(doc)
            except Exception as e:
                logger.error(f"Error creating document from chunk: {str(e)}")
    
    if not documents:
        logger.warning("No documents to embed")
        return None
    
    logger.info(f"Creating embeddings for {len(documents)} documents")
    
    try:
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Process documents in batches with progress tracking
        progress = EmbeddingProgress(len(documents))
        vectorstore = None
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            if vectorstore is None:
                # Create initial vectorstore
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                # Add documents to existing vectorstore
                vectorstore.add_documents(batch)
            progress(len(batch))
        
        progress.close()
        
        # Save vector store
        os.makedirs(index_dir, exist_ok=True)
        vectorstore.save_local(index_dir)
        logger.info(f"Saved vector store to {index_dir}")
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        return None

def main():
    """Main function to create embeddings"""
    logger.info("Starting embedding creation process...")
    vectorstore = create_embeddings()
    if vectorstore:
        logger.info("Successfully created and saved embeddings")
    else:
        logger.error("Failed to create embeddings")

if __name__ == "__main__":
    main()
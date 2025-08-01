import os
import json
import pdfplumber
import logging
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_pdfs(
    pdf_dir=os.path.join(os.path.dirname(__file__), "..", "data", "pdfs"),
    output_file=os.path.join(os.path.dirname(__file__), "..", "data", "chunks.json")
):
    """
    Process PDF files and extract text chunks
    Args:
        pdf_dir: Directory containing PDF files
        output_file: Output JSON file path
    Returns:
        List of Document objects for embedding
    """
    chunks = []
    documents = []
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Check if directory exists and has PDF files
    if not os.path.exists(pdf_dir):
        logger.warning(f"PDF directory does not exist: {pdf_dir}")
        os.makedirs(pdf_dir, exist_ok=True)
        return documents
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        return documents
    
    for filename in pdf_files:
        filepath = os.path.join(pdf_dir, filename)
        logger.info(f"Processing {filename}...")
        try:
            with pdfplumber.open(filepath) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        for i in range(0, len(text), 500):
                            chunk = text[i:i+500]
                            chunk_data = {
                                "text": chunk,
                                "source": filename,
                                "page": page_num
                            }
                            chunks.append(chunk_data)
                            
                            # Create Document object for embedding
                            doc = Document(
                                page_content=chunk,
                                metadata={"source": filename, "page": page_num}
                            )
                            documents.append(doc)
                    else:
                        logger.warning(f"No text extracted from {filename}, page {page_num}")
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
    
    if not chunks:
        logger.warning("No chunks extracted from any PDFs.")
        return documents
    
    logger.info(f"Saving {len(chunks)} chunks to {output_file}")
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    return documents

if __name__ == "__main__":
    preprocess_pdfs()